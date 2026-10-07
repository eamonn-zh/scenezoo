from __future__ import annotations

import fsspec
import pytest

from scenezoo.io.cache import cached_file_path, open_cached_file


def _remote(name: str, payload: bytes) -> str:
    url = f"memory://scenezoo-test/{name}"
    with fsspec.open(url, "wb") as handle:
        handle.write(payload)
    return url


def _forget(url: str) -> None:
    fsspec.filesystem("memory").rm(url)


def test_remote_files_are_downloaded_once_and_reused_offline(tmp_path):
    url = _remote("labels.tsv", b"id\tname\n1\tchair\n")
    with open_cached_file(url, "r", cache_dir=tmp_path) as handle:
        assert handle.read().splitlines()[1] == "1\tchair"
    assert cached_file_path(url, tmp_path).is_file()
    _forget(url)
    with open_cached_file(url, "rb", cache_dir=tmp_path, offline=True) as handle:
        assert handle.read().startswith(b"id\t")
    with open_cached_file(url, "rb", cache_dir=tmp_path) as handle:
        assert handle.read().startswith(b"id\t")


def test_offline_mode_never_downloads(tmp_path):
    url = _remote("missing.txt", b"data")
    with pytest.raises(FileNotFoundError, match="not cached"):
        with open_cached_file(url, cache_dir=tmp_path, offline=True):
            pass
    assert not cached_file_path(url, tmp_path).exists()
    _forget(url)


def test_offline_mode_reads_legacy_fsspec_cache(tmp_path):
    url = _remote("legacy.txt", b"legacy")
    with fsspec.open(
        f"filecache::{url}", "rb", filecache={"cache_storage": str(tmp_path)}
    ) as handle:
        handle.read()
    _forget(url)
    with open_cached_file(url, "rb", cache_dir=tmp_path, offline=True) as handle:
        assert handle.read() == b"legacy"


def test_local_paths_bypass_the_cache(tmp_path):
    path = tmp_path / "local.txt"
    path.write_text("local")
    with open_cached_file(f"file://{path}", "r", cache_dir=tmp_path / "c") as handle:
        assert handle.read() == "local"
    assert not (tmp_path / "c").exists()


def test_gzip_encoded_http_responses_are_cached_completely(tmp_path):
    """Imitate raw.githubusercontent.com, whose byte ranges index the gzip
    encoding. Ranged reads sized from the plain length truncate small files, so
    the cache must download responses whole."""

    import gzip
    import random
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # A small, incompressible file: its gzip encoding is larger than itself,
    # so ranges sized from the plain length cut the compressed stream short.
    payload = random.Random(0).randbytes(132)
    assert len(gzip.compress(payload)) > len(payload)

    class GzipHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _body(self):
            gzipped = "gzip" in self.headers.get("Accept-Encoding", "")
            return (gzip.compress(payload) if gzipped else payload), gzipped

        def _headers(self, status, data, gzipped, content_range=None):
            self.send_response(status)
            if gzipped:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if content_range:
                self.send_header("Content-Range", content_range)
            self.end_headers()

        def do_HEAD(self):
            self._headers(200, *self._body())

        def do_GET(self):
            body, gzipped = self._body()
            match = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            if match is None:
                self._headers(200, body, gzipped)
                self.wfile.write(body)
                return
            # Like GitHub, ranges index the encoded (compressed) bytes.
            start = int(match.group(1))
            end = min(int(match.group(2) or len(body) - 1), len(body) - 1)
            part = body[start : end + 1]
            self._headers(206, part, gzipped, f"bytes {start}-{end}/{len(body)}")
            self.wfile.write(part)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), GzipHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/splits.txt"
        with open_cached_file(url, "rb", cache_dir=tmp_path) as handle:
            assert handle.read() == payload
    finally:
        server.shutdown()
