"""Structured, manually invoked dataset layout checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence


CheckSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class CheckIssue:
    """One actionable finding produced by :meth:`Dataset.check`."""

    severity: CheckSeverity
    code: str
    message: str
    path: Path | None = None
    expected: str | None = None
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetCheckReport:
    """The structured result of an explicit dataset layout check."""

    dataset: str
    root_dir: Path
    issues: tuple[CheckIssue, ...] = ()
    stats: dict[str, int | str] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[CheckIssue, ...]:
        """Findings that prevent the requested operations from working."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[CheckIssue, ...]:
        """Findings that do not block use, such as an intentional subset."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        """Whether no error-level findings were produced."""

        return not self.errors

    def format(self) -> str:
        """Render a detailed terminal-friendly report."""

        if self.errors:
            status = "FAILED"
        elif self.warnings:
            status = "PASSED WITH WARNINGS"
        else:
            status = "PASSED"
        lines = [f"{self.dataset} dataset check: {status}", f"Root: {self.root_dir}"]
        if self.stats:
            lines.append("Summary:")
            for key, value in self.stats.items():
                lines.append(f"  - {key.replace('_', ' ')}: {value}")
        if not self.issues:
            lines.append("No layout problems were found.")
            return "\n".join(lines)

        lines.append("Findings:")
        for issue in self.issues:
            lines.append(f"  {issue.severity.upper()} [{issue.code}] {issue.message}")
            if issue.path is not None:
                lines.append(f"    Path: {issue.path}")
            if issue.expected:
                lines.append(f"    Expected: {issue.expected}")
            if issue.hint:
                lines.append(f"    Suggested action: {issue.hint}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        """Raise :class:`DatasetCheckError` when this report has errors."""

        if self.errors:
            from .base import DatasetCheckError

            raise DatasetCheckError(self)

    def __str__(self) -> str:
        return self.format()


class CheckBuilder:
    """Collects findings for :meth:`Dataset._check` implementations.

    Custom adapters use it the same way as built-in ones: add findings with
    :meth:`error`, :meth:`warning`, or :meth:`info`, fill :attr:`stats`, and
    return :meth:`finish`.
    """

    def __init__(self, dataset: str, root_dir: Path) -> None:
        self.dataset = dataset
        self.root_dir = root_dir
        self.issues: list[CheckIssue] = []
        self.stats: dict[str, int | str] = {}

    def add(
        self,
        severity: CheckSeverity,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        expected: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Record one finding with an optional path, expected layout, and hint."""

        self.issues.append(CheckIssue(severity, code, message, path, expected, hint))

    def error(self, code: str, message: str, **details) -> None:
        self.add("error", code, message, **details)

    def warning(self, code: str, message: str, **details) -> None:
        self.add("warning", code, message, **details)

    def info(self, code: str, message: str, **details) -> None:
        self.add("info", code, message, **details)

    def require_directory(
        self,
        path: Path,
        *,
        code: str,
        message: str,
        expected: str,
        hint: str | None = None,
    ) -> bool:
        """Report an error unless ``path`` is a directory; return whether it is."""

        if path.is_dir():
            return True
        self.error(
            code,
            message,
            path=path,
            expected=expected,
            hint=hint,
        )
        return False

    def missing_paths(
        self,
        paths: Iterable[Path],
        *,
        code: str,
        label: str,
        expected: str,
        hint: str | None = None,
        severity: CheckSeverity = "error",
        preview: int = 5,
    ) -> None:
        """Summarize missing ``paths`` as one finding with a few examples."""

        missing = list(paths)
        if not missing:
            return
        examples = ", ".join(str(path) for path in missing[:preview])
        if len(missing) > preview:
            examples += f", ... (+{len(missing) - preview} more)"
        self.add(
            severity,
            code,
            f"{len(missing)} {label} missing. Examples: {examples}",
            expected=expected,
            hint=hint,
        )

    def finish(self) -> DatasetCheckReport:
        """Return the immutable report of everything recorded so far."""

        return DatasetCheckReport(
            dataset=self.dataset,
            root_dir=self.root_dir,
            issues=tuple(self.issues),
            stats=dict(self.stats),
        )


def normalize_sample_ids(sample_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    """Validate requested IDs and drop duplicates while keeping their order."""

    if sample_ids is None:
        return None
    if isinstance(sample_ids, (str, bytes)):
        raise TypeError("sample_ids must be a sequence of IDs, not one string.")
    result = tuple(dict.fromkeys(str(value) for value in sample_ids))
    if any(not value for value in result):
        raise ValueError("sample_ids cannot contain empty IDs.")
    return result


def check_scene_coverage(
    builder: CheckBuilder,
    *,
    present_ids: Iterable[str],
    expected_ids: Iterable[str] | None,
    sample_ids: Sequence[str] | None,
    require_complete: bool,
    expected_layout: str,
) -> tuple[str, ...]:
    """Report missing samples and return the present IDs to inspect deeply."""

    present = set(present_ids)
    requested = normalize_sample_ids(sample_ids)
    builder.stats["scenes_found"] = len(present)

    if requested is not None:
        missing = [sample_id for sample_id in requested if sample_id not in present]
        if missing:
            builder.error(
                "missing-requested-scenes",
                f"Requested scene IDs are missing: {', '.join(missing[:10])}",
                expected=expected_layout,
                hint="Download or extract these scenes, or correct sample_ids/root_dir.",
            )
        builder.stats["scenes_requested"] = len(requested)
        return tuple(sample_id for sample_id in requested if sample_id in present)

    if expected_ids is not None:
        expected = set(expected_ids)
        missing = sorted(expected - present)
        builder.stats["scenes_expected"] = len(expected)
        builder.stats["scenes_missing"] = len(missing)
        if missing:
            severity: CheckSeverity = "error" if require_complete else "warning"
            examples = ", ".join(missing[:10])
            if len(missing) > 10:
                examples += f", ... (+{len(missing) - 10} more)"
            builder.add(
                severity,
                "incomplete-scene-set",
                f"{len(missing)} expected scenes are absent. Examples: {examples}",
                expected=expected_layout,
                hint=(
                    "Download/extract the missing scenes, or leave require_complete=False "
                    "when this is an intentional subset."
                ),
            )
    return tuple(sorted(present))


def load_expected_ids(dataset, builder: CheckBuilder) -> set[str] | None:
    """Load all declared splits without turning metadata failures into crashes."""

    try:
        return {sample_id for values in dataset.splits.values() for sample_id in values}
    except Exception as exc:  # The report must explain inaccessible split metadata.
        builder.warning(
            "split-metadata-unavailable",
            f"Could not load split metadata: {type(exc).__name__}: {exc}",
            hint=(
                "Check the metadata path/network cache. In offline mode, download "
                "the split metadata before requesting a completeness check."
            ),
        )
        return None


def check_required_scene_paths(
    builder: CheckBuilder,
    scene_ids: Iterable[str],
    requirements: Mapping[str, Callable[[str], Path]],
    *,
    expected: str,
    hint: str,
    severity: CheckSeverity = "error",
) -> None:
    """Aggregate missing per-scene files into one finding per requirement."""

    scene_ids = tuple(scene_ids)
    for label, resolve in requirements.items():
        builder.missing_paths(
            (
                path
                for sample_id in scene_ids
                if not (path := resolve(sample_id)).is_file()
            ),
            code=f"missing-{label.replace('_', '-')}",
            label=f"required {label.replace('_', ' ')} files",
            expected=expected,
            hint=hint,
            severity=severity,
        )


def archive_candidates(root: Path) -> tuple[Path, ...]:
    """Return archive-looking files directly under a dataset root."""

    if not root.is_dir():
        return ()
    suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")
    return tuple(
        path
        for path in root.iterdir()
        if path.is_file() and path.name.lower().endswith(suffixes)
    )
