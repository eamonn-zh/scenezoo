import os

XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", "~/.cache")
SCENEZOO_CACHE_DIR = os.path.expanduser(
    os.getenv("SCENEZOO_CACHE_DIR", os.path.join(XDG_CACHE_HOME, "scenezoo"))
)
