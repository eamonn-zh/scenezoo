from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "SceneZoo"
author = "SceneZoo contributors"
copyright = "2026, SceneZoo contributors"
# Single source of truth: ``__version__`` in the package.
release = re.search(
    r'__version__ = "(.+)"',
    (Path(__file__).resolve().parents[1] / "src/scenezoo/__init__.py").read_text(),
)[1]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
# Heavy runtime dependencies are not needed to render the API reference.
autodoc_mock_imports = [
    "av",
    "cachetools",
    "fsspec",
    "lz4",
    "open3d",
    "pandas",
    "plyfile",
    "scipy",
]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "piccolo_theme"
html_title = "SceneZoo"
html_theme_options = {
    "source_url": "https://github.com/eamonn-zh/scenezoo",
    "source_icon": "github",
    "show_theme_credit": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.png"
# Copy commands and code without prompts or console output.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
master_doc = "index"
