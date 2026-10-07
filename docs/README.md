# Building the documentation

For maintainers and deployment only: install the build requirements, then build
with warnings treated as errors. Readers use the hosted documentation and do
not need these dependencies. Run from the repository root:

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

The API pages import `scenezoo` but never open a dataset, so a documentation build
does not require downloaded dataset files.
