# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Port the R package `strat` (stratification index, https://cran.r-project.org/web/packages/strat/) to Python and publish it on PyPI. The stratification index metric is described in https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X197600200002. The full task spec is in `spec/stratindex.md` (in Russian).

- Target Python version: 3.12
- End deliverable: a package published to PyPI

The repository is currently greenfield — the spec exists but no source code has been written yet. When implementing, consult the R `strat` package source on CRAN as the reference implementation.

## Environment

- `venv` is a symlink to a pyenv virtualenv (`projects_stratindex`, Python 3.12.8); `.python-version` activates it automatically under pyenv. Use `venv/bin/python` / `venv/bin/pip` explicitly if pyenv shims aren't active.
- `uv` is installed in the venv. Dependencies follow the pip-tools convention: declare direct dependencies in `requirements.in`, compile to a pinned `requirements.txt` (e.g. `uv pip compile requirements.in -o requirements.txt`), then install with `uv pip sync requirements.txt`.
