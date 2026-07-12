# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`stratindex` — Python port of the R package `strat` (https://cran.r-project.org/package=strat), implementing the nonparametric stratification index of Zhou (2012). Published on PyPI as `stratindex`. License is GPL-3.0-or-later (the port is a derivative of the GPL R package — do not relicense). Task spec and progress checklist live in `spec/stratindex.md` (in Russian); update the checklist when completing stages.

## Environment & commands

- `venv` is a symlink to a pyenv virtualenv (Python 3.12.8). Use `venv/bin/python` / `venv/bin/uv` explicitly if pyenv shims aren't active.
- Dependencies: declare in `requirements.in`, then `venv/bin/uv pip compile requirements.in -o requirements.txt` and `venv/bin/uv pip sync requirements.txt`. Runtime dependency is numpy only; everything else in requirements.in is dev tooling.
- Install for development: `venv/bin/uv pip install -e .`
- Tests: `venv/bin/python -m pytest -q` (single test: `venv/bin/python -m pytest tests/test_strat.py::test_hand_computed_example -q`). The suite takes ~30 s; the slow part is the full-dataset R-golden cross-checks.
- Lint/format: `venv/bin/ruff check src tests scripts` and `venv/bin/ruff format src tests scripts` (CI enforces `ruff format --check`).

## Architecture

- `src/stratindex/_utils.py` — `clean()` (validation, complete cases, weight normalization to sum n) and `wtd_rank()` (port of `Hmisc::wtd.rank(normwt=TRUE)`, weighted midranks). Percentile ranks of equal outcomes are exactly equal by construction (unique-value mapping, not interpolation) — the kernel's tie-skipping depends on this.
- `src/stratindex/_kernel.py` — blocked NumPy ports of the C++ pairwise kernels (`strat_cpp`, `strat_cpp_by`). Inputs must be pre-sorted by `r`. Between-group sums use element-wise differences, not sum-of-sums subtraction, to keep exact zeros exact.
- `src/stratindex/core.py` — public `strat()` / `srank()`; `results.py` — result dataclasses whose `__str__` mirrors the R print methods; `datasets.py` — bundled `cpsmarch2015` loader.
- `tests/conftest.py` holds naive loop-based reference implementations translated literally from the R/C++ sources; kernel tests compare against them across block sizes.

## Cross-validation against R

`tests/data/r_golden.json` holds golden values produced by the original CRAN package; `tests/test_r_golden.py` compares against it (strata/group tables matched by label — R orders by factor levels, Python by sorted unique values). Regenerate with:
`docker run --rm -v "$PWD":/work rocker/r2u:jammy Rscript /work/scripts/r_golden.R /work/tests/data/r_golden.json`
CI's `r-crosscheck` job regenerates and diffs it against the committed fixture. Reference R sources are vendored nowhere — consult CRAN/GitHub (xiangzhou09/strat) when in doubt.

## Behavioral parity notes (intentional, don't "fix")

- Weighted percentile ranks can slightly exceed 1 (top midrank is `n + 0.5·(1 − w_top)`) — matches Hmisc.
- Single stratum (or all pairs skipped) yields NaN index, like R's 0/0.
- `strat(..., group=...)` raises on any missing value in `group` (R checks before complete-case filtering).

## Release

Tag `v*` → `.github/workflows/publish.yml` builds, smoke-tests the wheel, and publishes to PyPI via trusted publishing (environment `pypi`); manual `workflow_dispatch` publishes to TestPyPI (environment `testpypi`). Keep `__version__` in `src/stratindex/__init__.py` in sync with `pyproject.toml`.
