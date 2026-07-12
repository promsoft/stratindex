# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`stratindex` — Python port of the R package `strat` (https://cran.r-project.org/package=strat), implementing the nonparametric stratification index of Zhou (2012). Published on PyPI as `stratindex` (0.1.0 released 2026-07-12). License is GPL-3.0-or-later (the port is a derivative of the GPL R package — do not relicense).

## Spec workflow

- `spec/stratindex.md` (in Russian) holds the task spec, the Q&A decisions, and the progress checklist; `spec/backlog.md` holds ideas for future versions. Keep them current: record new decisions and check off completed work.
- Commit and push immediately after every change to `spec/` — don't batch spec edits with unrelated work.
- CI ignores `spec/**` (`paths-ignore` in `ci.yml`), so spec-only pushes don't burn Actions minutes. A commit that touches both spec and code still runs CI.

## Environment & commands

- `venv` is a symlink to a pyenv virtualenv (Python 3.12.8). Use `venv/bin/python` / `venv/bin/uv` explicitly if pyenv shims aren't active.
- Dependencies: declare in `requirements.in`, then `venv/bin/uv pip compile requirements.in -o requirements.txt` and `venv/bin/uv pip sync requirements.txt`. Runtime dependency is numpy only; everything else in requirements.in is dev tooling.
- Install for development: `venv/bin/uv pip install -e .`
- Tests: `venv/bin/python -m pytest -q` (single test: `venv/bin/python -m pytest tests/test_strat.py::test_hand_computed_example -q`). The suite takes ~30 s; the slow part is the full-dataset R-golden cross-checks.
- Lint/format: `venv/bin/ruff check src tests scripts` and `venv/bin/ruff format src tests scripts` (CI enforces `ruff format --check`).

## Architecture

- `src/stratindex/_utils.py` — `clean()` (validation, complete cases, weight normalization to sum n), `_Factor` (level encoding: sorted unique values for plain arrays, category order for pandas Categorical, R `factor()` semantics), and `wtd_rank()` (port of `Hmisc::wtd.rank(normwt=TRUE)`, weighted midranks). Percentile ranks of equal outcomes are exactly equal by construction (unique-value mapping, not interpolation) — the kernel's tie-skipping depends on this.
- `src/stratindex/_kernel.py` — `pair_sums`/`pair_sums_by` are O(n log n): Fenwick-tree weighted inversion counting for the numerator, inclusion-exclusion over tie blocks for the denominator; inputs must be pre-sorted by `r`; sums are returned as `np.float64` so degenerate 0/0 yields NaN (as in R), not ZeroDivisionError. The original O(n²) blocked ports (`pair_sums_blocked`/`_by_blocked`) are kept solely as an independent reference for tests.
- `src/stratindex/core.py` — public `strat()` / `srank()` with two call styles resolved by `_resolve_inputs` (arrays positionally, or DataFrame/mapping first with column-name keywords); bootstrap SE (`se_method="bootstrap"`) recomputes ranks and stratum order per replicate.
- `results.py` — result dataclasses; `__str__` mirrors the R print methods, `_repr_html_` renders tables in Jupyter. `datasets.py` — bundled `cpsmarch2015` loader.
- `tests/conftest.py` holds naive loop-based reference implementations translated literally from the R/C++ sources; kernel tests compare fast and blocked against them.

## Docs

`docs/` + `mkdocs.yml` (material, mkdocstrings). Build locally: `venv/bin/mkdocs build --strict`. Deployed to https://promsoft.github.io/stratindex/ by `.github/workflows/docs.yml` (gh-deploy to the `gh-pages` branch) on pushes to main touching docs or src.

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
