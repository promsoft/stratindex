# stratindex

[![CI](https://github.com/promsoft/stratindex/actions/workflows/ci.yml/badge.svg)](https://github.com/promsoft/stratindex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/stratindex)](https://pypi.org/project/stratindex/)
[![Docs](https://img.shields.io/badge/docs-promsoft.github.io%2Fstratindex-blue)](https://promsoft.github.io/stratindex/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)

A nonparametric index of stratification (Zhou 2012) — a Python port of the R
package [strat](https://cran.r-project.org/package=strat) by Xiang Zhou.

Documentation: https://promsoft.github.io/stratindex/

The index measures how strongly a set of strata (social classes, occupations,
schools, …) stratifies a numeric outcome (income, test scores, …). It is the
weighted excess of concordant over discordant pairs among all pairs of
observations drawn from different strata with distinct outcomes — an analogue
of Somers' D where strata are ordered by the average percentile rank of their
members (or taken as pre-ordered). The index lies in [-1, 1]: 1 means perfect
stratification (every member of a higher stratum outranks every member of a
lower one), 0 means no stratification.

> Zhou, Xiang. 2012. "A Nonparametric Index of Stratification."
> *Sociological Methodology*, 42(1): 365–389.
> [doi:10.1177/0081175012452207](https://doi.org/10.1177/0081175012452207)

## Installation

```bash
pip install stratindex
```

Requires Python ≥ 3.12. The only runtime dependency is NumPy.

## Usage

```python
from stratindex import strat, srank, load_cpsmarch2015

d = load_cpsmarch2015()  # bundled example data: March CPS 2015, 14,358 men

# stratum-specific information: population share and average percentile rank
print(srank(d["income"], d["big_class"], weights=d["weight"]))

# the stratification index with a between-/within-group decomposition
s = strat(
    d["income"], d["big_class"],
    weights=d["weight"],
    group=d["education"], group_name="education",
)
print(s.format(digits=4))
```

```
overall stratification:

 strat  std_error
0.4128    0.01296

decomposition by education:

                   weight   strat
 within education  0.2435  0.2684
between education  0.7565  0.4592
```

Results are plain dataclasses: `s.strat`, `s.std_error`, `s.strata_info`,
`s.decomposition`, `s.within_group`; in Jupyter they render as HTML tables.
If pandas is installed, `s.to_pandas()` returns the tables as DataFrames, and
`load_cpsmarch2015(as_pandas=True)` returns a DataFrame.

A DataFrame (or any mapping of columns, like the dict above) can be passed
directly — string keywords are resolved as column names, and the group label
is taken from the column name:

```python
s = strat(d, outcome="income", strata="big_class",
          weights="weight", group="education")
```

pandas `Categorical` strata keep their category order (used by
`ordered=True` and for row order in the tables). Besides the default
Goodman–Kruskal approximation, a bootstrap standard error is available:
`strat(..., se_method="bootstrap", n_boot=500, random_state=0)`.

## Correspondence with the R package

| R                                        | Python                                      |
| ---------------------------------------- | ------------------------------------------- |
| `strat(outcome, strata, weights, ordered, group)` | `strat(outcome, strata, weights=None, ordered=False, group=None, group_name="group")` |
| `srank(outcome, strata, weights, group)` | `srank(outcome, strata, weights=None, group=None)` |
| `s$overall` (`strat`, `std_error`)       | `s.strat`, `s.std_error` (or `s.overall`)   |
| `s$strata_info`                          | `s.strata_info`                             |
| `s$decomposition`                        | `s.decomposition`                           |
| `s$within_group`                         | `s.within_group`                            |
| `data(cpsmarch2015)`                     | `load_cpsmarch2015()`                       |

Behavioral notes:

- Strata and group levels are ordered by their sorted unique values (as R's
  `factor()` does for character vectors). With `ordered=True` this level
  order is the stratum order.
- R derives the group label from the expression passed as `group`; Python
  uses the column name in data mode (or `group_name=` explicitly).
- The pairwise comparisons (an O(n²) C++ loop in the original) are computed
  in O(n log n) via weighted inversion counting; the full 14,358-row example
  runs in a fraction of a second.
- Numerical output is cross-validated against the original R package
  (see `tests/data/r_golden.json`, regenerated with `scripts/r_golden.R`).

## Standard error

The reported standard error is the approximation of Goodman & Kruskal (1963),
as in the R package: `se = sqrt((1 - strat²) · n / deno)` where `deno` is the
total weight of comparable pairs.

## License

GPL-3.0-or-later, same as the original R package (this is a derivative work).
The bundled `cpsmarch2015` dataset originates from the March 2015 Current
Population Survey (U.S. Census Bureau / BLS public data), as distributed with
the R package.
