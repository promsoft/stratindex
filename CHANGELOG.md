# Changelog

## 0.2.0 (unreleased)

- **O(n log n) kernel**: pairwise comparisons are now computed via weighted
  inversion counting (Fenwick tree) with an inclusion-exclusion denominator
  instead of the O(n²) blocked loop; the full `cpsmarch2015` example with
  group decomposition dropped from ~6 s to ~0.15 s. The blocked kernel is
  kept as an independent reference in the test suite.
- **pandas Categorical**: strata and group passed as `pandas.Categorical`
  (or a Series of `category` dtype) keep their category order — it defines
  the stratum order under `ordered=True` and the row order of
  `strata_info` / `within_group`. Unused categories are dropped, keeping the
  order (R `factor()` semantics).
- **Data-mode calls**: `strat(df, outcome="income", strata="big_class",
  weights="weight", group="education")` accepts a DataFrame or any mapping
  of columns as the first argument; string keywords are resolved as column
  names and `group_name` defaults to the group column name.
- **Bootstrap standard error**: `strat(..., se_method="bootstrap",
  n_boot=200, random_state=...)`; percentile ranks and stratum ordering are
  recomputed in every replicate. The default remains the Goodman & Kruskal
  (1963) approximation.
- **Jupyter**: results render as HTML tables via `_repr_html_`.
- **Docs**: https://promsoft.github.io/stratindex/ (mkdocs, deployed from
  CI); README badges.

## 0.1.0 (2026-07-12)

- Initial release: port of the R package `strat` (Zhou 2012 stratification
  index) — `strat()`, `srank()`, between-/within-group decomposition,
  Goodman & Kruskal (1963) approximate standard error, bundled
  `cpsmarch2015` dataset.
- Numerical output cross-validated against the CRAN package (golden values
  regenerated from CRAN in CI).
