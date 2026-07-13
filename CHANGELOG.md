# Changelog

## 0.2.1 (2026-07-13)

Internal cleanup and performance release; no API or numerical changes.

- **Faster numerator**: pairwise concordant-minus-discordant sums are now
  computed blockwise (vectorized per equal-rank block) with a Fenwick-tree
  fallback for inputs with very many distinct stratum positions. Bootstrap
  SE with default `n_boot=200` at n=10,000: ~11 s → ~0.9 s; the full
  `cpsmarch2015` example with decomposition: ~0.15 s → ~0.02 s.
- **O(n) memory** for the tie-block denominator with many strata (the
  (r, y) cell key is densified before `bincount`; previously hundreds of MB
  with thousands of strata).
- `strat()` and the bootstrap replicates now share one implementation of
  the stratum-ordering pipeline, so they cannot drift apart.
- The O(n²) reference kernels moved from the package into the test suite.

## 0.2.0 (2026-07-13)

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
