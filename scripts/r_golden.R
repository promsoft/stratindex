#!/usr/bin/env Rscript
# Generate golden reference values from the original R package 'strat'
# (https://cran.r-project.org/package=strat) for cross-validation of the
# Python port. Run inside rocker/r2u for fast binary installs:
#
#   docker run --rm -v "$PWD":/work rocker/r2u:jammy \
#     Rscript /work/scripts/r_golden.R /work/tests/data/r_golden.json

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args)) args[[1]] else "r_golden.json"

if (!requireNamespace("strat", quietly = TRUE)) install.packages("strat")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

library(strat)
data(cpsmarch2015)

strat_to_list <- function(s) {
  out <- list(
    strat = unname(s$overall["strat"]),
    std_error = unname(s$overall["std_error"]),
    strata_info = list(
      strata = as.character(s$strata_info$strata),
      share = s$strata_info$share,
      s_prank = s$strata_info$s_prank
    )
  )
  if (!is.null(s$decomposition)) {
    out$decomposition <- list(
      # rows are c("within <g>", "between <g>")
      weight = unname(s$decomposition[, "weight"]),
      strat = unname(s$decomposition[, "strat"])
    )
    out$within_group <- list(
      group = as.character(s$within_group[[1]]),
      weight = s$within_group$weight,
      strat = s$within_group$strat
    )
  }
  out
}

cases <- list()
cases$main <- strat_to_list(
  with(cpsmarch2015, strat(income, big_class, weights = weight, group = education))
)
cases$unweighted <- strat_to_list(with(cpsmarch2015, strat(income, big_class)))
cases$micro <- strat_to_list(with(cpsmarch2015, strat(income, micro_class, weights = weight)))
cases$micro_ordered <- strat_to_list(
  with(cpsmarch2015, strat(income, micro_class, weights = weight, ordered = TRUE))
)

# small deterministic case with ties, non-integer weights and a group
x <- c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5)
st <- c("a", "b", "a", "c", "b", "c", "a", "b", "c", "a", "b")
w <- c(1, 2, 1, 0.5, 3, 1, 2, 1, 1.5, 1, 2)
g <- c("u", "v", "u", "v", "u", "v", "u", "v", "u", "v", "u")
cases$synthetic <- strat_to_list(strat(x, st, weights = w, group = g))
cases$synthetic_ordered <- strat_to_list(strat(x, st, weights = w, ordered = TRUE))

sr <- with(cpsmarch2015, srank(income, big_class, weights = weight))
cases$srank_main <- list(
  strata = as.character(sr$summary$strata),
  share = sr$summary$share,
  s_prank = sr$summary$s_prank
)

json <- jsonlite::toJSON(cases, digits = I(17), auto_unbox = TRUE, pretty = TRUE)
writeLines(json, out_path)
cat("wrote", out_path, "\n")
