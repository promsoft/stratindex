"""Result containers for :func:`stratindex.strat` and :func:`stratindex.srank`.

The ``__str__`` output mirrors the ``print.strat`` / ``print.srank`` methods
of the R package.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np


def _fmt(value: float, digits: int) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.{digits}g}"


def _cell(value, digits: int) -> tuple[str, bool]:
    """Render one table cell: ``(text, is_numeric)``."""
    if isinstance(value, (int, float, np.floating)):
        return _fmt(float(value), digits), True
    return str(value), False


def _html_cell(value, digits: int) -> str:
    text, numeric = _cell(value, digits)
    if numeric:
        return f'<td style="text-align:right">{text}</td>'
    return f"<td>{html.escape(text)}</td>"


def _html_table(columns: dict[str, list | np.ndarray], digits: int, caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    n_rows = len(next(iter(columns.values())))
    body = "".join(
        "<tr>" + "".join(_html_cell(col[i], digits) for col in columns.values()) + "</tr>"
        for i in range(n_rows)
    )
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table(columns: dict[str, list | np.ndarray], digits: int) -> str:
    """Render a dict of equal-length columns as an aligned text table."""
    cells: dict[str, list[str]] = {}
    for name, col in columns.items():
        cells[name] = [_cell(v, digits)[0] for v in col]
    widths = {name: max(len(name), *(len(c) for c in col)) for name, col in cells.items()}
    lines = ["  ".join(name.rjust(widths[name]) for name in cells)]
    n_rows = len(next(iter(cells.values())))
    for i in range(n_rows):
        lines.append("  ".join(cells[name][i].rjust(widths[name]) for name in cells))
    return "\n".join(lines)


@dataclass
class SrankResult:
    """Stratum-specific information: population share and average percentile rank.

    Attributes
    ----------
    raw:
        Complete cases of all inputs — dict with ``prank``, ``strata``,
        ``weights`` (and ``group`` when supplied), each an ndarray of length n.
    summary:
        Per-stratum table — dict with ``strata`` (level labels), ``share``
        and ``s_prank`` arrays.
    """

    raw: dict[str, np.ndarray] = field(repr=False)
    summary: dict[str, np.ndarray]

    def to_pandas(self):
        """Return ``(raw, summary)`` as pandas DataFrames (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.raw), pd.DataFrame(self.summary)

    def format(self, digits: int = 3) -> str:
        return _table(self.summary, digits)

    def __str__(self) -> str:
        return self.format()

    def _repr_html_(self) -> str:
        return _html_table(self.summary, digits=3)


@dataclass
class StratResult:
    """The stratification index and its approximate standard error.

    Attributes
    ----------
    strat:
        The overall stratification index.
    std_error:
        Approximate standard error (Goodman & Kruskal 1963).
    strata_info:
        Per-stratum table — dict with ``strata``, ``share`` and ``s_prank``.
    decomposition:
        ``None`` unless a group was supplied; otherwise a dict with rows
        ``within`` / ``between``, each a dict with ``weight`` and ``strat``.
    within_group:
        ``None`` unless a group was supplied; otherwise a dict with the group
        levels (keyed by ``group_name``) and per-group ``weight`` / ``strat``
        arrays.
    """

    strat: float
    std_error: float
    strata_info: dict[str, np.ndarray]
    decomposition: dict[str, dict[str, float]] | None = None
    within_group: dict[str, np.ndarray] | None = None
    group_name: str = "group"

    @property
    def overall(self) -> dict[str, float]:
        return {"strat": self.strat, "std_error": self.std_error}

    def to_pandas(self):
        """Return ``strata_info`` (and ``within_group`` if any) as DataFrames."""
        import pandas as pd

        strata_info = pd.DataFrame(self.strata_info)
        if self.within_group is None:
            return strata_info
        return strata_info, pd.DataFrame(self.within_group)

    def _overall_rows(self) -> dict[str, list]:
        return {name: [value] for name, value in self.overall.items()}

    def format(self, digits: int = 3) -> str:
        lines = [
            "overall stratification:",
            "",
            _table(self._overall_rows(), digits),
        ]
        if self.decomposition is not None:
            lines += [
                "",
                f"decomposition by {self.group_name}:",
                "",
                _table(self._decomposition_rows(), digits),
            ]
        return "\n".join(lines)

    def _decomposition_rows(self) -> dict[str, list]:
        return {
            "": [f"within {self.group_name}", f"between {self.group_name}"],
            "weight": [
                self.decomposition["within"]["weight"],
                self.decomposition["between"]["weight"],
            ],
            "strat": [
                self.decomposition["within"]["strat"],
                self.decomposition["between"]["strat"],
            ],
        }

    def __str__(self) -> str:
        return self.format()

    def _repr_html_(self) -> str:
        digits = 3
        parts = [_html_table(self._overall_rows(), digits, caption="overall stratification")]
        if self.decomposition is not None:
            parts.append(
                _html_table(
                    self._decomposition_rows(),
                    digits,
                    caption=f"decomposition by {self.group_name}",
                )
            )
        parts.append(_html_table(self.strata_info, digits, caption="strata"))
        return "<div>" + "".join(parts) + "</div>"
