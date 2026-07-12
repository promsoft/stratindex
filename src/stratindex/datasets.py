"""Bundled example dataset.

``cpsmarch2015`` is the dataset shipped with the R package ``strat``: income,
big class, microclass, education and sampling weight for 14,358 male
respondents aged 35-64 from the March 2015 Current Population Survey.
"""

from __future__ import annotations

import csv
import gzip
from importlib.resources import files

import numpy as np

__all__ = ["load_cpsmarch2015"]

_COLUMNS = ("income", "big_class", "micro_class", "education", "weight")


def load_cpsmarch2015(as_pandas: bool = False):
    """Load the cpsmarch2015 dataset.

    Parameters
    ----------
    as_pandas:
        If True, return a pandas DataFrame (requires pandas); otherwise a
        dict of NumPy arrays keyed by column name.

    Returns
    -------
    dict[str, numpy.ndarray] | pandas.DataFrame
        Columns: ``income`` (float, personal market income in US dollars),
        ``big_class`` (str), ``micro_class`` (int), ``education`` (str),
        ``weight`` (float, CPS sampling weight).
    """
    resource = files("stratindex.data").joinpath("cpsmarch2015.csv.gz")
    with resource.open("rb") as raw, gzip.open(raw, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader))
        assert header == _COLUMNS
        rows = list(reader)

    columns = list(zip(*rows, strict=True))
    data = {
        "income": np.array(columns[0], dtype=float),
        "big_class": np.array(columns[1]),
        "micro_class": np.array(columns[2], dtype=int),
        "education": np.array(columns[3]),
        "weight": np.array(columns[4], dtype=float),
    }
    if as_pandas:
        import pandas as pd

        return pd.DataFrame(data)
    return data
