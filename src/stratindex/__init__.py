"""stratindex: a nonparametric index of stratification (Zhou 2012).

Python port of the R package 'strat' by Xiang Zhou
(https://cran.r-project.org/package=strat).
"""

from .core import srank, strat
from .results import SrankResult, StratResult

__version__ = "0.1.0"

__all__ = ["srank", "strat", "SrankResult", "StratResult", "__version__"]
