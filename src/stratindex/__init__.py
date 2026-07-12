"""stratindex: a nonparametric index of stratification (Zhou 2012).

Python port of the R package 'strat' by Xiang Zhou
(https://cran.r-project.org/package=strat).
"""

from .core import srank, strat
from .datasets import load_cpsmarch2015
from .results import SrankResult, StratResult

__version__ = "0.2.0"

__all__ = ["srank", "strat", "SrankResult", "StratResult", "load_cpsmarch2015", "__version__"]
