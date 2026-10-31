"""cachesim: cache memory simulator with replacement policies, 3C miss classification and AMAT."""
from .cache import Cache, CacheStats, TwoLevelHierarchy, amat, classify_misses
from . import traces

__version__ = "0.1.0"
__all__ = ["Cache", "CacheStats", "TwoLevelHierarchy", "amat", "classify_misses", "traces"]
