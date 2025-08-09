"""Causal Discovery Toolkit - Automated causal inference and discovery tools."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .algorithms.base import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
from .utils.data_processing import DataProcessor
from .utils.metrics import CausalMetrics

__all__ = [
    "CausalDiscoveryModel",
    "SimpleLinearCausalModel",
    "CausalResult",
    "DataProcessor", 
    "CausalMetrics",
]