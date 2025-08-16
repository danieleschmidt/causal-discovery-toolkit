"""Base classes for causal discovery algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CausalResult:
    """Results from causal discovery."""
    adjacency_matrix: np.ndarray
    confidence_scores: np.ndarray
    method_used: str
    metadata: Dict[str, Any]


class CausalDiscoveryModel(ABC):
    """Abstract base class for causal discovery models."""
    
    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'CausalDiscoveryModel':
        """Fit the causal discovery model to data.
        
        Args:
            data: Input data with shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships.
        
        Args:
            data: Optional new data, uses fitted data if None
            
        Returns:
            CausalResult containing discovered relationships
        """
        pass
    
    def fit_discover(self, data: pd.DataFrame) -> CausalResult:
        """Convenience method to fit and discover in one step."""
        return self.fit(data).discover()


class SimpleLinearCausalModel(CausalDiscoveryModel):
    """Simple linear causal discovery using correlation with thresholding."""
    
    def __init__(self, threshold: float = 0.3, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
        self._data = None
        
    def fit(self, data: pd.DataFrame) -> 'SimpleLinearCausalModel':
        """Fit the model by storing data and computing correlations."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        self._data = data
        self.is_fitted = True
        return self
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using correlation thresholding."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
            
        if data is None:
            data = self._data
            
        # Compute correlation matrix
        corr_matrix = data.corr().abs()
        n_vars = len(corr_matrix)
        
        # Create adjacency matrix by thresholding correlations
        adjacency = (corr_matrix > self.threshold).astype(int)
        np.fill_diagonal(adjacency.values, 0)  # Remove self-connections
        
        # Use correlation values as confidence scores
        confidence = corr_matrix.values.copy()
        np.fill_diagonal(confidence, 0)
        
        return CausalResult(
            adjacency_matrix=adjacency.values,
            confidence_scores=confidence,
            method_used="SimpleLinearCausal",
            metadata={
                "threshold": self.threshold,
                "n_variables": n_vars,
                "n_edges": np.sum(adjacency.values),
                "variable_names": list(data.columns)
            }
        )
    
    def predict(self, data: pd.DataFrame) -> CausalResult:
        """Alias for discover method for compatibility."""
        return self.discover(data)