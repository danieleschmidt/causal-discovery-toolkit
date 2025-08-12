"""Comprehensive validation utilities for causal discovery."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation for causal discovery."""
    
    def __init__(self, strict: bool = True):
        """Initialize validator.
        
        Args:
            strict: If True, treats warnings as errors
        """
        self.strict = strict
    
    def validate_input_data(self, data: Any) -> ValidationResult:
        """Validate input data for causal discovery.
        
        Args:
            data: Input data to validate
            
        Returns:
            ValidationResult with detailed information
        """
        errors = []
        warnings_list = []
        metadata = {}
        
        # Basic type validation
        if not isinstance(data, pd.DataFrame):
            errors.append(f"Input must be pandas DataFrame, got {type(data)}")
            return ValidationResult(False, errors, warnings_list, metadata)
        
        # Shape validation
        n_samples, n_features = data.shape
        metadata.update({
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
        })
        
        if data.empty:
            errors.append("DataFrame is empty")
        
        if n_samples < 10:
            errors.append(f"Too few samples ({n_samples}). Minimum 10 required for reliable inference")
        elif n_samples < 50:
            warnings_list.append(f"Small sample size ({n_samples}). Consider more data for better reliability")
        
        if n_features < 2:
            errors.append(f"Too few features ({n_features}). Minimum 2 required for causal discovery")
        elif n_features > 1000:
            warnings_list.append(f"Large number of features ({n_features}). Consider dimensionality reduction")
        
        # Data type validation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        
        metadata.update({
            'n_numeric_cols': len(numeric_cols),
            'n_non_numeric_cols': len(non_numeric_cols),
            'numeric_cols': list(numeric_cols),
            'non_numeric_cols': list(non_numeric_cols)
        })
        
        if len(non_numeric_cols) > 0:
            errors.append(f"Non-numeric columns found: {list(non_numeric_cols)}")
        
        # Missing data validation
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (n_samples * n_features)) * 100
        
        metadata.update({
            'total_missing_values': total_missing,
            'missing_percentage': missing_pct,
            'cols_with_missing': list(missing_counts[missing_counts > 0].index)
        })
        
        if total_missing > 0:
            if missing_pct > 50:
                errors.append(f"Excessive missing data ({missing_pct:.1f}%)")
            elif missing_pct > 10:
                warnings_list.append(f"High missing data percentage ({missing_pct:.1f}%)")
        
        # Constant columns validation
        if len(numeric_cols) > 0:
            numeric_data = data[numeric_cols]
            constant_cols = numeric_cols[numeric_data.nunique() <= 1]
            
            metadata['constant_cols'] = list(constant_cols)
            
            if len(constant_cols) > 0:
                warnings_list.append(f"Constant columns found: {list(constant_cols)}")
        
        # Infinite/NaN validation in numeric columns
        if len(numeric_cols) > 0:
            inf_cols = []
            for col in numeric_cols:
                if np.isinf(data[col]).any():
                    inf_cols.append(col)
            
            metadata['inf_cols'] = inf_cols
            
            if inf_cols:
                errors.append(f"Infinite values found in columns: {inf_cols}")
        
        # Statistical validation
        if len(numeric_cols) >= 2:
            self._add_statistical_validation(data[numeric_cols], warnings_list, metadata)
        
        # Memory validation
        if metadata['memory_usage_mb'] > 1000:  # 1GB
            warnings_list.append(f"Large memory usage ({metadata['memory_usage_mb']:.1f} MB)")
        
        is_valid = len(errors) == 0 and (not self.strict or len(warnings_list) == 0)
        
        return ValidationResult(is_valid, errors, warnings_list, metadata)
    
    def _add_statistical_validation(self, data: pd.DataFrame, 
                                  warnings_list: List[str], 
                                  metadata: Dict[str, Any]) -> None:
        """Add statistical validation checks."""
        # Correlation-based validation
        corr_matrix = data.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        # Perfect correlations
        perfect_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.999:
                    perfect_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        metadata['perfect_correlations'] = perfect_corr_pairs
        
        if perfect_corr_pairs:
            warnings_list.append(f"Nearly identical variables found: {perfect_corr_pairs}")
        
        # Low variance validation
        low_var_cols = []
        for col in data.columns:
            if data[col].var() < 1e-10:
                low_var_cols.append(col)
        
        metadata['low_variance_cols'] = low_var_cols
        
        if low_var_cols:
            warnings_list.append(f"Very low variance columns: {low_var_cols}")


class ParameterValidator:
    """Validate algorithm parameters."""
    
    @staticmethod
    def validate_threshold(threshold: float) -> ValidationResult:
        """Validate correlation threshold parameter."""
        errors = []
        warnings_list = []
        
        if not isinstance(threshold, (int, float)):
            errors.append(f"Threshold must be numeric, got {type(threshold)}")
        elif threshold < 0 or threshold > 1:
            errors.append(f"Threshold must be between 0 and 1, got {threshold}")
        elif threshold < 0.1:
            warnings_list.append(f"Very low threshold ({threshold}) may result in many false positives")
        elif threshold > 0.8:
            warnings_list.append(f"High threshold ({threshold}) may miss weak relationships")
        
        is_valid = len(errors) == 0
        metadata = {'threshold': threshold, 'valid_range': (0, 1)}
        
        return ValidationResult(is_valid, errors, warnings_list, metadata)
    
    @staticmethod
    def validate_sample_size(n_samples: int, n_features: int) -> ValidationResult:
        """Validate sample size relative to feature count."""
        errors = []
        warnings_list = []
        
        if not isinstance(n_samples, int) or n_samples <= 0:
            errors.append(f"Sample size must be positive integer, got {n_samples}")
        
        if not isinstance(n_features, int) or n_features <= 0:
            errors.append(f"Feature count must be positive integer, got {n_features}")
        
        if len(errors) == 0:
            # Rule of thumb: need at least 10 samples per feature
            min_recommended = max(50, 10 * n_features)
            
            if n_samples < 10:
                errors.append(f"Insufficient samples ({n_samples}). Minimum 10 required")
            elif n_samples < min_recommended:
                warnings_list.append(
                    f"Small sample size ({n_samples}) for {n_features} features. "
                    f"Recommended: at least {min_recommended}"
                )
        
        is_valid = len(errors) == 0
        metadata = {
            'n_samples': n_samples,
            'n_features': n_features,
            'ratio': n_samples / max(n_features, 1),
            'recommended_min': max(50, 10 * n_features)
        }
        
        return ValidationResult(is_valid, errors, warnings_list, metadata)
    
    @staticmethod
    def validate_range(param_name: str, value: float, min_val: float, max_val: float) -> None:
        """Validate that a parameter is within a specified range.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Raises:
            ValueError: If value is outside the valid range
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{param_name} must be numeric, got {type(value)}")
        
        if value < min_val or value > max_val:
            raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")


def validate_adjacency_matrix(adj_matrix: np.ndarray) -> ValidationResult:
    """Validate adjacency matrix format and properties."""
    errors = []
    warnings_list = []
    metadata = {}
    
    if not isinstance(adj_matrix, np.ndarray):
        errors.append(f"Adjacency matrix must be numpy array, got {type(adj_matrix)}")
        return ValidationResult(False, errors, warnings_list, metadata)
    
    if adj_matrix.ndim != 2:
        errors.append(f"Adjacency matrix must be 2D, got {adj_matrix.ndim}D")
    
    n_rows, n_cols = adj_matrix.shape
    if n_rows != n_cols:
        errors.append(f"Adjacency matrix must be square, got shape {adj_matrix.shape}")
    
    metadata.update({
        'shape': adj_matrix.shape,
        'dtype': str(adj_matrix.dtype),
        'n_edges': np.sum(adj_matrix > 0),
        'is_binary': np.all(np.isin(adj_matrix, [0, 1])),
        'has_self_loops': np.any(np.diag(adj_matrix) != 0)
    })
    
    # Check for valid values (should be 0 or 1 for binary adjacency)
    if not np.all(np.isin(adj_matrix, [0, 1])):
        if np.all(adj_matrix >= 0):
            warnings_list.append("Non-binary adjacency matrix detected (values not in {0,1})")
        else:
            errors.append("Adjacency matrix contains negative values")
    
    # Check for self-loops
    if np.any(np.diag(adj_matrix) != 0):
        warnings_list.append("Self-loops detected in adjacency matrix")
    
    # Sparsity check
    if len(errors) == 0:
        n_possible_edges = n_rows * (n_rows - 1)  # Exclude diagonal
        sparsity = 1 - (metadata['n_edges'] / max(n_possible_edges, 1))
        metadata['sparsity'] = sparsity
        
        if sparsity < 0.1:
            warnings_list.append(f"Very dense graph (sparsity: {sparsity:.2f})")
    
    is_valid = len(errors) == 0
    
    return ValidationResult(is_valid, errors, warnings_list, metadata)