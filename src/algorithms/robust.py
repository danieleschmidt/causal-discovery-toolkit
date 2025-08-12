"""Robust causal discovery algorithms with comprehensive error handling."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings

try:
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.validation import DataValidator, ParameterValidator, validate_adjacency_matrix
    from ..utils.logging_config import get_logger
    from ..utils.monitoring import monitor_performance, CircuitBreaker, CircuitBreakerOpenException
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from base import CausalDiscoveryModel, CausalResult
    from validation import DataValidator, ParameterValidator, validate_adjacency_matrix
    from logging_config import get_logger
    from monitoring import monitor_performance, CircuitBreaker, CircuitBreakerOpenException

logger = get_logger(__name__)


class RobustSimpleLinearCausalModel(CausalDiscoveryModel):
    """Robust version of SimpleLinearCausalModel with comprehensive error handling."""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 validate_inputs: bool = True,
                 handle_missing: str = 'drop',  # 'drop', 'impute_mean', 'impute_median'
                 correlation_method: str = 'pearson',  # 'pearson', 'spearman', 'kendall'
                 min_samples: int = 10,
                 max_features: int = 1000,
                 **kwargs):
        """Initialize robust causal discovery model.
        
        Args:
            threshold: Correlation threshold for edge detection
            validate_inputs: Whether to validate inputs
            handle_missing: How to handle missing data
            correlation_method: Correlation method to use
            min_samples: Minimum samples required
            max_features: Maximum features allowed
            **kwargs: Additional parameters
        """
        super().__init__(threshold=threshold, **kwargs)
        
        # Validate parameters at initialization
        param_validation = ParameterValidator.validate_threshold(threshold)
        if not param_validation.is_valid:
            raise ValueError(f"Invalid threshold: {param_validation.errors}")
        
        self.threshold = threshold
        self.validate_inputs = validate_inputs
        self.handle_missing = handle_missing
        self.correlation_method = correlation_method
        self.min_samples = min_samples
        self.max_features = max_features
        
        # Initialize components
        self.data_validator = DataValidator(strict=False)
        self._data = None
        self._fitted_successfully = False
        self.fit_metadata = {}
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=30,
            expected_exception=Exception
        )
        
        logger.info(f"Initialized RobustSimpleLinearCausalModel with threshold={threshold}")
    
    @monitor_performance("fit")
    def fit(self, data: pd.DataFrame) -> 'RobustSimpleLinearCausalModel':
        """Fit the causal discovery model with robust error handling.
        
        Args:
            data: Input data with shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If data validation fails
            RuntimeError: If fitting fails
        """
        logger.info(f"Starting fit with data shape: {data.shape}")
        
        try:
            # Input validation
            if self.validate_inputs:
                validation_result = self.data_validator.validate_input_data(data)
                
                # Log validation results
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        logger.warning(f"Data validation warning: {warning}")
                        warnings.warn(warning, UserWarning)
                
                if not validation_result.is_valid:
                    error_msg = f"Data validation failed: {validation_result.errors}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                self.fit_metadata['validation'] = validation_result.metadata
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Additional safety checks
            if processed_data.shape[0] < self.min_samples:
                raise ValueError(f"Insufficient samples: {processed_data.shape[0]} < {self.min_samples}")
            
            if processed_data.shape[1] > self.max_features:
                raise ValueError(f"Too many features: {processed_data.shape[1]} > {self.max_features}")
            
            # Store processed data
            self._data = processed_data
            self.is_fitted = True
            self._fitted_successfully = True
            
            # Store fit metadata
            self.fit_metadata.update({
                'original_shape': data.shape,
                'processed_shape': processed_data.shape,
                'n_missing_handled': data.isnull().sum().sum(),
                'correlation_method': self.correlation_method,
                'threshold': self.threshold
            })
            
            logger.info(f"Successfully fit model. Processed data shape: {processed_data.shape}")
            return self
            
        except Exception as e:
            self._fitted_successfully = False
            self.is_fitted = False
            logger.error(f"Failed to fit model: {str(e)}")
            raise RuntimeError(f"Model fitting failed: {str(e)}") from e
    
    @monitor_performance("discover")
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships with robust error handling.
        
        Args:
            data: Optional new data, uses fitted data if None
            
        Returns:
            CausalResult containing discovered relationships
            
        Raises:
            RuntimeError: If model not fitted or discovery fails
        """
        if not self._fitted_successfully:
            raise RuntimeError("Model must be successfully fitted before discovery")
        
        logger.info("Starting causal discovery")
        
        try:
            # Use circuit breaker for resilience
            return self.circuit_breaker.call(self._perform_discovery, data)
            
        except CircuitBreakerOpenException:
            logger.error("Circuit breaker is OPEN - too many recent failures")
            raise RuntimeError("Discovery service temporarily unavailable due to repeated failures")
        except Exception as e:
            logger.error(f"Discovery failed: {str(e)}")
            raise RuntimeError(f"Causal discovery failed: {str(e)}") from e
    
    def _perform_discovery(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Internal method to perform discovery."""
        # Determine data to use
        if data is not None:
            if self.validate_inputs:
                validation_result = self.data_validator.validate_input_data(data)
                if not validation_result.is_valid:
                    raise ValueError(f"Input data validation failed: {validation_result.errors}")
            discovery_data = self._preprocess_data(data)
        else:
            discovery_data = self._data
        
        logger.debug(f"Computing correlations using {self.correlation_method} method")
        
        # Compute correlation matrix with error handling
        try:
            if self.correlation_method == 'pearson':
                corr_matrix = discovery_data.corr(method='pearson')
            elif self.correlation_method == 'spearman':
                corr_matrix = discovery_data.corr(method='spearman')
            elif self.correlation_method == 'kendall':
                corr_matrix = discovery_data.corr(method='kendall')
            else:
                raise ValueError(f"Unknown correlation method: {self.correlation_method}")
            
            # Check for invalid correlations
            if corr_matrix.isnull().any().any():
                logger.warning("NaN values found in correlation matrix")
                corr_matrix = corr_matrix.fillna(0)
            
        except Exception as e:
            logger.error(f"Failed to compute correlations: {str(e)}")
            raise RuntimeError(f"Correlation computation failed: {str(e)}") from e
        
        # Create adjacency matrix
        abs_corr_matrix = corr_matrix.abs()
        n_vars = len(corr_matrix)
        
        # Apply threshold with safety checks
        try:
            adjacency = (abs_corr_matrix > self.threshold).astype(int)
            np.fill_diagonal(adjacency.values, 0)  # Remove self-connections
            
            # Validate result
            adj_validation = validate_adjacency_matrix(adjacency.values)
            if not adj_validation.is_valid:
                logger.warning(f"Adjacency matrix validation warnings: {adj_validation.warnings}")
            
        except Exception as e:
            logger.error(f"Failed to create adjacency matrix: {str(e)}")
            raise RuntimeError(f"Adjacency matrix creation failed: {str(e)}") from e
        
        # Use correlation values as confidence scores
        confidence = abs_corr_matrix.values.copy()
        np.fill_diagonal(confidence, 0)
        
        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "correlation_method": self.correlation_method,
            "n_variables": n_vars,
            "n_edges": np.sum(adjacency.values),
            "variable_names": list(discovery_data.columns),
            "sparsity": 1 - (np.sum(adjacency.values) / (n_vars * (n_vars - 1))),
            "max_confidence": np.max(confidence),
            "mean_confidence": np.mean(confidence[confidence > 0]) if np.any(confidence > 0) else 0,
            **self.fit_metadata
        }
        
        logger.info(f"Discovery completed. Found {metadata['n_edges']} edges")
        
        return CausalResult(
            adjacency_matrix=adjacency.values,
            confidence_scores=confidence,
            method_used="RobustSimpleLinearCausal",
            metadata=metadata
        )
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with error handling.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
            
        Raises:
            ValueError: If preprocessing fails
        """
        try:
            processed = data.copy()
            
            # Handle missing values
            if self.handle_missing == 'drop':
                original_len = len(processed)
                processed = processed.dropna()
                if len(processed) == 0:
                    raise ValueError("All rows removed after dropping NaN values")
                if len(processed) < original_len * 0.5:
                    logger.warning(f"Dropped {original_len - len(processed)} rows with NaN values")
                    
            elif self.handle_missing == 'impute_mean':
                processed = processed.fillna(processed.mean())
                
            elif self.handle_missing == 'impute_median':
                processed = processed.fillna(processed.median())
                
            else:
                raise ValueError(f"Unknown missing data handling method: {self.handle_missing}")
            
            # Select only numeric columns
            numeric_cols = processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found after preprocessing")
            
            processed = processed[numeric_cols]
            
            # Remove constant columns
            constant_cols = processed.columns[processed.nunique() <= 1]
            if len(constant_cols) > 0:
                logger.warning(f"Removing constant columns: {list(constant_cols)}")
                processed = processed.drop(columns=constant_cols)
            
            if processed.empty:
                raise ValueError("No data remaining after preprocessing")
            
            logger.debug(f"Preprocessing completed. Shape: {data.shape} -> {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise ValueError(f"Data preprocessing failed: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'RobustSimpleLinearCausalModel',
            'parameters': {
                'threshold': self.threshold,
                'correlation_method': self.correlation_method,
                'handle_missing': self.handle_missing,
                'min_samples': self.min_samples,
                'max_features': self.max_features
            },
            'state': {
                'is_fitted': self.is_fitted,
                'fitted_successfully': self._fitted_successfully,
                'data_shape': self._data.shape if self._data is not None else None
            },
            'fit_metadata': self.fit_metadata,
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.last_failure_time = None
        logger.info("Circuit breaker manually reset")
    
    def validate_health(self) -> Dict[str, Any]:
        """Check model health and readiness."""
        health = {
            'healthy': True,
            'issues': [],
            'model_ready': self._fitted_successfully,
            'circuit_breaker_ok': self.circuit_breaker.state != "OPEN"
        }
        
        if not self._fitted_successfully:
            health['healthy'] = False
            health['issues'].append("Model not fitted successfully")
        
        if self.circuit_breaker.state == "OPEN":
            health['healthy'] = False
            health['issues'].append("Circuit breaker is OPEN")
        
        if self._data is not None and self._data.shape[1] == 0:
            health['healthy'] = False
            health['issues'].append("No features available")
        
        return health