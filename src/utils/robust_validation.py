"""Enhanced robust validation and error handling utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a validation check."""
    is_valid: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RobustValidator(ABC):
    """Abstract base class for robust validators."""
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate input and return result."""
        pass
    
    def __call__(self, data: Any) -> ValidationResult:
        """Make validator callable."""
        return self.validate(data)


class DataQualityValidator(RobustValidator):
    """Comprehensive data quality validation."""
    
    def __init__(self, 
                 min_samples: int = 10,
                 max_missing_ratio: float = 0.3,
                 min_variance: float = 1e-6,
                 max_correlation: float = 0.99):
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
        self.min_variance = min_variance
        self.max_correlation = max_correlation
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data quality."""
        issues = []
        
        # Check data type
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                message="Input must be a pandas DataFrame",
                severity="error"
            )
        
        # Check if empty
        if data.empty:
            return ValidationResult(
                is_valid=False,
                message="DataFrame is empty",
                severity="error"
            )
        
        # Check minimum samples
        if len(data) < self.min_samples:
            issues.append(f"Insufficient samples: {len(data)} < {self.min_samples}")
        
        # Check missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > self.max_missing_ratio:
            issues.append(f"High missing data ratio: {missing_ratio:.3f} > {self.max_missing_ratio}")
        
        # Check for constant variables
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].var() < self.min_variance:
                issues.append(f"Column '{col}' has near-zero variance: {data[col].var():.2e}")
        
        # Check for perfect correlations
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            # Remove diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = corr_matrix.max().max()
            
            if max_corr > self.max_correlation:
                issues.append(f"High correlation detected: {max_corr:.3f} > {self.max_correlation}")
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            issues.append("Infinite values detected in numeric columns")
        
        # Determine severity and message
        if any("Insufficient samples" in issue or "empty" in issue.lower() for issue in issues):
            severity = "error"
            is_valid = False
        elif issues:
            severity = "warning" 
            is_valid = True
        else:
            severity = "info"
            is_valid = True
            issues = ["Data quality validation passed"]
        
        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(issues),
            severity=severity,
            metadata={
                "n_samples": len(data),
                "n_variables": len(data.columns),
                "missing_ratio": missing_ratio,
                "numeric_columns": len(numeric_cols)
            }
        )


class ParameterValidator(RobustValidator):
    """Validate algorithm parameters."""
    
    def __init__(self, parameter_specs: Dict[str, Dict[str, Any]]):
        """
        Initialize with parameter specifications.
        
        Args:
            parameter_specs: Dict mapping parameter names to their specs
                Each spec can contain: type, min, max, allowed_values, required
        """
        self.parameter_specs = parameter_specs
    
    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameters against specifications."""
        issues = []
        
        # Check required parameters
        required_params = {k for k, v in self.parameter_specs.items() 
                          if v.get('required', False)}
        missing_params = required_params - set(params.keys())
        
        if missing_params:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required parameters: {missing_params}",
                severity="error"
            )
        
        # Validate each parameter
        for param_name, param_value in params.items():
            if param_name not in self.parameter_specs:
                issues.append(f"Unknown parameter: {param_name}")
                continue
                
            spec = self.parameter_specs[param_name]
            
            # Type validation
            if 'type' in spec and not isinstance(param_value, spec['type']):
                issues.append(f"Parameter '{param_name}' must be of type {spec['type'].__name__}")
                continue
            
            # Range validation for numeric types
            if isinstance(param_value, (int, float)):
                if 'min' in spec and param_value < spec['min']:
                    issues.append(f"Parameter '{param_name}' = {param_value} < minimum {spec['min']}")
                
                if 'max' in spec and param_value > spec['max']:
                    issues.append(f"Parameter '{param_name}' = {param_value} > maximum {spec['max']}")
            
            # Allowed values validation
            if 'allowed_values' in spec and param_value not in spec['allowed_values']:
                issues.append(f"Parameter '{param_name}' = {param_value} not in allowed values {spec['allowed_values']}")
        
        severity = "error" if any("must be" in issue or "Missing" in issue for issue in issues) else "warning"
        is_valid = severity != "error"
        
        if not issues:
            issues = ["Parameter validation passed"]
            severity = "info"
        
        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(issues),
            severity=severity,
            metadata={"validated_params": list(params.keys())}
        )


class MemoryValidator(RobustValidator):
    """Validate memory requirements and system resources."""
    
    def __init__(self, max_memory_gb: float = 8.0, min_free_memory_ratio: float = 0.2):
        self.max_memory_gb = max_memory_gb
        self.min_free_memory_ratio = min_free_memory_ratio
    
    def validate(self, data_size_estimate: int) -> ValidationResult:
        """
        Validate memory requirements.
        
        Args:
            data_size_estimate: Estimated memory usage in bytes
        """
        # Get system memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        # Estimate memory requirement (with safety factor)
        estimated_gb = (data_size_estimate * 3) / (1024**3)  # 3x safety factor
        
        issues = []
        
        # Check if we have enough available memory
        if estimated_gb > available_gb:
            issues.append(f"Insufficient available memory: need {estimated_gb:.2f}GB, have {available_gb:.2f}GB")
        
        # Check if we exceed maximum allowed usage
        if estimated_gb > self.max_memory_gb:
            issues.append(f"Memory requirement {estimated_gb:.2f}GB exceeds limit {self.max_memory_gb}GB")
        
        # Check if we'll leave enough free memory
        remaining_ratio = (available_gb - estimated_gb) / total_gb
        if remaining_ratio < self.min_free_memory_ratio:
            issues.append(f"Would leave insufficient free memory: {remaining_ratio:.2%} < {self.min_free_memory_ratio:.2%}")
        
        severity = "error" if any("Insufficient" in issue for issue in issues) else "warning"
        is_valid = severity != "error"
        
        if not issues:
            issues = [f"Memory validation passed (need {estimated_gb:.2f}GB, have {available_gb:.2f}GB)"]
            severity = "info"
        
        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(issues),
            severity=severity,
            metadata={
                "estimated_memory_gb": estimated_gb,
                "available_memory_gb": available_gb,
                "total_memory_gb": total_gb
            }
        )


class ResultValidator(RobustValidator):
    """Validate causal discovery results."""
    
    def validate(self, result) -> ValidationResult:
        """Validate a CausalResult object."""
        issues = []
        
        # Check if result has required attributes
        required_attrs = ['adjacency_matrix', 'confidence_scores', 'method_used', 'metadata']
        for attr in required_attrs:
            if not hasattr(result, attr):
                issues.append(f"Result missing required attribute: {attr}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                message="; ".join(issues),
                severity="error"
            )
        
        # Validate adjacency matrix
        adj_matrix = result.adjacency_matrix
        if not isinstance(adj_matrix, np.ndarray):
            issues.append("Adjacency matrix must be numpy array")
        elif adj_matrix.ndim != 2:
            issues.append("Adjacency matrix must be 2-dimensional")
        elif adj_matrix.shape[0] != adj_matrix.shape[1]:
            issues.append("Adjacency matrix must be square")
        else:
            # Check for valid values (0 or 1)
            unique_vals = np.unique(adj_matrix)
            if not all(val in [0, 1] for val in unique_vals):
                issues.append("Adjacency matrix should contain only 0 and 1")
            
            # Check diagonal (should be 0)
            if np.diag(adj_matrix).any():
                issues.append("Adjacency matrix diagonal should be zero (no self-loops)")
        
        # Validate confidence scores
        conf_scores = result.confidence_scores
        if not isinstance(conf_scores, np.ndarray):
            issues.append("Confidence scores must be numpy array")
        elif conf_scores.shape != adj_matrix.shape:
            issues.append("Confidence scores shape must match adjacency matrix")
        elif (conf_scores < 0).any() or (conf_scores > 1).any():
            issues.append("Confidence scores should be in range [0, 1]")
        
        # Validate metadata
        if not isinstance(result.metadata, dict):
            issues.append("Metadata must be a dictionary")
        
        severity = "error" if any("must be" in issue or "missing" in issue.lower() for issue in issues) else "warning"
        is_valid = severity != "error"
        
        if not issues:
            issues = ["Result validation passed"]
            severity = "info"
        
        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(issues),
            severity=severity,
            metadata={
                "adjacency_shape": adj_matrix.shape if hasattr(adj_matrix, 'shape') else None,
                "n_edges": int(np.sum(adj_matrix)) if isinstance(adj_matrix, np.ndarray) else None
            }
        )


class RobustValidationSuite:
    """Comprehensive validation suite for causal discovery."""
    
    def __init__(self):
        self.validators = {
            'data_quality': DataQualityValidator(),
            'memory': MemoryValidator(),
            'result': ResultValidator()
        }
        
        # Common parameter specifications
        self.common_param_specs = {
            'threshold': {'type': float, 'min': 0.0, 'max': 1.0},
            'alpha': {'type': float, 'min': 0.001, 'max': 0.5},
            'n_bins': {'type': int, 'min': 2, 'max': 50},
            'max_parents': {'type': int, 'min': 0, 'max': 10},
            'bootstrap_samples': {'type': int, 'min': 10, 'max': 1000}
        }
    
    def validate_preprocessing(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data before preprocessing."""
        results = []
        
        # Data quality validation
        results.append(self.validators['data_quality'].validate(data))
        
        # Memory validation
        data_size = data.memory_usage(deep=True).sum()
        results.append(self.validators['memory'].validate(data_size))
        
        return results
    
    def validate_parameters(self, method_name: str, params: Dict[str, Any]) -> ValidationResult:
        """Validate algorithm parameters."""
        # Use common specs plus method-specific ones
        param_specs = self.common_param_specs.copy()
        
        # Add method-specific specifications
        if method_name == 'BayesianNetwork':
            param_specs.update({
                'score_method': {'type': str, 'allowed_values': ['bic', 'aic']},
                'use_bootstrap': {'type': bool}
            })
        elif method_name == 'ConstraintBased':
            param_specs.update({
                'independence_test': {'type': str, 'allowed_values': ['correlation']},
                'max_conditioning_set_size': {'type': int, 'min': 0, 'max': 5}
            })
        elif method_name == 'MutualInformation':
            param_specs.update({
                'discretization_method': {'type': str, 'allowed_values': ['equal_width', 'equal_frequency']},
                'use_conditional_mi': {'type': bool}
            })
        
        validator = ParameterValidator(param_specs)
        return validator.validate(params)
    
    def validate_result(self, result) -> ValidationResult:
        """Validate causal discovery result."""
        return self.validators['result'].validate(result)
    
    def run_full_validation(self, data: pd.DataFrame, method_name: str, 
                           params: Dict[str, Any], result=None) -> Dict[str, ValidationResult]:
        """Run complete validation suite."""
        validation_results = {}
        
        # Pre-processing validation
        preprocessing_results = self.validate_preprocessing(data)
        validation_results['data_quality'] = preprocessing_results[0]
        validation_results['memory'] = preprocessing_results[1]
        
        # Parameter validation
        validation_results['parameters'] = self.validate_parameters(method_name, params)
        
        # Result validation (if provided)
        if result is not None:
            validation_results['result'] = self.validate_result(result)
        
        return validation_results
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results.values() if result.is_valid)
        
        errors = [name for name, result in validation_results.items() 
                 if result.severity == 'error']
        warnings = [name for name, result in validation_results.items() 
                   if result.severity == 'warning']
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': passed_checks / total_checks,
            'errors': errors,
            'warnings': warnings,
            'overall_status': 'passed' if not errors else 'failed'
        }