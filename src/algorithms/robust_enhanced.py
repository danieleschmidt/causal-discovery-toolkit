"""Enhanced robust causal discovery with comprehensive error handling and security."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import traceback
import warnings
from dataclasses import dataclass

try:
    from .base import CausalDiscoveryModel, CausalResult, SimpleLinearCausalModel
    from ..utils.validation import DataValidator, ParameterValidator, ValidationResult
    from ..utils.security import DataSecurityValidator, SecurityResult, global_audit_logger
    from ..utils.logging_config import get_logger
    from ..utils.error_handling import ErrorHandler, CausalDiscoveryError, DataValidationError
    from ..utils.monitoring import HealthMonitor, monitor_performance, CircuitBreaker
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from algorithms.base import CausalDiscoveryModel, CausalResult, SimpleLinearCausalModel
    from utils.validation import DataValidator, ParameterValidator, ValidationResult
    from utils.security import DataSecurityValidator, SecurityResult, global_audit_logger
    from utils.logging_config import get_logger
    from utils.error_handling import ErrorHandler, CausalDiscoveryError, DataValidationError
    from utils.monitoring import HealthMonitor, monitor_performance, CircuitBreaker


logger = get_logger(__name__)


@dataclass
class RobustCausalResult(CausalResult):
    """Enhanced result with validation and security information."""
    validation_result: ValidationResult
    security_result: SecurityResult
    quality_score: float
    processing_time: float
    warnings_raised: List[str]
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.quality_score < 0 or self.quality_score > 1:
            raise ValueError(f"Quality score must be between 0 and 1, got {self.quality_score}")
        if self.processing_time < 0:
            raise ValueError(f"Processing time cannot be negative, got {self.processing_time}")


class RobustCausalDiscoveryModel(CausalDiscoveryModel):
    """Robust causal discovery model with comprehensive error handling, validation, and security."""
    
    def __init__(self, 
                 base_model: Optional[CausalDiscoveryModel] = None,
                 threshold: float = 0.3,
                 enable_security: bool = True,
                 strict_validation: bool = True,
                 max_retries: int = 3,
                 circuit_breaker_threshold: int = 5,
                 user_id: Optional[str] = None,
                 **kwargs):
        """Initialize robust causal discovery model.
        
        Args:
            base_model: Underlying model to wrap. If None, uses SimpleLinearCausalModel
            threshold: Correlation threshold for causal detection
            enable_security: Whether to enable security validation
            strict_validation: Whether to treat warnings as errors
            max_retries: Maximum number of retries on failures
            circuit_breaker_threshold: Number of failures before opening circuit breaker
            user_id: User identifier for audit logging
            **kwargs: Additional parameters for base model
        """
        super().__init__(**kwargs)
        
        # Initialize base model
        if base_model is None:
            self.base_model = SimpleLinearCausalModel(threshold=threshold, **kwargs)
        else:
            self.base_model = base_model
        
        # Configuration
        self.threshold = threshold
        self.enable_security = enable_security
        self.strict_validation = strict_validation
        self.max_retries = max_retries
        self.user_id = user_id or "anonymous"
        
        # Initialize components
        self.data_validator = DataValidator(strict=strict_validation)
        self.security_validator = DataSecurityValidator() if enable_security else None
        self.error_handler = ErrorHandler()
        self.health_monitor = HealthMonitor()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout_seconds=60
        )
        
        # State tracking
        self.last_validation_result: Optional[ValidationResult] = None
        self.last_security_result: Optional[SecurityResult] = None
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized RobustCausalDiscoveryModel for user {self.user_id}")
    
    @monitor_performance
    def fit(self, data: pd.DataFrame, **kwargs) -> 'RobustCausalDiscoveryModel':
        """Fit the model with comprehensive validation and error handling.
        
        Args:
            data: Input data for fitting
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            DataValidationError: If data validation fails
            CausalDiscoveryError: If model fitting fails
        """
        operation_id = f"fit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Audit logging
            global_audit_logger.log_data_operation(
                user=self.user_id,
                operation="fit_model",
                data_info={
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "memory_mb": data.memory_usage(deep=True).sum() / 1024**2
                }
            )
            
            # Health check
            self.health_monitor.check_system_health()
            
            # Circuit breaker check
            if not self.circuit_breaker.can_proceed():
                raise CausalDiscoveryError("Circuit breaker is open - too many recent failures")
            
            # Comprehensive data validation
            validation_result = self._validate_input_data(data)
            self.last_validation_result = validation_result
            
            if not validation_result.is_valid:
                error_msg = f"Data validation failed: {'; '.join(validation_result.errors)}"
                self.circuit_breaker.record_failure()
                raise DataValidationError(error_msg)
            
            # Security validation
            if self.enable_security:
                security_result = self._validate_data_security(data)
                self.last_security_result = security_result
                
                if security_result.risk_level in ["HIGH", "CRITICAL"]:
                    warning_msg = f"Security concerns detected: {'; '.join(security_result.issues)}"
                    warnings.warn(warning_msg, UserWarning)
                    logger.warning(f"Security validation warning for user {self.user_id}: {warning_msg}")
            
            # Fit the base model with retry logic
            self._fit_with_retries(data, **kwargs)
            
            # Record successful operation
            self.circuit_breaker.record_success()
            self.is_fitted = True
            
            # Log success
            logger.info(f"Successfully fitted model for user {self.user_id} (operation: {operation_id})")
            
            return self
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            error_context_dict = {
                "operation_id": operation_id,
                "user_id": self.user_id,
                "data_shape": data.shape if hasattr(data, 'shape') else 'unknown',
                "error_type": type(e).__name__
            }
            
            # Create proper ErrorContext object
            try:
                from ..utils.error_handling import ErrorContext
            except ImportError:
                from utils.error_handling import ErrorContext
            
            error_context = ErrorContext(
                function_name="fit",
                module_name=__name__,
                input_shape=data.shape if hasattr(data, 'shape') else None,
                parameters=error_context_dict
            )
            
            # Enhanced error handling
            handled_error = self.error_handler.handle_error(e, error_context)
            logger.error(f"Model fitting failed for user {self.user_id}: {handled_error}")
            
            # Audit log the failure
            global_audit_logger.log_data_operation(
                user=self.user_id,
                operation="fit_model_failure",
                data_info=error_context_dict
            )
            
            raise handled_error
    
    @monitor_performance  
    def discover(self, data: Optional[pd.DataFrame] = None, **kwargs) -> RobustCausalResult:
        """Discover causal relationships with comprehensive validation.
        
        Args:
            data: Optional new data, uses fitted data if None
            **kwargs: Additional parameters
            
        Returns:
            RobustCausalResult with enhanced information
            
        Raises:
            CausalDiscoveryError: If discovery fails
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
        
        operation_id = f"discover_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        warnings_raised = []
        
        try:
            # Circuit breaker check
            if not self.circuit_breaker.can_proceed():
                raise CausalDiscoveryError("Circuit breaker is open - too many recent failures")
            
            # Use provided data or fallback to fitted data
            if data is not None:
                # Validate new data
                validation_result = self._validate_input_data(data)
                if not validation_result.is_valid:
                    raise DataValidationError(f"Input data validation failed: {'; '.join(validation_result.errors)}")
                
                # Security check for new data
                if self.enable_security:
                    security_result = self._validate_data_security(data)
                    if security_result.risk_level in ["HIGH", "CRITICAL"]:
                        warning_msg = f"Security concerns in input data: {'; '.join(security_result.issues)}"
                        warnings.warn(warning_msg, UserWarning)
                        warnings_raised.append(warning_msg)
            else:
                # Use cached validation results
                validation_result = self.last_validation_result
                security_result = self.last_security_result
            
            # Perform causal discovery with retry logic
            base_result = self._discover_with_retries(data, **kwargs)
            
            # Calculate processing time
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result, base_result)
            
            # Create enhanced result
            robust_result = RobustCausalResult(
                adjacency_matrix=base_result.adjacency_matrix,
                confidence_scores=base_result.confidence_scores,
                method_used=f"Robust_{base_result.method_used}",
                metadata={
                    **base_result.metadata,
                    "operation_id": operation_id,
                    "user_id": self.user_id,
                    "model_type": "RobustCausalDiscoveryModel",
                    "validation_passed": validation_result.is_valid if validation_result else False,
                    "security_level": security_result.risk_level if security_result else "N/A"
                },
                validation_result=validation_result,
                security_result=security_result,
                quality_score=quality_score,
                processing_time=processing_time,
                warnings_raised=warnings_raised
            )
            
            # Record successful operation
            self.circuit_breaker.record_success()
            
            # Update processing history
            self.processing_history.append({
                "operation_id": operation_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "quality_score": quality_score,
                "success": True
            })
            
            # Audit logging
            global_audit_logger.log_data_operation(
                user=self.user_id,
                operation="causal_discovery",
                data_info={
                    "operation_id": operation_id,
                    "n_edges": base_result.metadata.get('n_edges', 0),
                    "quality_score": quality_score,
                    "processing_time": processing_time
                }
            )
            
            logger.info(f"Successfully completed causal discovery for user {self.user_id} "
                       f"(operation: {operation_id}, quality: {quality_score:.3f})")
            
            return robust_result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            error_context_dict = {
                "operation_id": operation_id,
                "user_id": self.user_id,
                "processing_time": processing_time,
                "error_type": type(e).__name__
            }
            
            # Create proper ErrorContext object
            try:
                from ..utils.error_handling import ErrorContext
            except ImportError:
                from utils.error_handling import ErrorContext
            
            error_context = ErrorContext(
                function_name="discover",
                module_name=__name__,
                parameters=error_context_dict
            )
            
            # Enhanced error handling
            handled_error = self.error_handler.handle_error(e, error_context)
            logger.error(f"Causal discovery failed for user {self.user_id}: {handled_error}")
            
            # Record failed operation
            self.processing_history.append({
                "operation_id": operation_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "quality_score": 0.0,
                "success": False,
                "error": str(handled_error)
            })
            
            # Audit log the failure
            global_audit_logger.log_data_operation(
                user=self.user_id,
                operation="causal_discovery_failure",
                data_info=error_context_dict
            )
            
            raise handled_error
    
    def _validate_input_data(self, data: pd.DataFrame) -> ValidationResult:
        """Perform comprehensive data validation."""
        try:
            # Parameter validation
            param_result = ParameterValidator.validate_threshold(self.threshold)
            if not param_result.is_valid:
                raise DataValidationError(f"Invalid threshold parameter: {'; '.join(param_result.errors)}")
            
            # Data validation
            validation_result = self.data_validator.validate_input_data(data)
            
            # Additional sample size validation
            sample_result = ParameterValidator.validate_sample_size(data.shape[0], data.shape[1])
            if not sample_result.is_valid:
                validation_result.errors.extend(sample_result.errors)
                validation_result.is_valid = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise DataValidationError(f"Data validation error: {str(e)}")
    
    def _validate_data_security(self, data: pd.DataFrame) -> SecurityResult:
        """Perform comprehensive security validation."""
        try:
            return self.security_validator.validate_data_security(data)
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            # Return a safe default instead of failing
            return SecurityResult(
                is_secure=False,
                issues=[f"Security validation error: {str(e)}"],
                recommendations=["Review data security before proceeding"],
                risk_level="MEDIUM"
            )
    
    def _fit_with_retries(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit base model with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.base_model.fit(data, **kwargs)
                return
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Fit attempt {attempt + 1} failed, retrying: {str(e)}")
                    continue
                else:
                    logger.error(f"All fit attempts failed after {self.max_retries} tries")
                    break
        
        raise CausalDiscoveryError(f"Model fitting failed after {self.max_retries} attempts: {str(last_exception)}")
    
    def _discover_with_retries(self, data: Optional[pd.DataFrame] = None, **kwargs) -> CausalResult:
        """Perform causal discovery with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return self.base_model.discover(data, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Discovery attempt {attempt + 1} failed, retrying: {str(e)}")
                    continue
                else:
                    logger.error(f"All discovery attempts failed after {self.max_retries} tries")
                    break
        
        raise CausalDiscoveryError(f"Causal discovery failed after {self.max_retries} attempts: {str(last_exception)}")
    
    def _calculate_quality_score(self, validation_result: ValidationResult, 
                                causal_result: CausalResult) -> float:
        """Calculate overall quality score for the analysis."""
        if not validation_result or not validation_result.is_valid:
            return 0.0
        
        # Base quality from validation
        base_score = 0.5
        
        # Bonus for good data quality
        if validation_result.metadata:
            n_samples = validation_result.metadata.get('n_samples', 0)
            n_features = validation_result.metadata.get('n_features', 1)
            
            # Sample size bonus
            if n_samples >= 1000:
                base_score += 0.2
            elif n_samples >= 500:
                base_score += 0.1
            
            # Feature ratio bonus
            sample_to_feature_ratio = n_samples / max(n_features, 1)
            if sample_to_feature_ratio >= 100:
                base_score += 0.2
            elif sample_to_feature_ratio >= 50:
                base_score += 0.1
        
        # Penalty for warnings
        if validation_result.warnings:
            base_score -= len(validation_result.warnings) * 0.05
        
        # Security bonus
        if self.last_security_result:
            if self.last_security_result.risk_level == "LOW":
                base_score += 0.1
            elif self.last_security_result.risk_level in ["HIGH", "CRITICAL"]:
                base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_type": "RobustCausalDiscoveryModel",
            "base_model": type(self.base_model).__name__,
            "user_id": self.user_id,
            "is_fitted": self.is_fitted,
            "enable_security": self.enable_security,
            "strict_validation": self.strict_validation,
            "max_retries": self.max_retries,
            "circuit_breaker_state": self.circuit_breaker.state.name,
            "processing_history_length": len(self.processing_history),
            "last_quality_score": self.processing_history[-1].get('quality_score') if self.processing_history else None
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the model and its components."""
        try:
            health_checks = self.health_monitor.run_health_checks()
            
            return {
                "overall_health": "healthy" if all(check['status'] == 'healthy' for check in health_checks.values()) else "unhealthy",
                "circuit_breaker_state": self.circuit_breaker.state.name,
                "health_checks": health_checks,
                "recent_success_rate": self._calculate_recent_success_rate(),
                "average_processing_time": self._calculate_average_processing_time()
            }
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "overall_health": "unknown",
                "error": str(e)
            }
    
    def _calculate_recent_success_rate(self, window_size: int = 10) -> float:
        """Calculate recent success rate."""
        if not self.processing_history:
            return 1.0
        
        recent_operations = self.processing_history[-window_size:]
        successes = sum(1 for op in recent_operations if op.get('success', False))
        return successes / len(recent_operations)
    
    def _calculate_average_processing_time(self, window_size: int = 10) -> float:
        """Calculate average processing time."""
        if not self.processing_history:
            return 0.0
        
        recent_operations = self.processing_history[-window_size:]
        processing_times = [op.get('processing_time', 0) for op in recent_operations if op.get('success', False)]
        
        return sum(processing_times) / len(processing_times) if processing_times else 0.0