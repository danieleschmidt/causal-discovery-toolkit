"""Comprehensive error handling and recovery for causal discovery algorithms."""

import traceback
import functools
import logging
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import pandas as pd

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for errors."""
    function_name: str
    module_name: str
    input_shape: Optional[Tuple[int, ...]] = None
    parameters: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CausalDiscoveryError(Exception):
    """Base exception for causal discovery operations."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, 
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.context = context
        self.cause = cause
        self.timestamp = time.time()


class DataValidationError(CausalDiscoveryError):
    """Raised when input data validation fails."""
    pass


class AlgorithmError(CausalDiscoveryError):
    """Raised when causal discovery algorithm fails."""
    pass


class ConvergenceError(CausalDiscoveryError):
    """Raised when algorithm fails to converge."""
    pass


class MemoryError(CausalDiscoveryError):
    """Raised when operation exceeds memory limits."""
    pass


class TimeoutError(CausalDiscoveryError):
    """Raised when operation exceeds time limits."""
    pass


class ConfigurationError(CausalDiscoveryError):
    """Raised when invalid configuration is provided."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery for causal discovery."""
    
    def __init__(self, enable_recovery: bool = True, max_retries: int = 3):
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        self.error_log = []
        self.recovery_strategies = self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self) -> Dict[Type[Exception], Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            np.linalg.LinAlgError: self._handle_linalg_error,
            ValueError: self._handle_value_error,
            MemoryError: self._handle_memory_error,
            KeyboardInterrupt: self._handle_keyboard_interrupt,
            TimeoutError: self._handle_timeout_error,
            OverflowError: self._handle_overflow_error,
            RuntimeError: self._handle_runtime_error
        }
    
    def handle_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle an error with appropriate recovery strategy."""
        self.error_log.append({
            'error': error,
            'context': context,
            'timestamp': time.time()
        })
        
        logger.error(f"Error in {context.function_name}: {error}")
        logger.debug(f"Error context: {context}")
        
        if not self.enable_recovery:
            raise CausalDiscoveryError(str(error), context, error)
        
        # Try to find appropriate recovery strategy
        for error_type, recovery_func in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    return recovery_func(error, context)
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
                    break
        
        # No recovery possible, re-raise as CausalDiscoveryError
        raise CausalDiscoveryError(str(error), context, error)
    
    def _handle_linalg_error(self, error: np.linalg.LinAlgError, context: ErrorContext) -> Any:
        """Handle linear algebra errors (singular matrices, etc.)."""
        logger.warning(f"Linear algebra error in {context.function_name}: {error}")
        
        if "singular matrix" in str(error).lower():
            logger.info("Attempting recovery with regularization")
            # Return a small regularization suggestion
            return {'recovery_strategy': 'add_regularization', 'lambda': 1e-6}
        
        elif "not positive definite" in str(error).lower():
            logger.info("Attempting recovery with positive definite adjustment")
            return {'recovery_strategy': 'make_positive_definite', 'min_eigenvalue': 1e-6}
        
        else:
            # Fallback to robust alternative
            return {'recovery_strategy': 'use_robust_alternative', 'method': 'pseudo_inverse'}
    
    def _handle_value_error(self, error: ValueError, context: ErrorContext) -> Any:
        """Handle value errors (invalid inputs, etc.)."""
        error_msg = str(error).lower()
        
        if "empty" in error_msg or "no data" in error_msg:
            logger.warning("Empty data detected, suggesting data generation")
            return {'recovery_strategy': 'generate_synthetic_data', 'min_samples': 100}
        
        elif "nan" in error_msg or "inf" in error_msg:
            logger.warning("Invalid values detected, suggesting cleaning")
            return {'recovery_strategy': 'clean_data', 'remove_invalid': True}
        
        elif "shape" in error_msg or "dimension" in error_msg:
            logger.warning("Dimension mismatch detected")
            return {'recovery_strategy': 'reshape_data', 'auto_adjust': True}
        
        else:
            return {'recovery_strategy': 'validate_inputs', 'strict': False}
    
    def _handle_memory_error(self, error: MemoryError, context: ErrorContext) -> Any:
        """Handle memory errors."""
        logger.warning(f"Memory error in {context.function_name}")
        return {
            'recovery_strategy': 'reduce_memory_usage',
            'batch_processing': True,
            'chunk_size': 1000,
            'use_sparse': True
        }
    
    def _handle_keyboard_interrupt(self, error: KeyboardInterrupt, context: ErrorContext) -> Any:
        """Handle user interruption."""
        logger.info("User interrupted operation")
        return {'recovery_strategy': 'graceful_shutdown', 'save_partial_results': True}
    
    def _handle_timeout_error(self, error: TimeoutError, context: ErrorContext) -> Any:
        """Handle timeout errors."""
        logger.warning(f"Timeout in {context.function_name}")
        return {
            'recovery_strategy': 'reduce_complexity',
            'simplified_algorithm': True,
            'max_iterations': 100
        }
    
    def _handle_overflow_error(self, error: OverflowError, context: ErrorContext) -> Any:
        """Handle numerical overflow errors."""
        logger.warning(f"Numerical overflow in {context.function_name}")
        return {
            'recovery_strategy': 'numerical_stabilization',
            'use_log_space': True,
            'clip_values': True,
            'max_value': 1e10
        }
    
    def _handle_runtime_error(self, error: RuntimeError, context: ErrorContext) -> Any:
        """Handle runtime errors."""
        logger.warning(f"Runtime error in {context.function_name}: {error}")
        return {
            'recovery_strategy': 'fallback_algorithm',
            'use_simple_method': True
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors."""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}, 'functions_with_errors': []}
        
        error_types = {}
        functions_with_errors = set()
        
        for entry in self.error_log:
            error_type = type(entry['error']).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            functions_with_errors.add(entry['context'].function_name)
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'functions_with_errors': list(functions_with_errors),
            'last_error_time': self.error_log[-1]['timestamp']
        }


def robust_execution(max_retries: int = 3, fallback_result: Any = None, 
                    enable_recovery: bool = True):
    """Decorator for robust execution of functions with error handling and retries."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(enable_recovery, max_retries)
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Create error context
                    context = ErrorContext(
                        function_name=func.__name__,
                        module_name=func.__module__,
                        parameters={k: str(v)[:100] for k, v in kwargs.items()}  # Truncate long params
                    )
                    
                    # Try to extract input shape if possible
                    if args and hasattr(args[0], 'shape'):
                        context.input_shape = args[0].shape
                    elif 'data' in kwargs and hasattr(kwargs['data'], 'shape'):
                        context.input_shape = kwargs['data'].shape
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Log successful execution after previous failures
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                        
                        if fallback_result is not None:
                            logger.info(f"Returning fallback result for {func.__name__}")
                            return fallback_result
                        
                        # Try error handler one last time
                        try:
                            recovery_info = error_handler.handle_error(e, context)
                            logger.info(f"Error handler returned recovery info: {recovery_info}")
                            
                            if fallback_result is not None:
                                return fallback_result
                            else:
                                raise CausalDiscoveryError(
                                    f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}",
                                    context, e
                                )
                        except Exception:
                            raise CausalDiscoveryError(
                                f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}",
                                context, e
                            )
                    else:
                        # Not final attempt, try recovery
                        try:
                            recovery_info = error_handler.handle_error(e, context)
                            logger.info(f"Attempting recovery for {func.__name__}: {recovery_info}")
                            
                            # Apply recovery strategies to function arguments if possible
                            if recovery_info.get('recovery_strategy') == 'add_regularization':
                                if 'regularization' in kwargs:
                                    kwargs['regularization'] = recovery_info.get('lambda', 1e-6)
                            elif recovery_info.get('recovery_strategy') == 'clean_data':
                                if args and hasattr(args[0], 'dropna'):
                                    args = (args[0].dropna(),) + args[1:]
                            
                            # Wait before retry
                            time.sleep(0.1 * (attempt + 1))
                            
                        except Exception as recovery_error:
                            logger.warning(f"Recovery failed for {func.__name__}: {recovery_error}")
                            time.sleep(0.1 * (attempt + 1))
            
            # Should not reach here, but just in case
            raise CausalDiscoveryError(
                f"Unexpected error in robust_execution for {func.__name__}",
                context, last_exception
            )
        
        return wrapper
    return decorator


@contextmanager
def safe_execution(context_name: str = "operation"):
    """Context manager for safe execution with automatic error handling."""
    error_handler = ErrorHandler()
    start_time = time.time()
    
    try:
        logger.debug(f"Starting safe execution: {context_name}")
        yield error_handler
        
    except Exception as e:
        context = ErrorContext(
            function_name=context_name,
            module_name=__name__,
            timestamp=start_time
        )
        
        try:
            recovery_info = error_handler.handle_error(e, context)
            logger.info(f"Safe execution recovered: {recovery_info}")
        except Exception:
            logger.error(f"Safe execution failed completely: {e}")
            raise
    
    finally:
        execution_time = time.time() - start_time
        logger.debug(f"Safe execution completed: {context_name} in {execution_time:.3f}s")


def validate_input_safely(data: Any, expected_type: Type = pd.DataFrame, 
                          min_samples: int = 10, min_features: int = 2) -> pd.DataFrame:
    """Safely validate and convert input data."""
    
    with safe_execution("input_validation"):
        # Type validation
        if not isinstance(data, expected_type):
            if isinstance(data, np.ndarray) and expected_type == pd.DataFrame:
                logger.info("Converting numpy array to DataFrame")
                data = pd.DataFrame(data)
            else:
                raise DataValidationError(f"Expected {expected_type}, got {type(data)}")
        
        # Shape validation
        if hasattr(data, 'shape'):
            if len(data.shape) != 2:
                raise DataValidationError(f"Expected 2D data, got {len(data.shape)}D")
            
            n_samples, n_features = data.shape
            
            if n_samples < min_samples:
                raise DataValidationError(f"Too few samples: {n_samples} < {min_samples}")
            
            if n_features < min_features:
                raise DataValidationError(f"Too few features: {n_features} < {min_features}")
        
        # Content validation
        if isinstance(data, pd.DataFrame):
            # Check for non-numeric data
            non_numeric = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                logger.warning(f"Non-numeric columns found: {list(non_numeric)}")
                data = data.select_dtypes(include=[np.number])
            
            # Check for invalid values
            if data.isnull().any().any():
                logger.warning("Missing values detected, applying forward fill")
                data = data.fillna(method='ffill').fillna(method='bfill')
            
            if np.isinf(data.values).any():
                logger.warning("Infinite values detected, clipping to finite range")
                data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return data


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'half_open'
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise CausalDiscoveryError(
                        f"Circuit breaker open for {func.__name__}. "
                        f"Too many failures ({self.failure_count})"
                    )
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'half_open':
                    self.state = 'closed'
                    self.failure_count = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.warning(f"Circuit breaker opened for {func.__name__}")
                
                raise e
        
        return wrapper


def create_fallback_result(original_function: str, input_shape: Tuple[int, int], 
                          error: Exception) -> Dict[str, Any]:
    """Create a safe fallback result when algorithms fail completely."""
    n_samples, n_features = input_shape
    
    logger.warning(f"Creating fallback result for {original_function} due to: {error}")
    
    # Create minimal valid result structure
    fallback_adjacency = np.zeros((n_features, n_features))
    fallback_confidence = np.zeros((n_features, n_features))
    
    return {
        'adjacency_matrix': fallback_adjacency,
        'confidence_scores': fallback_confidence,
        'method_used': f'Fallback_{original_function}',
        'metadata': {
            'is_fallback': True,
            'original_error': str(error),
            'n_variables': n_features,
            'n_edges': 0,
            'fallback_reason': 'Algorithm failed, returning empty graph'
        }
    }