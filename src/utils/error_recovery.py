"""Error recovery and resilient execution utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import logging
import time
import warnings
from functools import wraps
from dataclasses import dataclass
import traceback
from concurrent.futures import TimeoutError
import psutil
import gc


logger = logging.getLogger(__name__)


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    name: str
    action: Callable
    condition: Callable[[Exception], bool]
    max_attempts: int = 3
    description: str = ""


class ResilientExecutor:
    """Resilient execution with automatic error recovery."""
    
    def __init__(self, max_total_attempts: int = 5):
        self.max_total_attempts = max_total_attempts
        self.recovery_actions = []
        self._execution_history = []
    
    def add_recovery_action(self, recovery: RecoveryAction):
        """Add a recovery action."""
        self.recovery_actions.append(recovery)
        logger.info(f"Added recovery action: {recovery.name}")
    
    def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic recovery on failure."""
        attempt = 0
        last_exception = None
        
        while attempt < self.max_total_attempts:
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_total_attempts}")
                result = func(*args, **kwargs)
                
                # Success - clear history and return
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt + 1} attempts")
                
                self._execution_history.append({
                    'attempt': attempt + 1,
                    'success': True,
                    'exception': None,
                    'recovery_used': None
                })
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Try recovery actions
                recovery_success = False
                for recovery in self.recovery_actions:
                    if recovery.condition(e):
                        logger.info(f"Applying recovery action: {recovery.name}")
                        try:
                            recovery.action()
                            recovery_success = True
                            
                            self._execution_history.append({
                                'attempt': attempt + 1,
                                'success': False,
                                'exception': str(e),
                                'recovery_used': recovery.name
                            })
                            
                            break
                        except Exception as recovery_error:
                            logger.error(f"Recovery action {recovery.name} failed: {recovery_error}")
                
                if not recovery_success:
                    self._execution_history.append({
                        'attempt': attempt + 1,
                        'success': False,
                        'exception': str(e),
                        'recovery_used': None
                    })
                
                attempt += 1
                
                # Small delay before retry
                if attempt < self.max_total_attempts:
                    time.sleep(0.5)
        
        # All attempts exhausted
        logger.error(f"All {self.max_total_attempts} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history.copy()


class MemoryRecovery:
    """Memory management and recovery utilities."""
    
    @staticmethod
    def clear_memory():
        """Clear Python memory."""
        gc.collect()
        logger.info("Cleared Python garbage collector")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def is_memory_critical(threshold_percent: float = 90.0) -> bool:
        """Check if memory usage is critical."""
        memory = psutil.virtual_memory()
        return memory.percent > threshold_percent


class DataRecovery:
    """Data-related error recovery utilities."""
    
    @staticmethod
    def handle_missing_data(data: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """Handle missing data with various strategies."""
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'forward_fill':
            return data.fillna(method='ffill')
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
    
    @staticmethod
    def handle_infinite_values(data: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with appropriate substitutes."""
        data_clean = data.copy()
        
        # Replace positive infinity with column maximum
        for col in data_clean.select_dtypes(include=[np.number]).columns:
            finite_values = data_clean[col][np.isfinite(data_clean[col])]
            if len(finite_values) > 0:
                max_val = finite_values.max()
                min_val = finite_values.min()
                
                data_clean[col] = data_clean[col].replace([np.inf], max_val * 1.1)
                data_clean[col] = data_clean[col].replace([-np.inf], min_val * 1.1)
        
        return data_clean
    
    @staticmethod
    def reduce_data_size(data: pd.DataFrame, max_samples: int = 10000) -> pd.DataFrame:
        """Reduce data size if too large."""
        if len(data) > max_samples:
            logger.warning(f"Reducing data from {len(data)} to {max_samples} samples")
            return data.sample(n=max_samples, random_state=42)
        return data
    
    @staticmethod
    def handle_constant_columns(data: pd.DataFrame, min_variance: float = 1e-6) -> pd.DataFrame:
        """Remove or handle constant columns."""
        data_clean = data.copy()
        
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        constant_cols = []
        
        for col in numeric_cols:
            if data_clean[col].var() < min_variance:
                constant_cols.append(col)
        
        if constant_cols:
            logger.warning(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            data_clean = data_clean.drop(columns=constant_cols)
        
        return data_clean


def resilient_causal_discovery(recovery_enabled: bool = True):
    """Decorator for resilient causal discovery execution."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not recovery_enabled:
                return func(*args, **kwargs)
            
            # Set up resilient executor
            executor = ResilientExecutor(max_total_attempts=3)
            
            # Add recovery actions
            
            # Memory recovery
            memory_recovery = RecoveryAction(
                name="memory_cleanup",
                action=lambda: (MemoryRecovery.clear_memory(), time.sleep(1)),
                condition=lambda e: "memory" in str(e).lower() or isinstance(e, MemoryError),
                description="Clear memory and garbage collect"
            )
            executor.add_recovery_action(memory_recovery)
            
            # Data size reduction
            def reduce_data_size():
                if len(args) > 0 and hasattr(args[0], '_data'):
                    args[0]._data = DataRecovery.reduce_data_size(args[0]._data, max_samples=5000)
                elif 'data' in kwargs:
                    kwargs['data'] = DataRecovery.reduce_data_size(kwargs['data'], max_samples=5000)
            
            size_recovery = RecoveryAction(
                name="data_size_reduction",
                action=reduce_data_size,
                condition=lambda e: "out of memory" in str(e).lower() or "too large" in str(e).lower(),
                description="Reduce data size to manageable level"
            )
            executor.add_recovery_action(size_recovery)
            
            # Numerical stability recovery
            def fix_numerical_issues():
                if len(args) > 0 and hasattr(args[0], '_data'):
                    args[0]._data = DataRecovery.handle_infinite_values(args[0]._data)
                elif 'data' in kwargs:
                    kwargs['data'] = DataRecovery.handle_infinite_values(kwargs['data'])
            
            numerical_recovery = RecoveryAction(
                name="numerical_stability",
                action=fix_numerical_issues,
                condition=lambda e: "singular" in str(e).lower() or "infinite" in str(e).lower() or "nan" in str(e).lower(),
                description="Handle infinite values and numerical instabilities"
            )
            executor.add_recovery_action(numerical_recovery)
            
            # Execute with recovery
            return executor.execute_with_recovery(func, *args, **kwargs)
        
        return wrapper
    return decorator


class ProgressiveExecution:
    """Progressive execution with increasing complexity."""
    
    def __init__(self):
        self.strategies = []
    
    def add_strategy(self, name: str, params: Dict[str, Any], description: str = ""):
        """Add an execution strategy."""
        self.strategies.append({
            'name': name,
            'params': params,
            'description': description
        })
    
    def execute_progressive(self, model_class, data: pd.DataFrame) -> Tuple[Any, str]:
        """Execute with progressively more complex strategies."""
        
        for i, strategy in enumerate(self.strategies):
            logger.info(f"Trying strategy {i+1}/{len(self.strategies)}: {strategy['name']}")
            
            try:
                # Create model with strategy parameters
                model = model_class(**strategy['params'])
                result = model.fit_discover(data)
                
                logger.info(f"Strategy '{strategy['name']}' succeeded")
                return result, strategy['name']
                
            except Exception as e:
                logger.warning(f"Strategy '{strategy['name']}' failed: {str(e)}")
                
                # If this is the last strategy, re-raise the exception
                if i == len(self.strategies) - 1:
                    logger.error("All strategies failed")
                    raise
                
                continue
        
        raise RuntimeError("All progressive execution strategies failed")


class SafetyWrapper:
    """Safety wrapper for causal discovery algorithms."""
    
    def __init__(self, 
                 max_execution_time: int = 300,  # 5 minutes
                 memory_limit_gb: float = 4.0,
                 enable_monitoring: bool = True):
        self.max_execution_time = max_execution_time
        self.memory_limit_gb = memory_limit_gb
        self.enable_monitoring = enable_monitoring
        
    def wrap_execution(self, func: Callable, *args, **kwargs) -> Any:
        """Wrap execution with safety monitoring."""
        start_time = time.time()
        start_memory = MemoryRecovery.get_memory_usage()
        
        try:
            # Monitor execution
            if self.enable_monitoring:
                return self._monitored_execution(func, start_time, *args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        finally:
            # Log execution statistics
            end_time = time.time()
            end_memory = MemoryRecovery.get_memory_usage()
            
            logger.info(f"Execution completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {end_memory['rss_mb']:.1f}MB (delta: {end_memory['rss_mb'] - start_memory['rss_mb']:+.1f}MB)")
    
    def _monitored_execution(self, func: Callable, start_time: float, *args, **kwargs) -> Any:
        """Execute with monitoring."""
        
        def check_constraints():
            # Check execution time
            current_time = time.time()
            if current_time - start_time > self.max_execution_time:
                raise TimeoutError(f"Execution exceeded {self.max_execution_time}s limit")
            
            # Check memory usage
            current_memory = MemoryRecovery.get_memory_usage()
            if current_memory['rss_mb'] / 1024 > self.memory_limit_gb:
                raise MemoryError(f"Memory usage {current_memory['rss_mb']/1024:.2f}GB exceeded {self.memory_limit_gb}GB limit")
        
        # Execute with periodic constraint checking
        # Note: This is a simplified version - in practice you'd want more sophisticated monitoring
        result = func(*args, **kwargs)
        check_constraints()
        
        return result