"""Monitoring and health check utilities."""

import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage_start: float = 0.0
    cpu_usage_end: float = 0.0
    memory_usage_start: float = 0.0
    memory_usage_end: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration,
            'cpu_usage_start_percent': self.cpu_usage_start,
            'cpu_usage_end_percent': self.cpu_usage_end,
            'memory_usage_start_mb': self.memory_usage_start,
            'memory_usage_end_mb': self.memory_usage_end,
            'memory_delta_mb': self.memory_usage_end - self.memory_usage_start,
            **self.custom_metrics
        }


class PerformanceMonitor:
    """Monitor performance metrics during causal discovery."""
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        """Initialize performance monitor.
        
        Args:
            enable_detailed_monitoring: Whether to collect detailed system metrics
        """
        self.enable_detailed = enable_detailed_monitoring
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
    
    def start_monitoring(self, operation_name: str = "causal_discovery") -> PerformanceMetrics:
        """Start monitoring an operation.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Returns:
            Performance metrics object
        """
        with self._lock:
            metrics = PerformanceMetrics(start_time=time.time())
            
            if self.enable_detailed and PSUTIL_AVAILABLE:
                metrics.cpu_usage_start = psutil.cpu_percent(interval=0.1)
                metrics.memory_usage_start = psutil.Process().memory_info().rss / 1024**2
            elif self.enable_detailed:
                metrics.cpu_usage_start = 0.0
                metrics.memory_usage_start = 0.0
            
            metrics.custom_metrics['operation_name'] = operation_name
            self.current_metrics = metrics
            
            logger.info(f"Started monitoring operation: {operation_name}")
            return metrics
    
    def stop_monitoring(self) -> Optional[PerformanceMetrics]:
        """Stop monitoring and finalize metrics.
        
        Returns:
            Finalized performance metrics
        """
        with self._lock:
            if self.current_metrics is None:
                logger.warning("No active monitoring session to stop")
                return None
            
            metrics = self.current_metrics
            metrics.end_time = time.time()
            
            if self.enable_detailed and PSUTIL_AVAILABLE:
                metrics.cpu_usage_end = psutil.cpu_percent(interval=0.1)
                metrics.memory_usage_end = psutil.Process().memory_info().rss / 1024**2
            elif self.enable_detailed:
                metrics.cpu_usage_end = 0.0
                metrics.memory_usage_end = 0.0
            
            metrics.finalize()
            self.history.append(metrics)
            self.current_metrics = None
            
            operation_name = metrics.custom_metrics.get('operation_name', 'unknown')
            logger.info(
                f"Completed monitoring operation: {operation_name}, "
                f"duration: {metrics.duration:.3f}s, "
                f"memory_delta: {metrics.memory_usage_end - metrics.memory_usage_start:.1f}MB"
            )
            
            return metrics
    
    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add a custom metric to current monitoring session."""
        with self._lock:
            if self.current_metrics is not None:
                self.current_metrics.custom_metrics[key] = value
                logger.debug(f"Added custom metric: {key} = {value}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from monitoring history."""
        if not self.history:
            return {}
        
        durations = [m.duration for m in self.history if m.duration is not None]
        memory_deltas = [
            m.memory_usage_end - m.memory_usage_start 
            for m in self.history 
            if m.memory_usage_end is not None and m.memory_usage_start is not None
        ]
        
        summary = {
            'total_operations': len(self.history),
            'avg_duration_seconds': sum(durations) / len(durations) if durations else 0,
            'min_duration_seconds': min(durations) if durations else 0,
            'max_duration_seconds': max(durations) if durations else 0,
            'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            'total_memory_delta_mb': sum(memory_deltas) if memory_deltas else 0
        }
        
        return summary


class HealthChecker:
    """System health monitoring for causal discovery toolkit."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
        self.last_check_time: Optional[datetime] = None
    
    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks.
        
        Returns:
            Dictionary with health check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': True,
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'healthy': is_healthy,
                    'duration_seconds': duration,
                    'error': None
                }
                
                if not is_healthy:
                    results['overall_healthy'] = False
                    logger.warning(f"Health check failed: {name}")
                
            except Exception as e:
                results['checks'][name] = {
                    'healthy': False,
                    'duration_seconds': 0,
                    'error': str(e)
                }
                results['overall_healthy'] = False
                logger.error(f"Health check error in {name}: {e}")
        
        self.last_check_results = results
        self.last_check_time = datetime.now()
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            return {
                'cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 1,
                'cpu_usage_percent': psutil.cpu_percent(interval=1) if PSUTIL_AVAILABLE else 0.0,
                'memory_total_gb': psutil.virtual_memory().total / 1024**3 if PSUTIL_AVAILABLE else 1.0,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3 if PSUTIL_AVAILABLE else 1.0,
                'memory_usage_percent': psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0,
                'disk_usage_percent': psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0,
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}" if PSUTIL_AVAILABLE else "3.12"
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: int = 60,
                 expected_exception: Optional[type] = None):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Seconds to wait before trying again
            expected_exception: Exception type to catch (None for all)
        """
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.expected_exception = expected_exception or Exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
        """
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > self.timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.debug("Circuit breaker: Success - reset to CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker: OPENED after {self.failure_count} failures"
            )


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global instances
global_monitor = PerformanceMonitor()
global_health_checker = HealthChecker()


def monitor_performance(operation_name: str = "operation"):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = global_monitor.start_monitoring(operation_name)
            try:
                result = func(*args, **kwargs)
                global_monitor.add_custom_metric('success', True)
                return result
            except Exception as e:
                global_monitor.add_custom_metric('success', False)
                global_monitor.add_custom_metric('error', str(e))
                raise
            finally:
                global_monitor.stop_monitoring()
        return wrapper
    return decorator


# Register default health checks
def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    memory_usage = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0
    return memory_usage < 90

def _check_cpu_usage() -> bool:
    """Check if CPU usage is within acceptable limits."""
    cpu_usage = psutil.cpu_percent(interval=1) if PSUTIL_AVAILABLE else 0.0
    return cpu_usage < 95

def _check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    disk_usage = psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0
    return disk_usage < 95


# Register default health checks
global_health_checker.register_check("memory_usage", _check_memory_usage)
global_health_checker.register_check("cpu_usage", _check_cpu_usage)
global_health_checker.register_check("disk_space", _check_disk_space)