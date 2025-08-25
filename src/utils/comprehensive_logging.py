"""
Comprehensive Logging System: Production-Ready Observability
============================================================

Advanced logging framework for causal discovery systems with structured
logging, performance monitoring, security audit trails, and real-time
observability for production environments.

Features:
- Structured JSON logging with correlation IDs
- Performance metrics and profiling integration
- Security audit trails and compliance logging
- Distributed tracing for complex workflows
- Real-time monitoring and alerting hooks
- Log aggregation and analysis tools
- Production-ready observability stack
"""

import logging
import json
import time
import threading
import uuid
import traceback
import inspect
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
import psutil
import os
from contextlib import contextmanager

class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60  # Custom level for security events

class EventType(Enum):
    """Event types for structured logging."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ALGORITHM_START = "algorithm_start"
    ALGORITHM_COMPLETE = "algorithm_complete"
    ALGORITHM_ERROR = "algorithm_error"
    DATA_VALIDATION = "data_validation"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    USER_ACTION = "user_action"
    RESOURCE_USAGE = "resource_usage"
    CONFIGURATION_CHANGE = "configuration_change"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    event_type: str
    message: str
    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    component: Optional[str] = None
    function: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    error_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    security_context: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    io_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class SecurityContext:
    """Security context for audit logging."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    data_access_level: str = "public"
    audit_required: bool = False

class CorrelationContext:
    """Thread-local correlation context."""
    
    _local = threading.local()
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get correlation ID for current thread."""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())
        return cls._local.correlation_id
    
    @classmethod
    def set_session_id(cls, session_id: str):
        """Set session ID for current thread."""
        cls._local.session_id = session_id
    
    @classmethod
    def get_session_id(cls) -> str:
        """Get session ID for current thread."""
        if not hasattr(cls._local, 'session_id'):
            cls._local.session_id = str(uuid.uuid4())
        return cls._local.session_id
    
    @classmethod
    def clear(cls):
        """Clear correlation context."""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')
        if hasattr(cls._local, 'session_id'):
            delattr(cls._local, 'session_id')

class PerformanceProfiler:
    """Performance profiler for logging."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.process = psutil.Process()
    
    def start(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
    
    def stop(self) -> PerformanceMetrics:
        """Stop profiling and return metrics."""
        if self.start_time is None:
            return PerformanceMetrics(0, 0, 0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        return PerformanceMetrics(
            execution_time_ms=(end_time - self.start_time) * 1000,
            memory_usage_mb=end_memory - self.start_memory,
            cpu_percent=max(end_cpu, self.start_cpu)  # Take max to avoid negative values
        )

class StructuredJsonFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'correlation_id': CorrelationContext.get_correlation_id(),
            'session_id': CorrelationContext.get_session_id(),
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields from LogRecord
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                             'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                             'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                             'thread', 'threadName', 'processName', 'process', 'getMessage']:
                    try:
                        # Ensure JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)

class ComprehensiveLogger:
    """
    Comprehensive logging system for production causal discovery systems.
    
    This system provides:
    1. Structured JSON logging with correlation tracking
    2. Performance profiling and metrics integration
    3. Security audit trails and compliance logging
    4. Distributed tracing for complex workflows
    5. Real-time monitoring and alerting hooks
    6. Log aggregation and analysis capabilities
    
    Key Features:
    - Thread-safe with correlation context tracking
    - Performance metrics integration
    - Security context for audit requirements
    - Multiple output formats (JSON, plain text, structured)
    - Production-ready observability stack
    - Real-time monitoring and alerting
    """
    
    def __init__(self, 
                 name: str = "causal_discovery",
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_json_format: bool = True,
                 enable_performance_metrics: bool = True,
                 enable_security_logging: bool = True):
        """
        Initialize comprehensive logger.
        
        Args:
            name: Logger name
            log_level: Minimum log level
            log_file: Optional log file path
            enable_console: Enable console logging
            enable_json_format: Use structured JSON format
            enable_performance_metrics: Enable performance tracking
            enable_security_logging: Enable security audit logging
        """
        self.name = name
        self.log_level = log_level
        self.enable_performance_metrics = enable_performance_metrics
        self.enable_security_logging = enable_security_logging
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers(log_file, enable_console, enable_json_format)
        
        # Performance tracking
        self.performance_metrics = {}
        self._lock = threading.Lock()
        
        # Security context
        self.security_context = SecurityContext()
        
        # System info
        self.system_info = self._collect_system_info()
        
        self.info("Comprehensive logger initialized", 
                 event_type=EventType.SYSTEM_START)
    
    def _setup_handlers(self, log_file: Optional[str], 
                       enable_console: bool, 
                       enable_json_format: bool):
        """Setup log handlers."""
        
        # Choose formatter
        if enable_json_format:
            formatter = StructuredJsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
            )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Separate JSON file for structured logs
            if enable_json_format:
                json_file = str(Path(log_file).with_suffix('.json'))
                json_handler = logging.FileHandler(json_file)
                json_handler.setFormatter(StructuredJsonFormatter())
                self.logger.addHandler(json_handler)
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        
        return {
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'platform': os.name,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def set_security_context(self, context: SecurityContext):
        """Set security context for audit logging."""
        self.security_context = context
    
    def _create_log_entry(self, 
                         level: LogLevel,
                         message: str,
                         event_type: Optional[EventType] = None,
                         component: Optional[str] = None,
                         duration_ms: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         error_info: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Create structured log entry."""
        
        # Get caller information
        frame = inspect.currentframe().f_back.f_back
        function_name = frame.f_code.co_name if frame else None
        
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            event_type=event_type.value if event_type else 'general',
            message=message,
            correlation_id=CorrelationContext.get_correlation_id(),
            session_id=CorrelationContext.get_session_id(),
            user_id=self.security_context.user_id,
            component=component,
            function=function_name,
            duration_ms=duration_ms,
            metadata=metadata or {},
            tags=tags or [],
            error_info=error_info,
            performance_metrics=self._get_current_metrics() if self.enable_performance_metrics else None,
            security_context=asdict(self.security_context) if self.enable_security_logging else None
        )
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        
        try:
            process = psutil.Process()
            return {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads()
            }
        except:
            return {}
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        
        log_entry = self._create_log_entry(level, message, **kwargs)
        
        # Convert to extra fields for logging
        extra = {
            'correlation_id': log_entry.correlation_id,
            'session_id': log_entry.session_id,
            'event_type': log_entry.event_type,
            'component': log_entry.component,
            'duration_ms': log_entry.duration_ms,
            'metadata': log_entry.metadata,
            'tags': log_entry.tags,
            'performance_metrics': log_entry.performance_metrics,
            'security_context': log_entry.security_context
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        # Log with appropriate level
        if level == LogLevel.TRACE:
            self.logger.log(5, message, extra=extra)
        elif level == LogLevel.DEBUG:
            self.logger.debug(message, extra=extra)
        elif level == LogLevel.INFO:
            self.logger.info(message, extra=extra)
        elif level == LogLevel.WARNING:
            self.logger.warning(message, extra=extra)
        elif level == LogLevel.ERROR:
            self.logger.error(message, extra=extra)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, extra=extra)
        elif level == LogLevel.SECURITY:
            self.logger.log(60, f"SECURITY: {message}", extra=extra)
    
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message."""
        
        error_info = None
        if error:
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
        
        self._log(LogLevel.ERROR, message, error_info=error_info, **kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        
        error_info = None
        if error:
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
        
        self._log(LogLevel.CRITICAL, message, error_info=error_info, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security event."""
        kwargs['event_type'] = EventType.SECURITY_EVENT
        self._log(LogLevel.SECURITY, message, **kwargs)
    
    def performance(self, message: str, metrics: PerformanceMetrics, **kwargs):
        """Log performance metrics."""
        
        kwargs['event_type'] = EventType.PERFORMANCE_METRIC
        kwargs['duration_ms'] = metrics.execution_time_ms
        kwargs['metadata'] = {
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_percent': metrics.cpu_percent,
            'io_operations': metrics.io_operations,
            'cache_hits': metrics.cache_hits,
            'cache_misses': metrics.cache_misses
        }
        
        self._log(LogLevel.INFO, message, **kwargs)
    
    @contextmanager
    def performance_context(self, operation_name: str, component: Optional[str] = None):
        """Context manager for performance logging."""
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        self.info(f"Starting {operation_name}",
                 event_type=EventType.ALGORITHM_START,
                 component=component,
                 metadata={'operation': operation_name})
        
        start_time = time.time()
        
        try:
            yield profiler
            
            metrics = profiler.stop()
            duration_ms = (time.time() - start_time) * 1000
            
            self.performance(f"Completed {operation_name}",
                           metrics=metrics,
                           component=component)
            
        except Exception as e:
            metrics = profiler.stop()
            duration_ms = (time.time() - start_time) * 1000
            
            self.error(f"Failed {operation_name}",
                      error=e,
                      event_type=EventType.ALGORITHM_ERROR,
                      component=component,
                      duration_ms=duration_ms)
            raise
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None,
                           session_id: Optional[str] = None):
        """Context manager for correlation tracking."""
        
        # Store previous values
        prev_correlation_id = getattr(CorrelationContext._local, 'correlation_id', None)
        prev_session_id = getattr(CorrelationContext._local, 'session_id', None)
        
        try:
            # Set new values
            if correlation_id:
                CorrelationContext.set_correlation_id(correlation_id)
            if session_id:
                CorrelationContext.set_session_id(session_id)
            
            yield
            
        finally:
            # Restore previous values
            if prev_correlation_id:
                CorrelationContext.set_correlation_id(prev_correlation_id)
            else:
                CorrelationContext.clear()
            
            if prev_session_id:
                CorrelationContext.set_session_id(prev_session_id)

def logged_causal_discovery(logger: Optional[ComprehensiveLogger] = None,
                           component: str = "causal_algorithm"):
    """Decorator for comprehensive causal discovery logging."""
    
    if logger is None:
        logger = ComprehensiveLogger()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__name__}"
            
            with logger.performance_context(operation_name, component):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# Global logger instance
global_logger = ComprehensiveLogger(
    name="causal_discovery_system",
    log_level=LogLevel.INFO,
    enable_json_format=True,
    enable_performance_metrics=True,
    enable_security_logging=True
)

# Export main components
__all__ = [
    'ComprehensiveLogger',
    'LogLevel',
    'EventType',
    'LogEntry',
    'PerformanceMetrics',
    'SecurityContext',
    'CorrelationContext',
    'PerformanceProfiler',
    'logged_causal_discovery',
    'global_logger'
]