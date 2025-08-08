"""Logging configuration for causal discovery toolkit."""

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


class CausalDiscoveryLogger:
    """Centralized logging configuration for causal discovery toolkit."""
    
    def __init__(self, 
                 name: str = "causal_discovery",
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 structured: bool = True):
        """Initialize logger configuration.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            structured: Whether to use structured JSON logging
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.structured = structured
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Remove existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        # File handler if specified
        file_handler = None
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(self.level)
        
        # Set formatters
        if self.structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if file_handler:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message'):
                log_entry[key] = value
        
        return json.dumps(log_entry)


# Global logger instance
default_logger = CausalDiscoveryLogger().get_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional logger name. If None, uses default logger.
        
    Returns:
        Configured logger instance
    """
    if name:
        return CausalDiscoveryLogger(name=name).get_logger()
    return default_logger