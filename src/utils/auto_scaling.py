"""Auto-scaling utilities for dynamic resource management."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources for auto-scaling decisions."""
    
    def __init__(self):
        """Initialize resource monitor."""
        pass
    
    def get_system_load(self) -> float:
        """Get normalized system load (0.0 to 1.0)."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Combine CPU and memory load
        cpu_load = cpu_percent / 100.0
        memory_load = memory_percent / 100.0
        
        # Weight different factors
        combined_load = (cpu_load * 0.6 + memory_load * 0.4)
        
        return min(combined_load, 1.0)


class AutoScaler:
    """Automatically scale resources based on system load."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = 8,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        """Initialize auto-scaler."""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Current state
        self.current_workers = min_workers
        
        logger.info(f"Initialized AutoScaler: {min_workers}-{max_workers} workers")
    
    def adjust_workers(self, current_load: float) -> int:
        """Adjust number of workers based on current load."""
        old_workers = self.current_workers
        
        # Check for scale up
        if current_load > self.scale_up_threshold and self.current_workers < self.max_workers:
            new_workers = min(self.max_workers, self.current_workers + 1)
            self.current_workers = new_workers
            logger.info(f"Scaled UP: {old_workers} -> {new_workers} workers (load: {current_load:.1%})")
        
        # Check for scale down
        elif current_load < self.scale_down_threshold and self.current_workers > self.min_workers:
            new_workers = max(self.min_workers, self.current_workers - 1)
            self.current_workers = new_workers
            logger.info(f"Scaled DOWN: {old_workers} -> {new_workers} workers (load: {current_load:.1%})")
        
        return self.current_workers