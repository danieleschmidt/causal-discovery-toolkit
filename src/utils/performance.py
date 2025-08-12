"""Performance optimization utilities for causal discovery."""

import time
import functools
from typing import Dict, Any, Optional, Callable, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from dataclasses import dataclass
import pickle
import hashlib
import os
from pathlib import Path

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access count and timestamp."""
        self.access_count += 1


class AdaptiveCache:
    """Adaptive cache with LRU eviction and performance-based TTL."""
    
    def __init__(self, 
                 max_size: int = 100,
                 max_memory_mb: float = 512,
                 default_ttl: Optional[float] = 3600):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = []  # For LRU tracking
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_pressure_evictions': 0
        }
        
        # Adaptive parameters
        self._avg_hit_rate = 0.5
        self._adjustment_factor = 0.1
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._access_order.remove(key)
                self._stats['misses'] += 1
                return None
            
            # Update access info
            entry.touch()
            self._update_access_order(key)
            self._stats['hits'] += 1
            
            # Update adaptive parameters
            self._update_adaptive_params()
            
            logger.debug(f"Cache hit for key: {key}")
            return entry.data
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
        """
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Use adaptive TTL if not specified
            effective_ttl = ttl or self._calculate_adaptive_ttl()
            
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=effective_ttl
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)
            
            # Evict if necessary
            self._evict_if_necessary()
            
            logger.debug(f"Cached item with key: {key}, size: {size_bytes} bytes")
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': sum(entry.size_bytes for entry in self._cache.values()),
                'max_memory_bytes': self.max_memory_bytes,
                'hit_rate': hit_rate,
                'total_hits': self._stats['hits'],
                'total_misses': self._stats['misses'],
                'total_evictions': self._stats['evictions'],
                'memory_pressure_evictions': self._stats['memory_pressure_evictions']
            }
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, (pd.DataFrame, pd.Series)):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(str(value)) * 2  # Rough estimate
    
    def _calculate_adaptive_ttl(self) -> Optional[float]:
        """Calculate adaptive TTL based on cache performance."""
        if self.default_ttl is None:
            return None
        
        # Adjust TTL based on hit rate
        if self._avg_hit_rate > 0.8:  # High hit rate - increase TTL
            return self.default_ttl * (1 + self._adjustment_factor)
        elif self._avg_hit_rate < 0.3:  # Low hit rate - decrease TTL
            return self.default_ttl * (1 - self._adjustment_factor)
        else:
            return self.default_ttl
    
    def _update_adaptive_params(self) -> None:
        """Update adaptive parameters based on recent performance."""
        total_requests = self._stats['hits'] + self._stats['misses']
        if total_requests > 0:
            current_hit_rate = self._stats['hits'] / total_requests
            # Exponential moving average
            self._avg_hit_rate = 0.9 * self._avg_hit_rate + 0.1 * current_hit_rate
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_if_necessary(self) -> None:
        """Evict entries if limits exceeded."""
        # Size-based eviction
        while len(self._cache) > self.max_size:
            self._evict_lru()
        
        # Memory-based eviction
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        while current_memory > self.max_memory_bytes and self._cache:
            self._evict_lru(memory_pressure=True)
            current_memory = sum(entry.size_bytes for entry in self._cache.values())
    
    def _evict_lru(self, memory_pressure: bool = False) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return
        
        lru_key = self._access_order[0]
        del self._cache[lru_key]
        self._access_order.remove(lru_key)
        
        self._stats['evictions'] += 1
        if memory_pressure:
            self._stats['memory_pressure_evictions'] += 1
        
        logger.debug(f"Evicted LRU entry: {lru_key}")


class ConcurrentProcessor:
    """Concurrent processing utilities for causal discovery."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: Optional[int] = None):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Whether to use processes instead of threads
            chunk_size: Size of data chunks for processing
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.chunk_size = chunk_size or 1000
        self._executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        logger.info(f"Initialized concurrent processor: {self.max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")
    
    def parallel_correlation_matrix(self, data: pd.DataFrame, 
                                  method: str = 'pearson') -> pd.DataFrame:
        """Compute correlation matrix in parallel.
        
        Args:
            data: Input DataFrame
            method: Correlation method
            
        Returns:
            Correlation matrix
        """
        n_vars = len(data.columns)
        if n_vars < 10:  # Not worth parallelizing for small data
            return data.corr(method=method)
        
        # Split computation into chunks
        chunks = self._create_correlation_chunks(data.columns)
        
        with self._executor_class(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._compute_correlation_chunk, data, chunk, method)
                for chunk in chunks
            ]
            
            results = {}
            for future in as_completed(futures):
                chunk_results = future.result()
                results.update(chunk_results)
        
        # Reconstruct full correlation matrix
        return self._reconstruct_correlation_matrix(results, data.columns)
    
    def parallel_apply(self, data: pd.DataFrame, 
                      func: Callable, 
                      axis: int = 0) -> pd.Series:
        """Apply function to DataFrame in parallel.
        
        Args:
            data: Input DataFrame
            func: Function to apply
            axis: Axis to apply along
            
        Returns:
            Series with results
        """
        if len(data) < self.chunk_size:
            return data.apply(func, axis=axis)
        
        # Split data into chunks
        chunks = np.array_split(data, self.max_workers)
        
        with self._executor_class(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(lambda chunk: chunk.apply(func, axis=axis), chunk)
                for chunk in chunks if not chunk.empty
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        return pd.concat(results)
    
    def _create_correlation_chunks(self, columns: pd.Index) -> list:
        """Create chunks for parallel correlation computation."""
        n_vars = len(columns)
        chunk_size = max(5, n_vars // self.max_workers)
        
        chunks = []
        for i in range(0, n_vars, chunk_size):
            chunk_cols = columns[i:i + chunk_size]
            chunks.append(chunk_cols)
        
        return chunks
    
    def _compute_correlation_chunk(self, data: pd.DataFrame, 
                                 chunk_columns: pd.Index, 
                                 method: str) -> Dict[Tuple[str, str], float]:
        """Compute correlations for a chunk of columns."""
        chunk_data = data[chunk_columns]
        chunk_corr = chunk_data.corr(method=method)
        
        results = {}
        for i, col1 in enumerate(chunk_columns):
            for j, col2 in enumerate(data.columns):
                if col2 in chunk_columns:
                    results[(col1, col2)] = chunk_corr.iloc[i, chunk_columns.get_loc(col2)]
                else:
                    # Compute cross-correlation
                    if method == 'pearson':
                        corr_val = data[col1].corr(data[col2], method='pearson')
                    elif method == 'spearman':
                        corr_val = data[col1].corr(data[col2], method='spearman')
                    else:  # kendall
                        corr_val = data[col1].corr(data[col2], method='kendall')
                    results[(col1, col2)] = corr_val
        
        return results
    
    def _reconstruct_correlation_matrix(self, results: Dict[Tuple[str, str], float], 
                                      columns: pd.Index) -> pd.DataFrame:
        """Reconstruct correlation matrix from chunk results."""
        n_vars = len(columns)
        corr_matrix = np.zeros((n_vars, n_vars))
        
        for (col1, col2), value in results.items():
            i, j = columns.get_loc(col1), columns.get_loc(col2)
            corr_matrix[i, j] = value
        
        return pd.DataFrame(corr_matrix, index=columns, columns=columns)


class PerformanceOptimizer:
    """Performance optimization with auto-scaling triggers."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache = AdaptiveCache()
        self.processor = ConcurrentProcessor()
        self._optimization_history = []
        self._auto_scale_thresholds = {
            'cpu_threshold': 80.0,
            'memory_threshold': 75.0,
            'response_time_threshold': 5.0,
            'queue_depth_threshold': 100
        }
        
    def optimize_computation(self, data: pd.DataFrame, 
                           operation: str,
                           force_recompute: bool = False) -> Any:
        """Optimize computation with caching and parallelization.
        
        Args:
            data: Input data
            operation: Operation name for caching
            force_recompute: Whether to force recomputation
            
        Returns:
            Computation result
        """
        # Generate cache key
        cache_key = self._generate_cache_key(data, operation)
        
        # Try cache first
        if not force_recompute:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for {operation}")
                return cached_result
        
        # Determine optimization strategy based on data size
        start_time = time.time()
        
        if operation == 'correlation' and len(data) > 1000:
            result = self.processor.parallel_correlation_matrix(data)
        else:
            # Use standard computation for smaller data
            if operation == 'correlation':
                result = data.corr()
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        computation_time = time.time() - start_time
        
        # Cache result
        self.cache.put(cache_key, result)
        
        # Record optimization metrics
        self._record_optimization(operation, len(data), computation_time)
        
        logger.info(f"Computed {operation} in {computation_time:.3f}s for {len(data)} samples")
        
        return result
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get auto-scaling recommendations based on performance metrics."""
        cache_stats = self.cache.get_stats()
        
        recommendations = {
            'scale_up': False,
            'scale_down': False,
            'reasons': [],
            'cache_efficiency': cache_stats['hit_rate'],
            'memory_pressure': cache_stats['memory_usage_bytes'] / cache_stats['max_memory_bytes']
        }
        
        # Analyze cache performance
        if cache_stats['hit_rate'] < 0.3:
            recommendations['scale_up'] = True
            recommendations['reasons'].append('Low cache hit rate - need more memory')
        
        if cache_stats['memory_pressure_evictions'] > cache_stats['total_evictions'] * 0.5:
            recommendations['scale_up'] = True
            recommendations['reasons'].append('High memory pressure - frequent evictions')
        
        # Analyze computation patterns
        if len(self._optimization_history) > 10:
            recent_times = [entry['computation_time'] for entry in self._optimization_history[-10:]]
            avg_time = sum(recent_times) / len(recent_times)
            
            if avg_time > self._auto_scale_thresholds['response_time_threshold']:
                recommendations['scale_up'] = True
                recommendations['reasons'].append(f'High response times: {avg_time:.2f}s')
        
        return recommendations
    
    def _generate_cache_key(self, data: pd.DataFrame, operation: str) -> str:
        """Generate cache key for data and operation."""
        # Use data shape, columns, and a sample of data for key
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': str(data.dtypes.to_dict()),
            'sample_hash': hashlib.md5(str(data.head().values).encode()).hexdigest()
        }
        
        key_string = f"{operation}_{data_info}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _record_optimization(self, operation: str, data_size: int, 
                           computation_time: float) -> None:
        """Record optimization metrics."""
        self._optimization_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'data_size': data_size,
            'computation_time': computation_time
        })
        
        # Keep only recent history
        if len(self._optimization_history) > 1000:
            self._optimization_history = self._optimization_history[-500:]


def memoize_with_ttl(ttl: float = 3600):
    """Decorator for memoizing function results with TTL.
    
    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func: Callable):
        cache = AdaptiveCache(default_ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Expose cache for inspection
        wrapper.cache = cache
        return wrapper
    
    return decorator


def batch_processing(batch_size: int = 1000):
    """Decorator for batch processing of large datasets.
    
    Args:
        batch_size: Size of each batch
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(data: pd.DataFrame, *args, **kwargs):
            if len(data) <= batch_size:
                return func(data, *args, **kwargs)
            
            # Process in batches
            results = []
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                batch_result = func(batch, *args, **kwargs)
                results.append(batch_result)
            
            # Combine results
            if isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif isinstance(results[0], pd.Series):
                return pd.concat(results)
            else:
                return results
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """Performance profiler for monitoring execution metrics."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self._profiles = {}
        self._active_profiles = {}
        self._start_times = {}
        
    def profile(self, func: Callable = None, name: str = None):
        """Profile a function or return a context manager.
        
        Args:
            func: Function to profile (if used as decorator)
            name: Custom name for the profile
            
        Returns:
            Profiling decorator or context manager
        """
        if func is not None:
            # Used as decorator
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self._record_profile(profile_name, execution_time)
            
            return wrapper
        else:
            # Used as context manager
            return ProfileContext(self, name)
    
    def start_profile(self, name: str) -> None:
        """Start profiling with given name."""
        self._start_times[name] = time.time()
        
    def end_profile(self, name: str) -> float:
        """End profiling and return execution time."""
        if name not in self._start_times:
            logger.warning(f"No active profile found for: {name}")
            return 0.0
        
        execution_time = time.time() - self._start_times[name]
        del self._start_times[name]
        self._record_profile(name, execution_time)
        return execution_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics.
        
        Returns:
            Dictionary with profiling statistics
        """
        stats = {}
        
        for profile_name, times in self._profiles.items():
            if times:
                stats[profile_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'last_time': times[-1] if times else 0
                }
        
        return stats
    
    def clear_stats(self) -> None:
        """Clear all profiling statistics."""
        self._profiles.clear()
        self._active_profiles.clear()
        self._start_times.clear()
        
    def _record_profile(self, name: str, execution_time: float) -> None:
        """Record a profile measurement."""
        if name not in self._profiles:
            self._profiles[name] = []
        
        self._profiles[name].append(execution_time)
        
        # Keep only recent measurements to prevent memory growth
        max_measurements = 1000
        if len(self._profiles[name]) > max_measurements:
            self._profiles[name] = self._profiles[name][-max_measurements//2:]


class ProfileContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str = None):
        """Initialize profile context.
        
        Args:
            profiler: PerformanceProfiler instance
            name: Name for the profile
        """
        self.profiler = profiler
        self.name = name or "anonymous_profile"
        self.start_time = None
    
    def __enter__(self):
        """Enter context and start profiling."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record profiling result."""
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            self.profiler._record_profile(self.name, execution_time)


# Global optimizer instance
global_optimizer = PerformanceOptimizer()