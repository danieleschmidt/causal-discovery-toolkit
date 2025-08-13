"""Advanced performance optimization utilities for causal discovery."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import psutil
import gc
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache, partial
import logging
from pathlib import Path

try:
    from .error_handling import robust_execution, safe_execution
except ImportError:
    try:
        from error_handling import robust_execution, safe_execution
    except ImportError:
        # Fallback implementations
        def robust_execution(max_retries=3):
            def decorator(func):
                return func
            return decorator
        
        def safe_execution(name):
            from contextlib import contextmanager
            @contextmanager
            def context():
                yield None
            return context()

# Simple system monitoring fallback
class SystemMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        import time
        import psutil
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    def stop_monitoring(self):
        import time
        import psutil
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'execution_time': end_time - (self.start_time or end_time),
            'peak_memory_mb': end_memory,
            'memory_used_mb': end_memory - (self.start_memory or end_memory),
            'avg_cpu_percent': psutil.cpu_percent()
        }


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_gpu: bool = False
    max_workers: int = None
    chunk_size: int = 1000
    enable_caching: bool = True
    memory_limit_mb: int = 2000
    use_sparse_matrices: bool = True
    enable_jit_compilation: bool = True
    parallel_threshold: int = 1000
    optimization_level: str = 'balanced'  # 'speed', 'memory', 'balanced'


class PerformanceProfiler:
    """Profile and monitor performance of causal discovery operations."""
    
    def __init__(self):
        self.profiles = {}
        self.logger = logging.getLogger(__name__)
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance."""
        def decorator(func):
            profile_name = func_name or func.__name__
            
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    profile_data = {
                        'execution_time': end_time - start_time,
                        'memory_used_mb': end_memory - start_memory,
                        'peak_memory_mb': end_memory,
                        'success': True
                    }
                    
                    self.profiles[profile_name] = profile_data
                    self.logger.debug(f"Profile {profile_name}: {profile_data}")
                    
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    profile_data = {
                        'execution_time': end_time - start_time,
                        'memory_used_mb': 0,
                        'peak_memory_mb': 0,
                        'success': False,
                        'error': str(e)
                    }
                    self.profiles[profile_name] = profile_data
                    raise
            
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all profiled functions."""
        if not self.profiles:
            return {}
        
        successful_profiles = {k: v for k, v in self.profiles.items() if v['success']}
        
        if not successful_profiles:
            return {'total_functions': len(self.profiles), 'successful_functions': 0}
        
        times = [p['execution_time'] for p in successful_profiles.values()]
        memories = [p['memory_used_mb'] for p in successful_profiles.values()]
        
        return {
            'total_functions': len(self.profiles),
            'successful_functions': len(successful_profiles),
            'total_time': sum(times),
            'avg_time': np.mean(times),
            'max_time': max(times),
            'total_memory_mb': sum(memories),
            'avg_memory_mb': np.mean(memories),
            'max_memory_mb': max(memories),
            'function_details': self.profiles
        }


class OptimizedDataProcessor:
    """Optimized data processing with performance enhancements."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize parallel processing
        self.max_workers = self.config.max_workers or min(8, mp.cpu_count())
        
    @lru_cache(maxsize=128)
    def _cached_correlation(self, data_hash: str, data_shape: Tuple[int, int]) -> np.ndarray:
        """Cached correlation computation for repeated operations."""
        # This is a placeholder - actual implementation would need the data
        # In practice, we'd need a more sophisticated caching mechanism
        return np.random.rand(data_shape[1], data_shape[1])
    
    @robust_execution(max_retries=2)
    def optimize_data_layout(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data layout for better cache performance."""
        
        with self.profiler.profile_function("optimize_data_layout"):
            # Sort columns by data type for better memory layout
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            other_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Reorder columns
            optimized_data = data[numeric_cols + other_cols].copy()
            
            # Convert to more efficient dtypes where possible
            for col in numeric_cols:
                if data[col].dtype == 'int64':
                    max_val = data[col].max()
                    min_val = data[col].min()
                    
                    if min_val >= 0 and max_val < 255:
                        optimized_data[col] = optimized_data[col].astype('uint8')
                    elif min_val >= -128 and max_val < 127:
                        optimized_data[col] = optimized_data[col].astype('int8')
                    elif min_val >= -32768 and max_val < 32767:
                        optimized_data[col] = optimized_data[col].astype('int16')
                    elif min_val >= -2147483648 and max_val < 2147483647:
                        optimized_data[col] = optimized_data[col].astype('int32')
                
                elif data[col].dtype == 'float64':
                    # Check if we can use float32 without significant precision loss
                    if np.allclose(data[col], data[col].astype('float32'), rtol=1e-6):
                        optimized_data[col] = optimized_data[col].astype('float32')
            
            self.logger.info(f"Optimized data layout: {data.memory_usage(deep=True).sum() / 1024**2:.1f}MB -> "
                           f"{optimized_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
            
            return optimized_data
    
    def parallel_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute correlation matrix using parallel processing."""
        
        with self.profiler.profile_function("parallel_correlation"):
            n_vars = data.shape[1]
            
            if n_vars < self.config.parallel_threshold:
                # Use regular computation for small matrices
                return data.corr().values
            
            # Split computation into chunks for parallel processing
            chunk_size = max(1, n_vars // self.max_workers)
            correlation_matrix = np.zeros((n_vars, n_vars))
            
            def compute_correlation_chunk(start_idx: int, end_idx: int) -> Tuple[int, int, np.ndarray]:
                chunk_corr = np.zeros((end_idx - start_idx, n_vars))
                data_values = data.values
                
                for i, col_i in enumerate(range(start_idx, end_idx)):
                    for col_j in range(n_vars):
                        if col_i == col_j:
                            chunk_corr[i, col_j] = 1.0
                        else:
                            chunk_corr[i, col_j] = np.corrcoef(data_values[:, col_i], data_values[:, col_j])[0, 1]
                
                return start_idx, end_idx, chunk_corr
            
            # Execute parallel computation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for start_idx in range(0, n_vars, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_vars)
                    future = executor.submit(compute_correlation_chunk, start_idx, end_idx)
                    futures.append(future)
                
                for future in as_completed(futures):
                    start_idx, end_idx, chunk_corr = future.result()
                    correlation_matrix[start_idx:end_idx, :] = chunk_corr
            
            return correlation_matrix
    
    def memory_efficient_processing(self, data: pd.DataFrame, 
                                  operation: Callable[[pd.DataFrame], Any],
                                  chunk_size: Optional[int] = None) -> List[Any]:
        """Process large datasets in memory-efficient chunks."""
        
        chunk_size = chunk_size or self.config.chunk_size
        n_samples = len(data)
        results = []
        
        with self.profiler.profile_function("memory_efficient_processing"):
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                chunk = data.iloc[start_idx:end_idx]
                
                # Process chunk
                chunk_result = operation(chunk)
                results.append(chunk_result)
                
                # Force garbage collection to free memory
                if start_idx % (chunk_size * 10) == 0:
                    gc.collect()
                
                self.logger.debug(f"Processed chunk {start_idx//chunk_size + 1}/{(n_samples-1)//chunk_size + 1}")
        
        return results
    
    def adaptive_algorithm_selection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Adaptively select algorithm parameters based on data characteristics."""
        
        n_samples, n_features = data.shape
        memory_usage_mb = data.memory_usage(deep=True).sum() / 1024**2
        
        # Compute data characteristics
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_usage_mb': memory_usage_mb,
            'sparsity': (data == 0).sum().sum() / (n_samples * n_features),
            'has_missing': data.isnull().any().any(),
            'numeric_ratio': len(data.select_dtypes(include=[np.number]).columns) / n_features
        }
        
        # Adaptive recommendations
        recommendations = {
            'use_parallel': n_features > self.config.parallel_threshold,
            'chunk_processing': memory_usage_mb > self.config.memory_limit_mb,
            'use_sparse': characteristics['sparsity'] > 0.5,
            'algorithm_timeout': min(300, max(30, n_features * 2)),
            'recommended_chunk_size': min(self.config.chunk_size, max(100, n_samples // 10))
        }
        
        # Algorithm-specific recommendations
        if characteristics['sparsity'] > 0.8:
            recommendations['preferred_algorithms'] = ['constraint_based', 'information_theory']
        elif n_features > 50:
            recommendations['preferred_algorithms'] = ['simple_linear', 'optimized']
        else:
            recommendations['preferred_algorithms'] = ['bayesian_network', 'mutual_information']
        
        self.logger.info(f"Adaptive recommendations: {recommendations}")
        
        return {
            'characteristics': characteristics,
            'recommendations': recommendations
        }


class GPUAccelerator:
    """GPU acceleration utilities for causal discovery (placeholder for future GPU support)."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.logger = logging.getLogger(__name__)
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Placeholder for GPU detection
            # In a real implementation, this would check for CUDA, OpenCL, etc.
            return False
        except Exception:
            return False
    
    def gpu_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix using GPU acceleration."""
        if not self.gpu_available:
            self.logger.warning("GPU not available, falling back to CPU computation")
            return np.corrcoef(data.T)
        
        # Placeholder for GPU-accelerated correlation computation
        # In a real implementation, this would use CuPy, JAX, or similar
        self.logger.info("Using GPU acceleration for correlation matrix")
        return np.corrcoef(data.T)


class StreamingProcessor:
    """Process streaming data for real-time causal discovery."""
    
    def __init__(self, window_size: int = 1000, overlap: float = 0.1):
        self.window_size = window_size
        self.overlap = overlap
        self.overlap_size = int(window_size * overlap)
        self.buffer = []
        self.results_history = []
        self.logger = logging.getLogger(__name__)
    
    def add_data_batch(self, batch: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Add a batch of data and process if window is complete."""
        self.buffer.append(batch)
        
        # Check if we have enough data for processing
        total_samples = sum(len(b) for b in self.buffer)
        
        if total_samples >= self.window_size:
            return self._process_window()
        
        return None
    
    def _process_window(self) -> Dict[str, Any]:
        """Process the current window of data."""
        # Combine all batches in buffer
        window_data = pd.concat(self.buffer, ignore_index=True)
        
        # Take only the window_size most recent samples
        if len(window_data) > self.window_size:
            window_data = window_data.tail(self.window_size)
        
        # Process the window (placeholder for actual causal discovery)
        processing_start = time.time()
        
        # Simple correlation-based processing for demonstration
        corr_matrix = window_data.corr().abs()
        
        result = {
            'timestamp': time.time(),
            'processing_time': time.time() - processing_start,
            'window_size': len(window_data),
            'n_features': len(window_data.columns),
            'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean(),
            'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max()
        }
        
        self.results_history.append(result)
        
        # Keep overlap for next window
        overlap_data = window_data.tail(self.overlap_size) if self.overlap_size > 0 else pd.DataFrame()
        self.buffer = [overlap_data] if not overlap_data.empty else []
        
        self.logger.info(f"Processed streaming window: {result}")
        return result
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends in streaming results."""
        if len(self.results_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_results = self.results_history[-10:]  # Last 10 windows
        
        correlations = [r['avg_correlation'] for r in recent_results]
        processing_times = [r['processing_time'] for r in recent_results]
        
        return {
            'n_windows': len(recent_results),
            'correlation_trend': 'increasing' if correlations[-1] > correlations[0] else 'decreasing',
            'avg_processing_time': np.mean(processing_times),
            'processing_time_trend': 'improving' if processing_times[-1] < processing_times[0] else 'degrading',
            'stability_score': 1.0 - np.std(correlations) / max(np.mean(correlations), 1e-6)
        }


class PerformanceOptimizedPipeline:
    """High-performance causal discovery pipeline with optimizations."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.data_processor = OptimizedDataProcessor(config)
        self.gpu_accelerator = GPUAccelerator() if config.enable_gpu else None
        self.profiler = PerformanceProfiler()
        self.monitor = SystemMonitor()
        self.logger = logging.getLogger(__name__)
    
    @robust_execution(max_retries=2)
    def optimize_and_discover(self, data: pd.DataFrame, algorithm) -> Dict[str, Any]:
        """Run optimized causal discovery with performance monitoring."""
        
        start_time = time.time()
        self.monitor.start_monitoring()
        
        try:
            # Step 1: Optimize data layout
            with self.profiler.profile_function("data_optimization"):
                optimized_data = self.data_processor.optimize_data_layout(data)
            
            # Step 2: Adaptive algorithm configuration
            with self.profiler.profile_function("adaptive_config"):
                adaptation = self.data_processor.adaptive_algorithm_selection(optimized_data)
            
            # Step 3: Configure algorithm based on recommendations
            self._apply_adaptive_config(algorithm, adaptation['recommendations'])
            
            # Step 4: Execute causal discovery
            with self.profiler.profile_function("causal_discovery"):
                if hasattr(algorithm, 'fit_discover'):
                    result = algorithm.fit_discover(optimized_data)
                else:
                    algorithm.fit(optimized_data)
                    result = algorithm.discover()
            
            # Step 5: Collect performance metrics
            execution_time = time.time() - start_time
            system_metrics = self.monitor.stop_monitoring()
            performance_summary = self.profiler.get_performance_summary()
            
            return {
                'causal_result': result,
                'execution_time': execution_time,
                'system_metrics': system_metrics,
                'performance_summary': performance_summary,
                'data_characteristics': adaptation['characteristics'],
                'optimizations_applied': adaptation['recommendations']
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            self.logger.error(f"Optimized causal discovery failed: {e}")
            raise
    
    def _apply_adaptive_config(self, algorithm, recommendations: Dict[str, Any]):
        """Apply adaptive configuration to algorithm."""
        
        # Set timeout if algorithm supports it
        if hasattr(algorithm, 'timeout') and 'algorithm_timeout' in recommendations:
            algorithm.timeout = recommendations['algorithm_timeout']
        
        # Enable parallel processing if supported
        if hasattr(algorithm, 'parallel') and recommendations.get('use_parallel', False):
            algorithm.parallel = True
            if hasattr(algorithm, 'n_jobs'):
                algorithm.n_jobs = self.config.max_workers
        
        # Configure memory-efficient processing
        if hasattr(algorithm, 'chunk_size') and recommendations.get('chunk_processing', False):
            algorithm.chunk_size = recommendations.get('recommended_chunk_size', self.config.chunk_size)
        
        # Enable sparse matrix support if available
        if hasattr(algorithm, 'use_sparse') and recommendations.get('use_sparse', False):
            algorithm.use_sparse = True
        
        self.logger.info(f"Applied adaptive configuration to {algorithm.__class__.__name__}")
    
    def benchmark_algorithms(self, algorithms: Dict[str, Any], 
                           data: pd.DataFrame) -> pd.DataFrame:
        """Benchmark multiple algorithms with performance optimization."""
        
        results = []
        
        for name, algorithm in algorithms.items():
            self.logger.info(f"Benchmarking {name}...")
            
            try:
                result = self.optimize_and_discover(data, algorithm)
                
                results.append({
                    'algorithm': name,
                    'execution_time': result['execution_time'],
                    'peak_memory_mb': result['system_metrics'].get('peak_memory_mb', 0),
                    'cpu_usage_avg': result['system_metrics'].get('avg_cpu_percent', 0),
                    'success': True,
                    'n_edges': np.sum(result['causal_result'].adjacency_matrix),
                    'optimizations_applied': len(result['optimizations_applied'])
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {name}: {e}")
                results.append({
                    'algorithm': name,
                    'execution_time': float('inf'),
                    'peak_memory_mb': 0,
                    'cpu_usage_avg': 0,
                    'success': False,
                    'n_edges': 0,
                    'optimizations_applied': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


def create_performance_optimized_config(optimization_level: str = 'balanced') -> PerformanceConfig:
    """Create performance configuration based on optimization level."""
    
    if optimization_level == 'speed':
        return PerformanceConfig(
            enable_gpu=True,
            max_workers=mp.cpu_count(),
            chunk_size=500,
            enable_caching=True,
            memory_limit_mb=4000,
            use_sparse_matrices=True,
            enable_jit_compilation=True,
            parallel_threshold=500,
            optimization_level='speed'
        )
    elif optimization_level == 'memory':
        return PerformanceConfig(
            enable_gpu=False,
            max_workers=2,
            chunk_size=100,
            enable_caching=False,
            memory_limit_mb=500,
            use_sparse_matrices=True,
            enable_jit_compilation=False,
            parallel_threshold=2000,
            optimization_level='memory'
        )
    else:  # balanced
        return PerformanceConfig(
            enable_gpu=False,
            max_workers=min(4, mp.cpu_count()),
            chunk_size=1000,
            enable_caching=True,
            memory_limit_mb=2000,
            use_sparse_matrices=True,
            enable_jit_compilation=True,
            parallel_threshold=1000,
            optimization_level='balanced'
        )