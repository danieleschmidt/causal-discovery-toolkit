"""High-performance optimized causal discovery algorithms."""

from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .robust import RobustSimpleLinearCausalModel
from ..utils.performance import global_optimizer, memoize_with_ttl, batch_processing
from ..utils.monitoring import monitor_performance

try:
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class OptimizedCausalModel(RobustSimpleLinearCausalModel):
    """High-performance optimized causal discovery model."""
    
    def __init__(self,
                 threshold: float = 0.3,
                 enable_caching: bool = True,
                 enable_parallel: bool = True,
                 cache_ttl: float = 3600,
                 max_workers: Optional[int] = None,
                 auto_optimize: bool = True,
                 **kwargs):
        """Initialize optimized causal discovery model.
        
        Args:
            threshold: Correlation threshold for edge detection
            enable_caching: Whether to enable result caching
            enable_parallel: Whether to enable parallel processing
            cache_ttl: Cache time-to-live in seconds
            max_workers: Maximum number of parallel workers
            auto_optimize: Whether to enable auto-optimization
            **kwargs: Additional parameters
        """
        super().__init__(threshold=threshold, **kwargs)
        
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 2)
        self.auto_optimize = auto_optimize
        
        # Performance tracking
        self._performance_stats = {
            'fit_times': [],
            'discover_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0
        }
        
        self._lock = threading.Lock()
        
        logger.info(f"Initialized OptimizedCausalModel with caching={enable_caching}, "
                   f"parallel={enable_parallel}, workers={self.max_workers}")
    
    @monitor_performance("optimized_fit")
    def fit(self, data: pd.DataFrame) -> 'OptimizedCausalModel':
        """Fit model with performance optimizations."""
        start_time = time.time()
        
        # Use parent's robust fitting
        result = super().fit(data)
        
        # Record performance
        fit_time = time.time() - start_time
        with self._lock:
            self._performance_stats['fit_times'].append(fit_time)
        
        # Auto-optimize based on data characteristics
        if self.auto_optimize:
            self._auto_optimize_parameters(data)
        
        logger.info(f"Optimized fit completed in {fit_time:.3f}s")
        return result
    
    @monitor_performance("optimized_discover")
    def discover(self, data: Optional[pd.DataFrame] = None) -> Any:
        """Discover causal relationships with performance optimizations."""
        start_time = time.time()
        
        # Determine data to use
        discovery_data = data if data is not None else self._data
        
        if discovery_data is None:
            raise RuntimeError("No data available for discovery")
        
        # Use optimized discovery if enabled, otherwise fall back to robust
        if self.enable_caching or self.enable_parallel:
            result = self._optimized_discover(discovery_data)
        else:
            result = super().discover(data)
        
        # Record performance
        discover_time = time.time() - start_time
        with self._lock:
            self._performance_stats['discover_times'].append(discover_time)
        
        logger.info(f"Optimized discovery completed in {discover_time:.3f}s")
        return result
    
    def _optimized_discover(self, data: pd.DataFrame) -> Any:
        """Internal optimized discovery method."""
        # Use global optimizer for caching and parallelization
        if self.enable_caching:
            correlation_matrix = global_optimizer.optimize_computation(
                data, 'correlation'
            )
            self._performance_stats['cache_hits'] += 1
        else:
            correlation_matrix = self._compute_correlation_optimized(data)
            self._performance_stats['cache_misses'] += 1
        
        # Create adjacency matrix efficiently
        adjacency_matrix = self._create_adjacency_matrix_optimized(correlation_matrix)
        
        # Prepare confidence scores
        confidence_scores = correlation_matrix.abs().values
        np.fill_diagonal(confidence_scores, 0)
        
        # Build metadata efficiently
        metadata = self._build_metadata_optimized(data, adjacency_matrix, confidence_scores)
        
        return type(super().discover())(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_used="OptimizedCausal",
            metadata=metadata
        )
    
    def _compute_correlation_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix with optimizations."""
        n_samples, n_features = data.shape
        
        # Use parallel processing for large datasets
        if self.enable_parallel and n_features > 10 and n_samples > 1000:
            self._performance_stats['parallel_operations'] += 1
            return self._parallel_correlation(data)
        else:
            return data.corr(method=self.correlation_method)
    
    def _parallel_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix in parallel chunks."""
        n_features = len(data.columns)
        chunk_size = max(2, n_features // self.max_workers)
        
        # Create column chunks
        chunks = [
            data.columns[i:i + chunk_size] 
            for i in range(0, n_features, chunk_size)
        ]
        
        correlation_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit correlation tasks for each chunk
            future_to_chunk = {
                executor.submit(self._compute_chunk_correlations, data, chunk): chunk
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                chunk_results = future.result()
                correlation_results.update(chunk_results)
        
        # Reconstruct full correlation matrix
        return self._reconstruct_correlation_matrix(correlation_results, data.columns)
    
    def _compute_chunk_correlations(self, data: pd.DataFrame, 
                                  chunk_columns: pd.Index) -> Dict[tuple, float]:
        """Compute correlations for a chunk of columns."""
        results = {}
        
        for col1 in chunk_columns:
            for col2 in data.columns:
                if self.correlation_method == 'pearson':
                    corr = data[col1].corr(data[col2], method='pearson')
                elif self.correlation_method == 'spearman':
                    corr = data[col1].corr(data[col2], method='spearman')
                else:  # kendall
                    corr = data[col1].corr(data[col2], method='kendall')
                
                results[(col1, col2)] = corr
        
        return results
    
    def _reconstruct_correlation_matrix(self, results: Dict[tuple, float], 
                                      columns: pd.Index) -> pd.DataFrame:
        """Reconstruct correlation matrix from parallel results."""
        n_features = len(columns)
        matrix = np.zeros((n_features, n_features))
        
        for (col1, col2), corr_value in results.items():
            i = columns.get_loc(col1)
            j = columns.get_loc(col2)
            matrix[i, j] = corr_value
        
        return pd.DataFrame(matrix, index=columns, columns=columns)
    
    def _create_adjacency_matrix_optimized(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """Create adjacency matrix with optimizations."""
        # Vectorized operations for better performance
        abs_corr = np.abs(correlation_matrix.values)
        adjacency = (abs_corr > self.threshold).astype(np.int8)
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def _build_metadata_optimized(self, data: pd.DataFrame, 
                                 adjacency_matrix: np.ndarray,
                                 confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Build metadata efficiently."""
        n_vars = len(data.columns)
        n_edges = np.sum(adjacency_matrix)
        
        # Use numpy operations for efficiency
        max_confidence = np.max(confidence_scores)
        nonzero_confidences = confidence_scores[confidence_scores > 0]
        mean_confidence = np.mean(nonzero_confidences) if len(nonzero_confidences) > 0 else 0
        
        metadata = {
            "threshold": self.threshold,
            "correlation_method": self.correlation_method,
            "n_variables": n_vars,
            "n_edges": int(n_edges),
            "variable_names": list(data.columns),
            "sparsity": float(1 - (n_edges / (n_vars * (n_vars - 1)))),
            "max_confidence": float(max_confidence),
            "mean_confidence": float(mean_confidence),
            "optimization_enabled": True,
            "caching_enabled": self.enable_caching,
            "parallel_enabled": self.enable_parallel,
            **self.fit_metadata
        }
        
        return metadata
    
    def _auto_optimize_parameters(self, data: pd.DataFrame) -> None:
        """Auto-optimize parameters based on data characteristics."""
        n_samples, n_features = data.shape
        
        # Adjust parallel processing threshold
        if n_samples > 10000 and n_features > 20:
            if not self.enable_parallel:
                self.enable_parallel = True
                logger.info("Auto-enabled parallel processing for large dataset")
        
        # Adjust caching based on data size
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        if data_size_mb > 100 and not self.enable_caching:
            self.enable_caching = True
            logger.info(f"Auto-enabled caching for large dataset ({data_size_mb:.1f}MB)")
        
        # Suggest scaling recommendations
        scaling_rec = global_optimizer.get_scaling_recommendations()
        if scaling_rec['scale_up']:
            logger.warning(f"Scaling recommendation: {scaling_rec['reasons']}")
    
    @memoize_with_ttl(ttl=3600)
    def cached_correlation_subset(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Compute correlation for subset of columns with caching."""
        subset_data = data[columns]
        return subset_data.corr(method=self.correlation_method)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = self._performance_stats.copy()
        
        # Add computed statistics
        if stats['fit_times']:
            stats['avg_fit_time'] = np.mean(stats['fit_times'])
            stats['total_fit_time'] = np.sum(stats['fit_times'])
        
        if stats['discover_times']:
            stats['avg_discover_time'] = np.mean(stats['discover_times'])
            stats['total_discover_time'] = np.sum(stats['discover_times'])
        
        # Add cache statistics
        cache_stats = global_optimizer.cache.get_stats()
        stats['cache_hit_rate'] = cache_stats['hit_rate']
        stats['cache_size'] = cache_stats['size']
        stats['cache_memory_usage_mb'] = cache_stats['memory_usage_bytes'] / 1024 / 1024
        
        return stats
    
    def benchmark_performance(self, data_sizes: list = None, 
                            n_runs: int = 3) -> pd.DataFrame:
        """Benchmark performance across different data sizes.
        
        Args:
            data_sizes: List of (n_samples, n_features) tuples
            n_runs: Number of runs per size
            
        Returns:
            DataFrame with benchmark results
        """
        if data_sizes is None:
            data_sizes = [(100, 5), (500, 10), (1000, 15), (2000, 20)]
        
        results = []
        
        for n_samples, n_features in data_sizes:
            for run in range(n_runs):
                # Generate test data
                from ..utils.data_processing import DataProcessor
                data_processor = DataProcessor()
                test_data = data_processor.generate_synthetic_data(
                    n_samples=n_samples,
                    n_variables=n_features,
                    random_state=42 + run
                )
                
                # Benchmark fit
                start_time = time.time()
                self.fit(test_data)
                fit_time = time.time() - start_time
                
                # Benchmark discovery
                start_time = time.time()
                result = self.discover()
                discover_time = time.time() - start_time
                
                results.append({
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'run': run,
                    'fit_time': fit_time,
                    'discover_time': discover_time,
                    'total_time': fit_time + discover_time,
                    'n_edges': result.metadata['n_edges'],
                    'memory_mb': test_data.memory_usage(deep=True).sum() / 1024 / 1024
                })
        
        return pd.DataFrame(results)
    
    def enable_turbo_mode(self) -> None:
        """Enable all performance optimizations."""
        self.enable_caching = True
        self.enable_parallel = True
        self.auto_optimize = True
        
        # Increase worker count for turbo mode
        self.max_workers = min(16, (os.cpu_count() or 1) * 2)
        
        logger.info("Turbo mode enabled: all optimizations active")
    
    def disable_optimizations(self) -> None:
        """Disable all optimizations for debugging."""
        self.enable_caching = False
        self.enable_parallel = False
        self.auto_optimize = False
        
        logger.info("All optimizations disabled")


@batch_processing(batch_size=2000)
def batch_causal_discovery(data: pd.DataFrame, 
                          model_params: Dict[str, Any] = None) -> pd.DataFrame:
    """Process large datasets in batches for causal discovery.
    
    Args:
        data: Large input dataset
        model_params: Parameters for causal model
        
    Returns:
        Combined results from batch processing
    """
    params = model_params or {}
    model = OptimizedCausalModel(**params)
    
    model.fit(data)
    result = model.discover()
    
    # Return adjacency matrix as DataFrame for easier combination
    return pd.DataFrame(
        result.adjacency_matrix,
        index=data.columns,
        columns=data.columns
    )


class AdaptiveScalingManager:
    """Manages adaptive scaling based on performance metrics."""
    
    def __init__(self):
        """Initialize adaptive scaling manager."""
        self.models = {}
        self.performance_thresholds = {
            'response_time': 2.0,  # seconds
            'memory_usage': 0.8,   # 80% of available
            'cpu_usage': 0.75,     # 75% of available
            'queue_depth': 50      # number of pending requests
        }
        self.scaling_factor = 1.5
        self._metrics_history = []
    
    def register_model(self, model_id: str, model: OptimizedCausalModel) -> None:
        """Register a model for adaptive scaling."""
        self.models[model_id] = {
            'model': model,
            'current_workers': model.max_workers,
            'scaling_events': []
        }
        logger.info(f"Registered model {model_id} for adaptive scaling")
    
    def check_and_scale(self, model_id: str, 
                       current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check metrics and scale if necessary.
        
        Args:
            model_id: Model identifier
            current_metrics: Current performance metrics
            
        Returns:
            Scaling decision and actions taken
        """
        if model_id not in self.models:
            return {'action': 'error', 'message': f'Model {model_id} not registered'}
        
        model_info = self.models[model_id]
        model = model_info['model']
        
        scaling_decision = self._make_scaling_decision(current_metrics)
        
        if scaling_decision['action'] == 'scale_up':
            new_workers = min(32, int(model.max_workers * self.scaling_factor))
            model.max_workers = new_workers
            model_info['current_workers'] = new_workers
            
            scaling_event = {
                'timestamp': time.time(),
                'action': 'scale_up',
                'old_workers': model_info['current_workers'],
                'new_workers': new_workers,
                'reason': scaling_decision['reason']
            }
            model_info['scaling_events'].append(scaling_event)
            
            logger.info(f"Scaled up model {model_id}: {model_info['current_workers']} → {new_workers} workers")
            
        elif scaling_decision['action'] == 'scale_down':
            new_workers = max(1, int(model.max_workers / self.scaling_factor))
            model.max_workers = new_workers
            model_info['current_workers'] = new_workers
            
            scaling_event = {
                'timestamp': time.time(),
                'action': 'scale_down',
                'old_workers': model_info['current_workers'],
                'new_workers': new_workers,
                'reason': scaling_decision['reason']
            }
            model_info['scaling_events'].append(scaling_event)
            
            logger.info(f"Scaled down model {model_id}: {model_info['current_workers']} → {new_workers} workers")
        
        return scaling_decision
    
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make scaling decision based on metrics."""
        reasons = []
        
        # Check if scaling up is needed
        if metrics.get('response_time', 0) > self.performance_thresholds['response_time']:
            reasons.append(f"High response time: {metrics['response_time']:.2f}s")
        
        if metrics.get('memory_usage', 0) > self.performance_thresholds['memory_usage']:
            reasons.append(f"High memory usage: {metrics['memory_usage']:.1%}")
        
        if metrics.get('cpu_usage', 0) > self.performance_thresholds['cpu_usage']:
            reasons.append(f"High CPU usage: {metrics['cpu_usage']:.1%}")
        
        if metrics.get('queue_depth', 0) > self.performance_thresholds['queue_depth']:
            reasons.append(f"High queue depth: {metrics['queue_depth']}")
        
        if reasons:
            return {
                'action': 'scale_up',
                'reason': '; '.join(reasons),
                'metrics': metrics
            }
        
        # Check if scaling down is possible
        all_metrics_low = all([
            metrics.get('response_time', 999) < self.performance_thresholds['response_time'] * 0.5,
            metrics.get('memory_usage', 999) < self.performance_thresholds['memory_usage'] * 0.5,
            metrics.get('cpu_usage', 999) < self.performance_thresholds['cpu_usage'] * 0.5,
            metrics.get('queue_depth', 999) < self.performance_thresholds['queue_depth'] * 0.3
        ])
        
        if all_metrics_low:
            return {
                'action': 'scale_down',
                'reason': 'All metrics well below thresholds',
                'metrics': metrics
            }
        
        return {
            'action': 'no_change',
            'reason': 'Metrics within acceptable range',
            'metrics': metrics
        }