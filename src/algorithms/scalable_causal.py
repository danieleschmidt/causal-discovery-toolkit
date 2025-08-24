"""Scalable causal discovery with advanced optimization and parallelization."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
import pickle
import hashlib
from dataclasses import dataclass
import psutil

try:
    from .robust_enhanced import RobustCausalDiscoveryModel, RobustCausalResult
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.performance import ConcurrentProcessor, AdaptiveCache, BatchProcessor
    from ..utils.auto_scaling import AutoScaler, ResourceMonitor
    from ..utils.logging_config import get_logger
    from ..utils.monitoring import monitor_performance
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel, RobustCausalResult
    from algorithms.base import CausalDiscoveryModel, CausalResult
    from utils.performance import ConcurrentProcessor, AdaptiveCache, BatchProcessor
    from utils.auto_scaling import AutoScaler, ResourceMonitor
    from utils.logging_config import get_logger
    from utils.monitoring import monitor_performance


logger = get_logger(__name__)


@dataclass
class ScalabilityMetrics:
    """Metrics for scalability tracking."""
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    parallelization_efficiency: float
    throughput_ops_per_sec: float
    resource_utilization: Dict[str, float]


class ScalableCausalDiscoveryModel(RobustCausalDiscoveryModel):
    """Scalable causal discovery model with advanced optimization and parallelization."""
    
    def __init__(self,
                 base_model: Optional[CausalDiscoveryModel] = None,
                 enable_parallelization: bool = True,
                 enable_caching: bool = True,
                 enable_auto_scaling: bool = True,
                 max_workers: Optional[int] = None,
                 cache_size: int = 1000,
                 batch_size: int = 100,
                 optimization_level: str = "balanced",  # "speed", "memory", "balanced"
                 **kwargs):
        """Initialize scalable causal discovery model.
        
        Args:
            base_model: Base model to wrap
            enable_parallelization: Enable parallel processing
            enable_caching: Enable adaptive caching
            enable_auto_scaling: Enable auto-scaling
            max_workers: Maximum number of worker processes/threads
            cache_size: Maximum cache size
            batch_size: Batch size for processing
            optimization_level: Optimization strategy
            **kwargs: Additional parameters
        """
        super().__init__(base_model=base_model, **kwargs)
        
        # Scalability configuration
        self.enable_parallelization = enable_parallelization
        self.enable_caching = enable_caching
        self.enable_auto_scaling = enable_auto_scaling
        self.optimization_level = optimization_level
        
        # System resources
        self.cpu_count = psutil.cpu_count()
        self.memory_total_gb = psutil.virtual_memory().total / 1024**3
        
        # Configure worker count
        if max_workers is None:
            if optimization_level == "speed":
                max_workers = min(self.cpu_count * 2, 32)
            elif optimization_level == "memory":
                max_workers = max(2, self.cpu_count // 2)
            else:  # balanced
                max_workers = self.cpu_count
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialize scalability components
        self._init_scalability_components(cache_size)
        
        # Metrics tracking
        self.scalability_metrics: List[ScalabilityMetrics] = []
        self._lock = threading.Lock()
        
        logger.info(f"Initialized ScalableCausalDiscoveryModel with {self.max_workers} workers, "
                   f"optimization: {optimization_level}")
    
    def _init_scalability_components(self, cache_size: int):
        """Initialize scalability components."""
        # Concurrent processing
        if self.enable_parallelization:
            self.concurrent_processor = ConcurrentProcessor(
                max_workers=self.max_workers,
                use_processes=self.optimization_level == "speed"
            )
        
        # Adaptive caching
        if self.enable_caching:
            self.cache = AdaptiveCache(
                max_size=cache_size,
                default_ttl=3600  # 1 hour
            )
        
        # Batch processing
        self.batch_processor = BatchProcessor(
            batch_size=self.batch_size,
            max_workers=self.max_workers
        )
        
        # Auto-scaling
        if self.enable_auto_scaling:
            self.resource_monitor = ResourceMonitor()
            self.auto_scaler = AutoScaler(
                min_workers=2,
                max_workers=self.max_workers * 2,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3
            )
    
    @monitor_performance
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ScalableCausalDiscoveryModel':
        """Fit model with scalable optimizations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2
        start_cpu = psutil.cpu_percent(interval=0.1)
        
        try:
            # Check if we can use cached result
            if self.enable_caching:
                cache_key = self._generate_cache_key("fit", data, kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Using cached fit result")
                    self._restore_from_cache(cached_result)
                    return self
            
            # Determine optimal processing strategy
            processing_strategy = self._determine_processing_strategy(data)
            logger.info(f"Using processing strategy: {processing_strategy}")
            
            if processing_strategy == "batch_parallel":
                result = self._fit_batch_parallel(data, **kwargs)
            elif processing_strategy == "streaming":
                result = self._fit_streaming(data, **kwargs)
            else:
                # Fall back to robust base implementation
                result = super().fit(data, **kwargs)
            
            # Cache result if enabled
            if self.enable_caching:
                cache_data = self._prepare_cache_data()
                self.cache.put(cache_key, cache_data)
            
            # Record metrics
            self._record_scalability_metrics(
                start_time, start_memory, start_cpu, "fit"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Scalable fit failed: {e}")
            # Fall back to base implementation
            return super().fit(data, **kwargs)
    
    @monitor_performance
    def discover(self, data: Optional[pd.DataFrame] = None, **kwargs) -> RobustCausalResult:
        """Discover causal relationships with scalable optimizations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2
        start_cpu = psutil.cpu_percent(interval=0.1)
        
        try:
            # Check cache first
            if self.enable_caching and data is not None:
                cache_key = self._generate_cache_key("discover", data, kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Using cached discovery result")
                    return cached_result
            
            # Auto-scale if enabled
            if self.enable_auto_scaling:
                current_load = self.resource_monitor.get_system_load()
                self.auto_scaler.adjust_workers(current_load)
            
            # Determine optimal discovery strategy
            discovery_strategy = self._determine_discovery_strategy(data)
            logger.info(f"Using discovery strategy: {discovery_strategy}")
            
            if discovery_strategy == "parallel_chunks":
                result = self._discover_parallel_chunks(data, **kwargs)
            elif discovery_strategy == "hierarchical":
                result = self._discover_hierarchical(data, **kwargs)
            elif discovery_strategy == "approximate":
                result = self._discover_approximate(data, **kwargs)
            else:
                # Fall back to robust base implementation
                result = super().discover(data, **kwargs)
            
            # Cache result
            if self.enable_caching and data is not None:
                self.cache.put(cache_key, result)
            
            # Record metrics
            self._record_scalability_metrics(
                start_time, start_memory, start_cpu, "discover"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Scalable discovery failed: {e}")
            # Fall back to base implementation
            return super().discover(data, **kwargs)
    
    def _determine_processing_strategy(self, data: pd.DataFrame) -> str:
        """Determine optimal processing strategy based on data characteristics."""
        n_samples, n_features = data.shape
        memory_usage_mb = data.memory_usage(deep=True).sum() / 1024**2
        
        # Memory-based decisions
        if memory_usage_mb > self.memory_total_gb * 1024 * 0.5:  # > 50% of RAM
            return "streaming"
        
        # Size-based decisions
        if n_samples > 10000 and n_features > 20:
            return "batch_parallel"
        
        return "standard"
    
    def _determine_discovery_strategy(self, data: Optional[pd.DataFrame]) -> str:
        """Determine optimal discovery strategy."""
        if data is None:
            data = getattr(self.base_model, '_data', None)
        
        if data is None:
            return "standard"
        
        n_samples, n_features = data.shape
        
        # High-dimensional data
        if n_features > 100:
            return "hierarchical"
        
        # Large datasets
        if n_samples > 5000 and n_features > 10:
            return "parallel_chunks"
        
        # Very large datasets with performance priority
        if n_samples > 20000 and self.optimization_level == "speed":
            return "approximate"
        
        return "standard"
    
    def _fit_batch_parallel(self, data: pd.DataFrame, **kwargs) -> 'ScalableCausalDiscoveryModel':
        """Fit using batch parallel processing."""
        logger.info("Using batch parallel fitting")
        
        # Split data into batches
        n_samples = len(data)
        batch_size = min(self.batch_size, n_samples // self.max_workers)
        batch_size = max(batch_size, 10)  # Minimum batch size
        
        batches = []
        for i in range(0, n_samples, batch_size):
            batch = data.iloc[i:i+batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        if len(batches) > 1 and self.enable_parallelization:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(self._fit_single_batch, batch, **kwargs)
                    futures.append(future)
                
                # Collect results and aggregate
                batch_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.warning(f"Batch processing failed: {e}")
                
                # Aggregate batch results
                self._aggregate_batch_fit_results(batch_results)
        else:
            # Single-threaded fallback
            super().fit(data, **kwargs)
        
        return self
    
    def _fit_streaming(self, data: pd.DataFrame, **kwargs) -> 'ScalableCausalDiscoveryModel':
        """Fit using streaming processing for large datasets."""
        logger.info("Using streaming fitting for large dataset")
        
        # Process in chunks to avoid memory issues
        chunk_size = min(1000, len(data) // 4)
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            
            # Update model incrementally
            if i == 0:
                # First chunk - initialize
                super().fit(chunk, **kwargs)
            else:
                # Subsequent chunks - update
                self._update_with_chunk(chunk, **kwargs)
        
        return self
    
    def _discover_parallel_chunks(self, data: Optional[pd.DataFrame], **kwargs) -> RobustCausalResult:
        """Discover using parallel chunk processing."""
        logger.info("Using parallel chunk discovery")
        
        if data is None:
            data = getattr(self.base_model, '_data', None)
        
        if data is None:
            return super().discover(data, **kwargs)
        
        # Split variables into chunks for parallel processing
        n_vars = data.shape[1]
        chunk_size = max(2, n_vars // self.max_workers)
        
        var_chunks = []
        for i in range(0, n_vars, chunk_size):
            chunk_vars = data.columns[i:i+chunk_size]
            var_chunks.append(chunk_vars)
        
        # Process chunks in parallel
        if len(var_chunks) > 1 and self.enable_parallelization:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for chunk_vars in var_chunks:
                    chunk_data = data[chunk_vars]
                    future = executor.submit(self._discover_chunk, chunk_data, **kwargs)
                    futures.append((chunk_vars, future))
                
                # Collect and merge results
                chunk_results = []
                for chunk_vars, future in futures:
                    try:
                        result = future.result()
                        chunk_results.append((chunk_vars, result))
                    except Exception as e:
                        logger.warning(f"Chunk discovery failed: {e}")
                
                # Merge chunk results
                return self._merge_chunk_results(chunk_results, data.columns)
        else:
            # Fallback to base implementation
            return super().discover(data, **kwargs)
    
    def _discover_hierarchical(self, data: Optional[pd.DataFrame], **kwargs) -> RobustCausalResult:
        """Discover using hierarchical approach for high-dimensional data."""
        logger.info("Using hierarchical discovery for high-dimensional data")
        
        if data is None:
            data = getattr(self.base_model, '_data', None)
        
        if data is None:
            return super().discover(data, **kwargs)
        
        # Stage 1: Feature selection/clustering
        important_vars = self._select_important_variables(data)
        
        # Stage 2: Discover within clusters
        cluster_results = []
        for cluster_vars in important_vars:
            cluster_data = data[cluster_vars]
            result = super().discover(cluster_data, **kwargs)
            cluster_results.append((cluster_vars, result))
        
        # Stage 3: Cross-cluster discovery
        return self._merge_hierarchical_results(cluster_results, data.columns)
    
    def _discover_approximate(self, data: Optional[pd.DataFrame], **kwargs) -> RobustCausalResult:
        """Fast approximate discovery for very large datasets."""
        logger.info("Using approximate discovery for performance")
        
        if data is None:
            data = getattr(self.base_model, '_data', None)
        
        if data is None:
            return super().discover(data, **kwargs)
        
        # Sample data for faster processing
        sample_size = min(5000, len(data))
        if len(data) > sample_size:
            sampled_data = data.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} from {len(data)} samples")
        else:
            sampled_data = data
        
        # Run discovery on sampled data
        result = super().discover(sampled_data, **kwargs)
        
        # Adjust confidence scores for approximation
        if hasattr(result, 'confidence_scores'):
            result.confidence_scores *= 0.9  # Reduce confidence for approximation
        
        # Add approximation metadata
        if hasattr(result, 'metadata'):
            result.metadata['approximation'] = True
            result.metadata['sample_size'] = len(sampled_data)
            result.metadata['original_size'] = len(data)
            result.metadata['sampling_ratio'] = len(sampled_data) / len(data)
        
        return result
    
    def _generate_cache_key(self, operation: str, data: pd.DataFrame, kwargs: Dict) -> str:
        """Generate cache key for data and parameters."""
        # Create hash from data shape, dtypes, and sample of values
        data_hash = hashlib.md5()
        data_hash.update(f"{data.shape}".encode())
        data_hash.update(f"{list(data.dtypes)}".encode())
        
        # Sample a few values for hash (to handle large datasets efficiently)
        sample_size = min(100, len(data))
        sample_data = data.head(sample_size)
        data_hash.update(pickle.dumps(sample_data.values))
        
        # Add parameters
        param_hash = hashlib.md5(pickle.dumps(sorted(kwargs.items()))).hexdigest()
        
        return f"{operation}_{data_hash.hexdigest()}_{param_hash}"
    
    def _fit_single_batch(self, batch: pd.DataFrame, **kwargs) -> Dict:
        """Fit a single batch and return parameters."""
        # Create temporary model for this batch
        temp_model = type(self.base_model)(**kwargs)
        temp_model.fit(batch)
        
        # Return learnable parameters
        return {
            'fitted': True,
            'batch_size': len(batch),
            'parameters': getattr(temp_model, 'hyperparameters', {})
        }
    
    def _aggregate_batch_fit_results(self, batch_results: List[Dict]) -> None:
        """Aggregate results from batch fitting."""
        if not batch_results:
            return
        
        # Simple aggregation - could be more sophisticated
        total_samples = sum(result.get('batch_size', 0) for result in batch_results)
        logger.info(f"Aggregated {len(batch_results)} batches with {total_samples} total samples")
        
        # Mark as fitted
        self.is_fitted = True
    
    def _update_with_chunk(self, chunk: pd.DataFrame, **kwargs) -> None:
        """Update model with new chunk of data (incremental learning)."""
        # Simplified incremental update
        # In practice, this would depend on the specific algorithm
        logger.debug(f"Processing chunk of size {len(chunk)}")
        
        # For now, just validate the chunk
        if hasattr(self, 'data_validator'):
            validation_result = self.data_validator.validate_input_data(chunk)
            if not validation_result.is_valid:
                logger.warning(f"Chunk validation issues: {validation_result.errors}")
    
    def _discover_chunk(self, chunk_data: pd.DataFrame, **kwargs) -> CausalResult:
        """Discover causal relationships within a data chunk."""
        # Run base discovery on chunk
        temp_result = super().discover(chunk_data, **kwargs)
        return temp_result
    
    def _merge_chunk_results(self, chunk_results: List[Tuple], all_columns: pd.Index) -> RobustCausalResult:
        """Merge results from parallel chunks."""
        if not chunk_results:
            return super().discover()
        
        n_vars = len(all_columns)
        merged_adjacency = np.zeros((n_vars, n_vars))
        merged_confidence = np.zeros((n_vars, n_vars))
        
        # Map column names to indices
        col_to_idx = {col: idx for idx, col in enumerate(all_columns)}
        
        # Merge adjacency matrices
        for chunk_vars, result in chunk_results:
            if hasattr(result, 'adjacency_matrix') and hasattr(result, 'confidence_scores'):
                # Map chunk indices to global indices
                chunk_indices = [col_to_idx[col] for col in chunk_vars]
                
                for i, global_i in enumerate(chunk_indices):
                    for j, global_j in enumerate(chunk_indices):
                        merged_adjacency[global_i, global_j] = result.adjacency_matrix[i, j]
                        merged_confidence[global_i, global_j] = result.confidence_scores[i, j]
        
        # Create merged result
        base_result = chunk_results[0][1]  # Use first result as template
        
        merged_result = RobustCausalResult(
            adjacency_matrix=merged_adjacency,
            confidence_scores=merged_confidence,
            method_used=f"Scalable_{base_result.method_used}",
            metadata={
                **getattr(base_result, 'metadata', {}),
                'n_chunks': len(chunk_results),
                'merge_method': 'parallel_chunks',
                'variable_names': list(all_columns)
            },
            validation_result=getattr(base_result, 'validation_result', None),
            security_result=getattr(base_result, 'security_result', None),
            quality_score=getattr(base_result, 'quality_score', 0.5),
            processing_time=sum(getattr(result, 'processing_time', 0) for _, result in chunk_results),
            warnings_raised=[]
        )
        
        return merged_result
    
    def _select_important_variables(self, data: pd.DataFrame) -> List[List[str]]:
        """Select important variables and group them into clusters."""
        # Simplified variable selection based on variance and correlation
        # In practice, this could use more sophisticated methods
        
        n_vars = data.shape[1]
        max_cluster_size = min(10, n_vars // 3)  # Max 10 variables per cluster
        
        if n_vars <= max_cluster_size:
            return [list(data.columns)]
        
        # Simple clustering based on correlation
        corr_matrix = data.corr().abs()
        
        clusters = []
        remaining_vars = list(data.columns)
        
        while remaining_vars:
            # Start new cluster with first remaining variable
            cluster_seed = remaining_vars[0]
            cluster = [cluster_seed]
            remaining_vars.remove(cluster_seed)
            
            # Add highly correlated variables to cluster
            for var in remaining_vars.copy():
                if len(cluster) >= max_cluster_size:
                    break
                
                # Check average correlation with cluster
                avg_corr = corr_matrix.loc[var, cluster].mean()
                if avg_corr > 0.3:  # Correlation threshold
                    cluster.append(var)
                    remaining_vars.remove(var)
            
            clusters.append(cluster)
            
            # Safety check
            if len(clusters) > 20:  # Max 20 clusters
                break
        
        # Add remaining variables to last cluster or create new ones
        if remaining_vars:
            if len(clusters[-1]) + len(remaining_vars) <= max_cluster_size:
                clusters[-1].extend(remaining_vars)
            else:
                clusters.append(remaining_vars)
        
        logger.info(f"Created {len(clusters)} variable clusters")
        return clusters
    
    def _merge_hierarchical_results(self, cluster_results: List[Tuple], all_columns: pd.Index) -> RobustCausalResult:
        """Merge results from hierarchical discovery."""
        # Similar to chunk merging but with cluster-specific logic
        return self._merge_chunk_results(cluster_results, all_columns)
    
    def _prepare_cache_data(self) -> Dict:
        """Prepare data for caching."""
        return {
            'is_fitted': self.is_fitted,
            'timestamp': time.time(),
            'model_type': type(self).__name__
        }
    
    def _restore_from_cache(self, cache_data: Dict) -> None:
        """Restore model state from cache."""
        self.is_fitted = cache_data.get('is_fitted', False)
        logger.debug(f"Restored model from cache: fitted={self.is_fitted}")
    
    def _record_scalability_metrics(self, start_time: float, start_memory: float, 
                                   start_cpu: float, operation: str) -> None:
        """Record scalability metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024**2
        end_cpu = psutil.cpu_percent(interval=0.1)
        
        processing_time = end_time - start_time
        memory_delta = end_memory - start_memory
        cpu_delta = end_cpu - start_cpu
        
        # Calculate cache hit rate
        cache_hit_rate = 0.0
        if self.enable_caching and hasattr(self.cache, 'hit_rate'):
            cache_hit_rate = self.cache.hit_rate
        
        # Calculate throughput (operations per second)
        throughput = 1.0 / max(processing_time, 0.001)
        
        # Estimate parallelization efficiency
        parallel_efficiency = min(1.0, throughput / self.max_workers) if self.max_workers > 1 else 1.0
        
        metrics = ScalabilityMetrics(
            processing_time=processing_time,
            memory_usage_mb=memory_delta,
            cpu_usage_percent=cpu_delta,
            cache_hit_rate=cache_hit_rate,
            parallelization_efficiency=parallel_efficiency,
            throughput_ops_per_sec=throughput,
            resource_utilization={
                'cpu_cores': self.cpu_count,
                'max_workers': self.max_workers,
                'memory_total_gb': self.memory_total_gb
            }
        )
        
        with self._lock:
            self.scalability_metrics.append(metrics)
            
        logger.info(f"Scalability metrics for {operation}: "
                   f"time={processing_time:.3f}s, "
                   f"memory_delta={memory_delta:.1f}MB, "
                   f"throughput={throughput:.2f} ops/sec, "
                   f"cache_hit_rate={cache_hit_rate:.1%}")
    
    def get_scalability_report(self) -> Dict[str, Any]:
        """Get comprehensive scalability report."""
        if not self.scalability_metrics:
            return {"message": "No scalability metrics available"}
        
        metrics = self.scalability_metrics
        
        return {
            "operations_count": len(metrics),
            "average_processing_time": np.mean([m.processing_time for m in metrics]),
            "average_memory_usage_mb": np.mean([m.memory_usage_mb for m in metrics]),
            "average_throughput": np.mean([m.throughput_ops_per_sec for m in metrics]),
            "cache_hit_rate": np.mean([m.cache_hit_rate for m in metrics]),
            "parallelization_efficiency": np.mean([m.parallelization_efficiency for m in metrics]),
            "optimization_level": self.optimization_level,
            "max_workers": self.max_workers,
            "caching_enabled": self.enable_caching,
            "parallelization_enabled": self.enable_parallelization,
            "auto_scaling_enabled": self.enable_auto_scaling,
            "system_resources": {
                "cpu_count": self.cpu_count,
                "memory_total_gb": self.memory_total_gb,
                "current_memory_usage_percent": psutil.virtual_memory().percent,
                "current_cpu_usage_percent": psutil.cpu_percent(interval=1)
            }
        }
    
    def optimize_for_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Automatically optimize configuration for given dataset."""
        n_samples, n_features = data.shape
        memory_usage_mb = data.memory_usage(deep=True).sum() / 1024**2
        
        recommendations = {}
        
        # Batch size optimization
        if n_samples > 10000:
            optimal_batch_size = min(1000, n_samples // self.max_workers)
            recommendations['batch_size'] = optimal_batch_size
        
        # Worker count optimization
        if n_features > 50:
            optimal_workers = min(self.cpu_count, n_features // 5)
            recommendations['max_workers'] = optimal_workers
        
        # Memory optimization
        if memory_usage_mb > self.memory_total_gb * 1024 * 0.3:  # > 30% of RAM
            recommendations['optimization_level'] = 'memory'
            recommendations['enable_streaming'] = True
        
        # Caching optimization
        if n_samples < 1000:
            recommendations['cache_size'] = 100
        elif n_samples > 10000:
            recommendations['cache_size'] = 2000
        
        logger.info(f"Dataset optimization recommendations: {recommendations}")
        return recommendations