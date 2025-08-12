"""Distributed and high-performance causal discovery algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import logging
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
import queue
import warnings
from functools import partial
import gc

try:
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.performance import ConcurrentProcessor, PerformanceProfiler
except ImportError:
    try:
        from base import CausalDiscoveryModel, CausalResult
        from utils.performance import ConcurrentProcessor, PerformanceProfiler
    except ImportError:
        # Final fallback - minimal implementations
        from base import CausalDiscoveryModel, CausalResult
        
        class ConcurrentProcessor:
            def __init__(self, *args, **kwargs): 
                pass
            def process_batch(self, func, items): 
                return [func(item) for item in items]
        
        class PerformanceProfiler:
            def __init__(self): 
                pass
            def profile(self, func): 
                return func
            def get_stats(self): 
                return {}

logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Task for distributed causal discovery."""
    task_id: str
    data_chunk: pd.DataFrame
    parameters: Dict[str, Any]
    method_name: str
    chunk_info: Dict[str, Any] = None


@dataclass 
class DistributedResult:
    """Result from distributed causal discovery."""
    task_id: str
    adjacency_matrix: np.ndarray
    confidence_scores: np.ndarray
    processing_time: float
    chunk_info: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class DistributedCausalDiscovery(CausalDiscoveryModel):
    """Distributed causal discovery for large datasets."""
    
    def __init__(self,
                 base_model_class: type,
                 chunk_size: int = 1000,
                 overlap_size: int = 100,
                 n_processes: int = None,
                 aggregation_method: str = "weighted_average",
                 enable_caching: bool = True,
                 memory_limit_gb: float = 2.0,
                 **base_model_kwargs):
        """
        Initialize distributed causal discovery.
        
        Args:
            base_model_class: Base causal discovery model class
            chunk_size: Size of data chunks for parallel processing
            overlap_size: Overlap between chunks for continuity
            n_processes: Number of processes (None for auto)
            aggregation_method: Method to aggregate results
            enable_caching: Enable result caching
            memory_limit_gb: Memory limit per process
        """
        super().__init__()
        self.base_model_class = base_model_class
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.aggregation_method = aggregation_method
        self.enable_caching = enable_caching
        self.memory_limit_gb = memory_limit_gb
        self.base_model_kwargs = base_model_kwargs
        
        self._data = None
        self._result_cache = {} if enable_caching else None
        
        # Initialize performance monitoring
        self.profiler = PerformanceProfiler()
        
        logger.info(f"Initialized distributed discovery with {self.n_processes} processes")
    
    def fit(self, data: pd.DataFrame) -> 'DistributedCausalDiscovery':
        """Fit distributed model."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        self._data = data.copy()
        self.is_fitted = True
        
        # Pre-compute data chunks
        self._data_chunks = self._create_data_chunks(data)
        logger.info(f"Created {len(self._data_chunks)} data chunks for processing")
        
        return self
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using distributed processing."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
        
        data_to_use = data if data is not None else self._data
        
        # Create tasks
        tasks = self._create_distributed_tasks(data_to_use)
        
        # Execute distributed processing
        chunk_results = self._execute_distributed_tasks(tasks)
        
        # Aggregate results
        final_result = self._aggregate_results(chunk_results, data_to_use)
        
        return final_result
    
    def _create_data_chunks(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Create overlapping data chunks."""
        chunks = []
        n_samples = len(data)
        
        start = 0
        while start < n_samples:
            end = min(start + self.chunk_size, n_samples)
            chunks.append((start, end))
            
            # Move start position accounting for overlap
            start = end - self.overlap_size
            if start >= n_samples:
                break
        
        return chunks
    
    def _create_distributed_tasks(self, data: pd.DataFrame) -> List[DistributedTask]:
        """Create distributed processing tasks."""
        tasks = []
        
        for i, (start, end) in enumerate(self._data_chunks):
            chunk_data = data.iloc[start:end].copy()
            
            task = DistributedTask(
                task_id=f"chunk_{i:04d}",
                data_chunk=chunk_data,
                parameters=self.base_model_kwargs,
                method_name=self.base_model_class.__name__,
                chunk_info={
                    'start': start,
                    'end': end,
                    'size': len(chunk_data),
                    'chunk_index': i
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _execute_distributed_tasks(self, tasks: List[DistributedTask]) -> List[DistributedResult]:
        """Execute tasks in distributed manner."""
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound causal discovery
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_task, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed task {task.task_id}")
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {str(e)}")
                    # Create empty result to maintain structure
                    empty_result = DistributedResult(
                        task_id=task.task_id,
                        adjacency_matrix=np.zeros((len(task.data_chunk.columns), len(task.data_chunk.columns))),
                        confidence_scores=np.zeros((len(task.data_chunk.columns), len(task.data_chunk.columns))),
                        processing_time=0.0,
                        chunk_info=task.chunk_info,
                        metadata={'error': str(e)}
                    )
                    results.append(empty_result)
        
        # Sort results by chunk index to maintain order
        results.sort(key=lambda r: r.chunk_info['chunk_index'] if r.chunk_info else 0)
        
        return results
    
    @staticmethod
    def _process_single_task(task: DistributedTask) -> DistributedResult:
        """Process a single distributed task (static method for multiprocessing)."""
        start_time = time.time()
        
        try:
            # Import required modules in worker process
            from base import SimpleLinearCausalModel
            
            # Map method names to classes (simplified for this implementation)
            model_map = {
                'SimpleLinearCausalModel': SimpleLinearCausalModel
            }
            
            model_class = model_map.get(task.method_name, SimpleLinearCausalModel)
            
            # Create and run model
            model = model_class(**task.parameters)
            result = model.fit_discover(task.data_chunk)
            
            processing_time = time.time() - start_time
            
            return DistributedResult(
                task_id=task.task_id,
                adjacency_matrix=result.adjacency_matrix,
                confidence_scores=result.confidence_scores,
                processing_time=processing_time,
                chunk_info=task.chunk_info,
                metadata=result.metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing task {task.task_id}: {str(e)}")
            
            # Return empty result
            n_vars = len(task.data_chunk.columns)
            return DistributedResult(
                task_id=task.task_id,
                adjacency_matrix=np.zeros((n_vars, n_vars)),
                confidence_scores=np.zeros((n_vars, n_vars)),
                processing_time=processing_time,
                chunk_info=task.chunk_info,
                metadata={'error': str(e)}
            )
    
    def _aggregate_results(self, chunk_results: List[DistributedResult], 
                          original_data: pd.DataFrame) -> CausalResult:
        """Aggregate distributed results."""
        
        if not chunk_results:
            raise RuntimeError("No results to aggregate")
        
        # Filter out failed results
        valid_results = [r for r in chunk_results if 'error' not in r.metadata]
        
        if not valid_results:
            raise RuntimeError("All distributed tasks failed")
        
        logger.info(f"Aggregating {len(valid_results)}/{len(chunk_results)} valid results")
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(valid_results, original_data)
        elif self.aggregation_method == "majority_vote":
            return self._majority_vote_aggregation(valid_results, original_data)
        elif self.aggregation_method == "confidence_weighted":
            return self._confidence_weighted_aggregation(valid_results, original_data)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _weighted_average_aggregation(self, results: List[DistributedResult], 
                                     original_data: pd.DataFrame) -> CausalResult:
        """Aggregate using weighted average based on chunk sizes."""
        
        total_weight = 0
        weighted_adjacency = None
        weighted_confidence = None
        
        for result in results:
            weight = result.chunk_info['size'] if result.chunk_info else 1
            total_weight += weight
            
            if weighted_adjacency is None:
                weighted_adjacency = weight * result.adjacency_matrix.astype(float)
                weighted_confidence = weight * result.confidence_scores
            else:
                weighted_adjacency += weight * result.adjacency_matrix.astype(float)
                weighted_confidence += weight * result.confidence_scores
        
        # Normalize by total weight
        final_adjacency = (weighted_adjacency / total_weight > 0.5).astype(int)
        final_confidence = weighted_confidence / total_weight
        
        # Aggregate metadata
        total_processing_time = sum(r.processing_time for r in results)
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=final_confidence,
            method_used=f"Distributed({self.base_model_class.__name__})",
            metadata={
                "aggregation_method": self.aggregation_method,
                "n_chunks": len(results),
                "total_processing_time": total_processing_time,
                "n_processes": self.n_processes,
                "chunk_size": self.chunk_size,
                "overlap_size": self.overlap_size,
                "n_variables": len(original_data.columns),
                "n_edges": int(np.sum(final_adjacency)),
                "variable_names": list(original_data.columns)
            }
        )
    
    def _majority_vote_aggregation(self, results: List[DistributedResult], 
                                  original_data: pd.DataFrame) -> CausalResult:
        """Aggregate using majority vote."""
        
        # Stack adjacency matrices
        adjacency_stack = np.stack([r.adjacency_matrix for r in results])
        confidence_stack = np.stack([r.confidence_scores for r in results])
        
        # Majority vote
        vote_counts = np.sum(adjacency_stack, axis=0)
        final_adjacency = (vote_counts > len(results) / 2).astype(int)
        
        # Average confidence
        final_confidence = np.mean(confidence_stack, axis=0)
        
        total_processing_time = sum(r.processing_time for r in results)
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=final_confidence,
            method_used=f"Distributed({self.base_model_class.__name__})",
            metadata={
                "aggregation_method": self.aggregation_method,
                "n_chunks": len(results),
                "total_processing_time": total_processing_time,
                "n_variables": len(original_data.columns),
                "n_edges": int(np.sum(final_adjacency)),
                "variable_names": list(original_data.columns)
            }
        )
    
    def _confidence_weighted_aggregation(self, results: List[DistributedResult], 
                                        original_data: pd.DataFrame) -> CausalResult:
        """Aggregate using confidence-weighted combination."""
        
        total_confidence_weight = None
        weighted_adjacency = None
        
        for result in results:
            confidence_weights = result.confidence_scores
            
            if total_confidence_weight is None:
                total_confidence_weight = confidence_weights
                weighted_adjacency = confidence_weights * result.adjacency_matrix.astype(float)
            else:
                total_confidence_weight += confidence_weights
                weighted_adjacency += confidence_weights * result.adjacency_matrix.astype(float)
        
        # Avoid division by zero
        total_confidence_weight = np.maximum(total_confidence_weight, 1e-8)
        
        final_adjacency = (weighted_adjacency / total_confidence_weight > 0.5).astype(int)
        final_confidence = weighted_adjacency / total_confidence_weight
        
        total_processing_time = sum(r.processing_time for r in results)
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=final_confidence,
            method_used=f"Distributed({self.base_model_class.__name__})",
            metadata={
                "aggregation_method": self.aggregation_method,
                "n_chunks": len(results),
                "total_processing_time": total_processing_time,
                "n_variables": len(original_data.columns),
                "n_edges": int(np.sum(final_adjacency)),
                "variable_names": list(original_data.columns)
            }
        )


class StreamingCausalDiscovery(CausalDiscoveryModel):
    """Streaming causal discovery for real-time data."""
    
    def __init__(self,
                 base_model_class: type,
                 window_size: int = 1000,
                 update_frequency: int = 100,
                 decay_factor: float = 0.95,
                 min_samples_for_update: int = 50,
                 **base_model_kwargs):
        """
        Initialize streaming causal discovery.
        
        Args:
            base_model_class: Base causal discovery model class
            window_size: Size of sliding window
            update_frequency: How often to update model
            decay_factor: Exponential decay for historical results
            min_samples_for_update: Minimum samples before updating
        """
        super().__init__()
        self.base_model_class = base_model_class
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.decay_factor = decay_factor
        self.min_samples_for_update = min_samples_for_update
        self.base_model_kwargs = base_model_kwargs
        
        # Streaming state
        self._data_buffer = pd.DataFrame()
        self._current_result = None
        self._update_counter = 0
        self._historical_results = []
        
    def fit(self, initial_data: pd.DataFrame) -> 'StreamingCausalDiscovery':
        """Initialize with initial data."""
        self._data_buffer = initial_data.copy()
        self._update_model()
        self.is_fitted = True
        return self
    
    def update(self, new_data: pd.DataFrame) -> Optional[CausalResult]:
        """Update model with new streaming data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before updates")
        
        # Add new data to buffer
        self._data_buffer = pd.concat([self._data_buffer, new_data], ignore_index=True)
        
        # Maintain window size
        if len(self._data_buffer) > self.window_size:
            self._data_buffer = self._data_buffer.tail(self.window_size)
        
        self._update_counter += len(new_data)
        
        # Check if we should update model
        if (self._update_counter >= self.update_frequency and 
            len(self._data_buffer) >= self.min_samples_for_update):
            
            self._update_model()
            self._update_counter = 0
            return self._current_result
        
        return None
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Get current causal discovery result."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
        
        if data is not None:
            # Discover on new data immediately
            model = self.base_model_class(**self.base_model_kwargs)
            return model.fit_discover(data)
        
        if self._current_result is None:
            raise RuntimeError("No current result available")
        
        return self._current_result
    
    def _update_model(self):
        """Update the causal model with current buffer."""
        try:
            model = self.base_model_class(**self.base_model_kwargs)
            new_result = model.fit_discover(self._data_buffer)
            
            if self._current_result is None:
                # First result
                self._current_result = new_result
            else:
                # Combine with historical results using exponential decay
                self._current_result = self._combine_with_history(new_result)
            
            # Store in history
            self._historical_results.append(new_result)
            
            # Limit history size
            max_history = 10
            if len(self._historical_results) > max_history:
                self._historical_results = self._historical_results[-max_history:]
            
            logger.debug(f"Updated streaming model, buffer size: {len(self._data_buffer)}")
            
        except Exception as e:
            logger.error(f"Failed to update streaming model: {str(e)}")
    
    def _combine_with_history(self, new_result: CausalResult) -> CausalResult:
        """Combine new result with historical results using decay."""
        
        # Exponential decay combination
        decayed_adjacency = self.decay_factor * self._current_result.adjacency_matrix.astype(float)
        decayed_confidence = self.decay_factor * self._current_result.confidence_scores
        
        new_weight = 1 - self.decay_factor
        combined_adjacency = decayed_adjacency + new_weight * new_result.adjacency_matrix.astype(float)
        combined_confidence = decayed_confidence + new_weight * new_result.confidence_scores
        
        # Threshold for binary adjacency
        final_adjacency = (combined_adjacency > 0.5).astype(int)
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=combined_confidence,
            method_used=f"Streaming({self.base_model_class.__name__})",
            metadata={
                "decay_factor": self.decay_factor,
                "window_size": self.window_size,
                "buffer_size": len(self._data_buffer),
                "n_updates": len(self._historical_results),
                "n_variables": len(self._data_buffer.columns) if not self._data_buffer.empty else 0,
                "n_edges": int(np.sum(final_adjacency)),
                "variable_names": list(self._data_buffer.columns) if not self._data_buffer.empty else []
            }
        )


class MemoryEfficientDiscovery(CausalDiscoveryModel):
    """Memory-efficient causal discovery for large datasets."""
    
    def __init__(self,
                 base_model_class: type,
                 memory_budget_gb: float = 1.0,
                 chunk_strategy: str = "adaptive",
                 compression_enabled: bool = True,
                 disk_cache_enabled: bool = True,
                 **base_model_kwargs):
        """
        Initialize memory-efficient discovery.
        
        Args:
            base_model_class: Base causal discovery model
            memory_budget_gb: Memory budget in GB
            chunk_strategy: Strategy for data chunking
            compression_enabled: Enable data compression
            disk_cache_enabled: Enable disk caching
        """
        super().__init__()
        self.base_model_class = base_model_class
        self.memory_budget_gb = memory_budget_gb
        self.chunk_strategy = chunk_strategy
        self.compression_enabled = compression_enabled
        self.disk_cache_enabled = disk_cache_enabled
        self.base_model_kwargs = base_model_kwargs
        
        # Memory management
        self._memory_budget_bytes = int(memory_budget_gb * 1024**3)
        self._data_cache = {}
        
    def fit(self, data: pd.DataFrame) -> 'MemoryEfficientDiscovery':
        """Fit with memory-efficient processing."""
        
        # Estimate memory requirements
        data_size_bytes = data.memory_usage(deep=True).sum()
        
        if data_size_bytes > self._memory_budget_bytes:
            logger.info(f"Data size ({data_size_bytes/1024**3:.2f}GB) exceeds budget "
                       f"({self.memory_budget_gb}GB), using chunked processing")
            self._use_chunked_processing = True
            self._chunk_size = self._calculate_optimal_chunk_size(data)
        else:
            self._use_chunked_processing = False
            self._chunk_size = len(data)
        
        self._data = data.copy()
        self.is_fitted = True
        
        return self
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Memory-efficient causal discovery."""
        
        data_to_use = data if data is not None else self._data
        
        if not self._use_chunked_processing:
            # Process normally
            model = self.base_model_class(**self.base_model_kwargs)
            return model.fit_discover(data_to_use)
        else:
            # Process in memory-efficient chunks
            return self._chunked_discovery(data_to_use)
    
    def _calculate_optimal_chunk_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on memory budget."""
        
        # Estimate bytes per sample
        bytes_per_sample = data.memory_usage(deep=True).sum() / len(data)
        
        # Reserve space for processing overhead (factor of 3)
        available_bytes = self._memory_budget_bytes // 3
        
        # Calculate chunk size
        chunk_size = int(available_bytes / bytes_per_sample)
        
        # Ensure minimum viable chunk size
        min_chunk_size = min(100, len(data) // 10)
        chunk_size = max(chunk_size, min_chunk_size)
        
        logger.info(f"Calculated optimal chunk size: {chunk_size} samples")
        return chunk_size
    
    def _chunked_discovery(self, data: pd.DataFrame) -> CausalResult:
        """Process data in memory-efficient chunks."""
        
        n_samples = len(data)
        results = []
        
        for start in range(0, n_samples, self._chunk_size):
            end = min(start + self._chunk_size, n_samples)
            chunk = data.iloc[start:end]
            
            logger.debug(f"Processing chunk {start}-{end} ({len(chunk)} samples)")
            
            try:
                # Process chunk
                model = self.base_model_class(**self.base_model_kwargs)
                chunk_result = model.fit_discover(chunk)
                results.append(chunk_result)
                
                # Force garbage collection to free memory
                del model, chunk
                gc.collect()
                
            except MemoryError:
                logger.warning(f"Memory error in chunk {start}-{end}, skipping")
                continue
            except Exception as e:
                logger.error(f"Error in chunk {start}-{end}: {str(e)}")
                continue
        
        if not results:
            raise RuntimeError("All chunks failed to process")
        
        # Aggregate chunk results
        return self._aggregate_chunk_results(results, data)
    
    def _aggregate_chunk_results(self, results: List[CausalResult], 
                                original_data: pd.DataFrame) -> CausalResult:
        """Aggregate results from memory-efficient chunks."""
        
        # Simple averaging aggregation
        adjacency_sum = np.zeros_like(results[0].adjacency_matrix, dtype=float)
        confidence_sum = np.zeros_like(results[0].confidence_scores, dtype=float)
        
        for result in results:
            adjacency_sum += result.adjacency_matrix.astype(float)
            confidence_sum += result.confidence_scores
        
        n_results = len(results)
        final_adjacency = (adjacency_sum / n_results > 0.5).astype(int)
        final_confidence = confidence_sum / n_results
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=final_confidence,
            method_used=f"MemoryEfficient({self.base_model_class.__name__})",
            metadata={
                "memory_budget_gb": self.memory_budget_gb,
                "chunk_size": self._chunk_size,
                "n_chunks": n_results,
                "chunked_processing": self._use_chunked_processing,
                "n_variables": len(original_data.columns),
                "n_edges": int(np.sum(final_adjacency)),
                "variable_names": list(original_data.columns)
            }
        )