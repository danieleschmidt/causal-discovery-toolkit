"""Auto-scaling and adaptive resource management for causal discovery."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import logging
import time
import psutil
import threading
import queue
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    timestamp: float
    
    def is_resource_stressed(self, cpu_threshold=85.0, memory_threshold=85.0) -> bool:
        """Check if system resources are stressed."""
        return self.cpu_percent > cpu_threshold or self.memory_percent > memory_threshold


@dataclass
class WorkloadCharacteristics:
    """Characteristics of the causal discovery workload."""
    data_size_mb: float
    n_variables: int
    n_samples: int
    complexity_score: float
    estimated_runtime_seconds: float
    memory_requirement_gb: float


class ResourceMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 100):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self._metrics_history = []
        self._monitoring_thread = None
        self._stop_monitoring = False
        self._metrics_queue = queue.Queue()
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring already running")
            return
            
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._stop_monitoring = True
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=2.0)
        logger.info("Stopped resource monitoring")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Get disk I/O if available
        try:
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        except AttributeError:
            disk_read_mb = disk_write_mb = 0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            timestamp=time.time()
        )
    
    def get_metrics_history(self) -> List[ResourceMetrics]:
        """Get historical metrics."""
        return self._metrics_history.copy()
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                metrics = self.get_current_metrics()
                self._metrics_queue.put(metrics)
                
                # Update history
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self.history_size:
                    self._metrics_history.pop(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)


class WorkloadEstimator:
    """Estimate workload characteristics for optimal resource allocation."""
    
    def __init__(self):
        self.baseline_metrics = {}
        
    def estimate_workload(self, data: pd.DataFrame, algorithm_name: str, 
                         parameters: Dict[str, Any]) -> WorkloadCharacteristics:
        """Estimate workload characteristics."""
        
        n_samples, n_variables = data.shape
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Calculate complexity score based on algorithm and parameters
        complexity_score = self._calculate_complexity_score(
            n_samples, n_variables, algorithm_name, parameters
        )
        
        # Estimate runtime based on complexity
        estimated_runtime = self._estimate_runtime(complexity_score, n_samples, n_variables)
        
        # Estimate memory requirements
        memory_requirement = self._estimate_memory_requirement(
            data_size_mb, complexity_score, algorithm_name
        )
        
        return WorkloadCharacteristics(
            data_size_mb=data_size_mb,
            n_variables=n_variables,
            n_samples=n_samples,
            complexity_score=complexity_score,
            estimated_runtime_seconds=estimated_runtime,
            memory_requirement_gb=memory_requirement
        )
    
    def _calculate_complexity_score(self, n_samples: int, n_variables: int, 
                                   algorithm_name: str, parameters: Dict[str, Any]) -> float:
        """Calculate algorithm complexity score."""
        base_complexity = n_samples * n_variables
        
        # Algorithm-specific complexity adjustments
        algorithm_factors = {
            'SimpleLinearCausalModel': 1.0,
            'BayesianNetworkDiscovery': 2.5,
            'ConstraintBasedDiscovery': 3.0,
            'MutualInformationDiscovery': 2.0,
            'TransferEntropyDiscovery': 1.8,
            'EnsembleDiscovery': 4.0
        }
        
        algorithm_factor = algorithm_factors.get(algorithm_name, 1.5)
        
        # Parameter-specific adjustments
        param_factor = 1.0
        if 'max_parents' in parameters:
            param_factor *= (1 + parameters['max_parents'] * 0.2)
        if 'bootstrap_samples' in parameters:
            param_factor *= (1 + parameters['bootstrap_samples'] * 0.001)
        if 'n_bins' in parameters:
            param_factor *= (1 + parameters['n_bins'] * 0.05)
        
        return base_complexity * algorithm_factor * param_factor
    
    def _estimate_runtime(self, complexity_score: float, n_samples: int, n_variables: int) -> float:
        """Estimate runtime in seconds."""
        # Base estimation: assume 1M complexity units = 1 second
        base_time = complexity_score / 1e6
        
        # Adjust based on variable count (combinatorial explosion)
        variable_factor = min(n_variables**1.5 / 100, 10)
        
        # Minimum runtime
        min_time = 0.1
        
        return max(base_time * variable_factor, min_time)
    
    def _estimate_memory_requirement(self, data_size_mb: float, complexity_score: float, 
                                    algorithm_name: str) -> float:
        """Estimate memory requirement in GB."""
        base_memory = data_size_mb / 1024  # Convert to GB
        
        # Algorithm-specific memory multipliers
        memory_multipliers = {
            'SimpleLinearCausalModel': 2.0,
            'BayesianNetworkDiscovery': 4.0,
            'ConstraintBasedDiscovery': 3.5,
            'MutualInformationDiscovery': 3.0,
            'TransferEntropyDiscovery': 2.5,
            'EnsembleDiscovery': 6.0
        }
        
        multiplier = memory_multipliers.get(algorithm_name, 3.0)
        
        # Add overhead for intermediate computations
        overhead_factor = 1.5
        
        # Complexity-based additional memory
        complexity_memory = (complexity_score / 1e8) * 0.5  # 0.5GB per 100M complexity units
        
        total_memory = (base_memory * multiplier * overhead_factor) + complexity_memory
        
        # Minimum and maximum bounds
        return max(min(total_memory, 32.0), 0.1)  # Min 100MB, Max 32GB


class AutoScaler:
    """Automatic resource scaling for causal discovery."""
    
    def __init__(self, 
                 max_processes: int = None,
                 max_memory_gb: float = None,
                 scaling_strategy: str = "conservative"):
        """
        Initialize auto-scaler.
        
        Args:
            max_processes: Maximum number of processes to use
            max_memory_gb: Maximum memory to use
            scaling_strategy: Scaling strategy ('aggressive', 'conservative', 'adaptive')
        """
        self.max_processes = max_processes or psutil.cpu_count()
        self.max_memory_gb = max_memory_gb or (psutil.virtual_memory().total / (1024**3) * 0.8)
        self.scaling_strategy = scaling_strategy
        
        self.resource_monitor = ResourceMonitor()
        self.workload_estimator = WorkloadEstimator()
        
        # Scaling parameters
        self.scaling_params = self._get_scaling_parameters()
        
    def _get_scaling_parameters(self) -> Dict[str, Any]:
        """Get scaling parameters based on strategy."""
        strategies = {
            'conservative': {
                'cpu_utilization_target': 70.0,
                'memory_utilization_target': 75.0,
                'scale_up_threshold': 85.0,
                'scale_down_threshold': 50.0,
                'max_concurrent_jobs': self.max_processes // 2
            },
            'aggressive': {
                'cpu_utilization_target': 90.0,
                'memory_utilization_target': 85.0,
                'scale_up_threshold': 95.0,
                'scale_down_threshold': 70.0,
                'max_concurrent_jobs': self.max_processes
            },
            'adaptive': {
                'cpu_utilization_target': 80.0,
                'memory_utilization_target': 80.0,
                'scale_up_threshold': 90.0,
                'scale_down_threshold': 60.0,
                'max_concurrent_jobs': int(self.max_processes * 0.8)
            }
        }
        
        return strategies.get(self.scaling_strategy, strategies['conservative'])
    
    def get_optimal_configuration(self, workload: WorkloadCharacteristics) -> Dict[str, Any]:
        """Get optimal configuration for the given workload."""
        
        # Start monitoring if not already running
        if not self.resource_monitor._monitoring_thread or not self.resource_monitor._monitoring_thread.is_alive():
            self.resource_monitor.start_monitoring()
            time.sleep(1)  # Allow some monitoring data to accumulate
        
        current_metrics = self.resource_monitor.get_current_metrics()
        
        # Determine optimal number of processes
        optimal_processes = self._calculate_optimal_processes(workload, current_metrics)
        
        # Determine optimal memory allocation
        optimal_memory = self._calculate_optimal_memory(workload, current_metrics)
        
        # Determine chunking strategy
        chunk_config = self._determine_chunking_strategy(workload, optimal_memory)
        
        # Determine parallel execution strategy
        parallel_config = self._determine_parallel_strategy(workload, optimal_processes)
        
        config = {
            'n_processes': optimal_processes,
            'memory_limit_gb': optimal_memory,
            'chunk_size': chunk_config['chunk_size'],
            'enable_chunking': chunk_config['enable_chunking'],
            'parallel_execution': parallel_config['parallel_execution'],
            'batch_size': parallel_config['batch_size'],
            'enable_caching': workload.data_size_mb < 1000,  # Cache for smaller datasets
            'compression_enabled': workload.data_size_mb > 500,  # Compress for larger datasets
        }
        
        logger.info(f"Optimal configuration: {config}")
        return config
    
    def _calculate_optimal_processes(self, workload: WorkloadCharacteristics, 
                                   metrics: ResourceMetrics) -> int:
        """Calculate optimal number of processes."""
        
        # Start with available CPU capacity
        available_cpu = 100 - metrics.cpu_percent
        target_utilization = self.scaling_params['cpu_utilization_target']
        
        # Estimate processes that won't exceed target utilization
        if available_cpu < 20:  # System already under load
            optimal_processes = 1
        else:
            # Rule of thumb: each process uses ~CPU/n_cores %
            processes_for_target = int(target_utilization / (100 / psutil.cpu_count()))
            optimal_processes = min(processes_for_target, self.max_processes)
        
        # Adjust based on workload characteristics
        if workload.complexity_score > 1e8:  # High complexity
            optimal_processes = min(optimal_processes, 4)  # Limit for complex tasks
        elif workload.data_size_mb < 100:  # Small data
            optimal_processes = min(optimal_processes, 2)  # Don't over-parallelize
        
        # Memory constraint check
        memory_per_process = workload.memory_requirement_gb
        max_processes_by_memory = int(metrics.memory_available_gb / memory_per_process)
        optimal_processes = min(optimal_processes, max_processes_by_memory)
        
        return max(1, optimal_processes)
    
    def _calculate_optimal_memory(self, workload: WorkloadCharacteristics, 
                                metrics: ResourceMetrics) -> float:
        """Calculate optimal memory allocation."""
        
        # Base requirement from workload estimation
        base_requirement = workload.memory_requirement_gb
        
        # Available memory
        available_memory = metrics.memory_available_gb
        
        # Apply safety margins
        safety_margin = 0.8  # Use only 80% of available memory
        safe_available = available_memory * safety_margin
        
        # Choose between requirement and available
        optimal_memory = min(base_requirement, safe_available)
        
        # Minimum and maximum bounds
        optimal_memory = max(optimal_memory, 0.5)  # At least 500MB
        optimal_memory = min(optimal_memory, self.max_memory_gb)
        
        return optimal_memory
    
    def _determine_chunking_strategy(self, workload: WorkloadCharacteristics, 
                                   memory_limit: float) -> Dict[str, Any]:
        """Determine optimal chunking strategy."""
        
        # Enable chunking if data is large or memory is limited
        enable_chunking = (
            workload.data_size_mb > 500 or
            workload.memory_requirement_gb > memory_limit or
            workload.n_samples > 10000
        )
        
        if enable_chunking:
            # Calculate chunk size based on memory limit
            memory_per_chunk_mb = (memory_limit * 1024) / 3  # Use 1/3 of limit per chunk
            samples_per_mb = workload.n_samples / workload.data_size_mb
            chunk_size = int(memory_per_chunk_mb * samples_per_mb)
            
            # Apply reasonable bounds
            chunk_size = max(min(chunk_size, 5000), 100)
        else:
            chunk_size = workload.n_samples
        
        return {
            'enable_chunking': enable_chunking,
            'chunk_size': chunk_size
        }
    
    def _determine_parallel_strategy(self, workload: WorkloadCharacteristics, 
                                   n_processes: int) -> Dict[str, Any]:
        """Determine parallel execution strategy."""
        
        # Enable parallel execution for larger workloads
        parallel_execution = (
            workload.complexity_score > 1e6 and
            n_processes > 1 and
            workload.n_variables > 3
        )
        
        # Batch size for parallel processing
        if parallel_execution:
            batch_size = max(workload.n_samples // n_processes, 100)
        else:
            batch_size = workload.n_samples
        
        return {
            'parallel_execution': parallel_execution,
            'batch_size': batch_size
        }
    
    def monitor_and_adjust(self, current_config: Dict[str, Any], 
                          performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor performance and adjust configuration if needed."""
        
        current_metrics = self.resource_monitor.get_current_metrics()
        
        # Check if we need to scale up or down
        adjustment_needed = False
        new_config = current_config.copy()
        
        # CPU-based adjustments
        if current_metrics.cpu_percent > self.scaling_params['scale_up_threshold']:
            # System overloaded - scale down
            new_config['n_processes'] = max(1, current_config['n_processes'] - 1)
            adjustment_needed = True
            logger.info("Scaling down due to high CPU usage")
            
        elif current_metrics.cpu_percent < self.scaling_params['scale_down_threshold']:
            # System underutilized - scale up (if possible)
            max_increase = min(self.max_processes, current_config['n_processes'] + 1)
            if max_increase > current_config['n_processes']:
                new_config['n_processes'] = max_increase
                adjustment_needed = True
                logger.info("Scaling up due to low CPU usage")
        
        # Memory-based adjustments
        if current_metrics.memory_percent > 90:
            # High memory usage - enable more aggressive memory management
            new_config['enable_chunking'] = True
            new_config['compression_enabled'] = True
            new_config['chunk_size'] = min(current_config.get('chunk_size', 1000), 500)
            adjustment_needed = True
            logger.info("Enabling memory optimization due to high memory usage")
        
        return new_config if adjustment_needed else current_config
    
    def cleanup(self):
        """Cleanup resources."""
        self.resource_monitor.stop_monitoring()


class AdaptiveParameterTuner:
    """Adaptive parameter tuning based on performance feedback."""
    
    def __init__(self):
        self.performance_history = []
        self.parameter_history = []
        
    def suggest_parameters(self, algorithm_name: str, workload: WorkloadCharacteristics,
                          base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimized parameters based on workload."""
        
        optimized_params = base_parameters.copy()
        
        # Algorithm-specific optimizations
        if algorithm_name == 'BayesianNetworkDiscovery':
            optimized_params = self._optimize_bayesian_params(optimized_params, workload)
        elif algorithm_name == 'ConstraintBasedDiscovery':
            optimized_params = self._optimize_constraint_params(optimized_params, workload)
        elif algorithm_name == 'MutualInformationDiscovery':
            optimized_params = self._optimize_mi_params(optimized_params, workload)
        
        return optimized_params
    
    def _optimize_bayesian_params(self, params: Dict[str, Any], 
                                 workload: WorkloadCharacteristics) -> Dict[str, Any]:
        """Optimize Bayesian Network parameters."""
        
        # Adjust max_parents based on dataset size
        if workload.n_variables > 20:
            params['max_parents'] = min(params.get('max_parents', 3), 2)
        elif workload.n_variables < 5:
            params['max_parents'] = min(params.get('max_parents', 3), workload.n_variables - 1)
        
        # Adjust bootstrap based on sample size
        if workload.n_samples < 500:
            params['use_bootstrap'] = False
        elif workload.n_samples > 5000:
            params['bootstrap_samples'] = min(params.get('bootstrap_samples', 100), 50)
        
        return params
    
    def _optimize_constraint_params(self, params: Dict[str, Any], 
                                   workload: WorkloadCharacteristics) -> Dict[str, Any]:
        """Optimize Constraint-based parameters."""
        
        # Adjust alpha based on sample size
        if workload.n_samples < 300:
            params['alpha'] = max(params.get('alpha', 0.05), 0.1)  # More lenient
        elif workload.n_samples > 2000:
            params['alpha'] = min(params.get('alpha', 0.05), 0.01)  # More stringent
        
        # Adjust conditioning set size based on variable count
        if workload.n_variables > 15:
            params['max_conditioning_set_size'] = min(params.get('max_conditioning_set_size', 3), 2)
        
        return params
    
    def _optimize_mi_params(self, params: Dict[str, Any], 
                           workload: WorkloadCharacteristics) -> Dict[str, Any]:
        """Optimize Mutual Information parameters."""
        
        # Adjust binning based on sample size
        if workload.n_samples < 500:
            params['n_bins'] = min(params.get('n_bins', 10), 5)
        elif workload.n_samples > 5000:
            params['n_bins'] = min(params.get('n_bins', 10), 15)
        
        # Disable conditional MI for small datasets
        if workload.n_samples < 200 or workload.n_variables < 4:
            params['use_conditional_mi'] = False
        
        return params