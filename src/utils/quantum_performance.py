"""Quantum performance optimization and caching systems."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import hashlib
import pickle
from pathlib import Path
import threading
from collections import OrderedDict
import psutil


@dataclass
class CacheEntry:
    """Cache entry for quantum computations."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    computation_time: float
    memory_size: int


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    algorithm_name: str
    execution_time: float
    memory_peak_mb: float
    cpu_utilization: float
    cache_efficiency: float
    quantum_operations: int
    parallel_efficiency: float


class QuantumCache:
    """High-performance cache for quantum computations."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 ttl_seconds: float = 3600,
                 memory_limit_mb: float = 1000):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.memory_limit_mb = memory_limit_mb
        
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_used_mb': 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = (args, tuple(sorted(kwargs.items())))
        key_string = pickle.dumps(key_data)
        return hashlib.sha256(key_string).hexdigest()[:16]
    
    def _estimate_memory_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1000  # Default estimate
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove cache entry and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats['memory_used_mb'] -= entry.memory_size / 1024 / 1024
            self._stats['evictions'] += 1
    
    def _enforce_memory_limit(self):
        """Enforce memory limit by removing LRU entries."""
        while (self._stats['memory_used_mb'] > self.memory_limit_mb and 
               len(self._cache) > 0):
            # Remove least recently used entry
            lru_key = next(iter(self._cache))
            self._remove_entry(lru_key)
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get value from cache."""
        key = self._generate_key(*args, **kwargs)
        
        with self._lock:
            self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                entry.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._stats['hits'] += 1
                return entry.value
            
            self._stats['misses'] += 1
            return None
    
    def put(self, value: Any, computation_time: float, *args, **kwargs):
        """Put value in cache."""
        key = self._generate_key(*args, **kwargs)
        memory_size = self._estimate_memory_size(value)
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                computation_time=computation_time,
                memory_size=memory_size
            )
            
            self._cache[key] = entry
            self._stats['memory_used_mb'] += memory_size / 1024 / 1024
            
            # Enforce size and memory limits
            while len(self._cache) > self.max_size:
                lru_key = next(iter(self._cache))
                self._remove_entry(lru_key)
            
            self._enforce_memory_limit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total_requests)
            
            return {
                'hit_rate': hit_rate,
                'total_entries': len(self._cache),
                'memory_used_mb': self._stats['memory_used_mb'],
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions']
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'memory_used_mb': 0
            }


class QuantumPerformanceProfiler:
    """Profiler for quantum algorithm performance."""
    
    def __init__(self):
        self.profiles = []
        self.current_profile = None
        
    def start_profiling(self, algorithm_name: str):
        """Start profiling an algorithm."""
        self.current_profile = {
            'algorithm_name': algorithm_name,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'start_cpu': psutil.cpu_percent(),
            'quantum_operations': 0
        }
    
    def record_quantum_operation(self):
        """Record a quantum operation."""
        if self.current_profile:
            self.current_profile['quantum_operations'] += 1
    
    def end_profiling(self, cache_efficiency: float = 0.0) -> PerformanceProfile:
        """End profiling and return results."""
        if not self.current_profile:
            raise ValueError("No active profiling session")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - self.current_profile['start_time']
        memory_peak = max(end_memory, self.current_profile['start_memory'])
        cpu_utilization = (end_cpu + self.current_profile['start_cpu']) / 2
        
        profile = PerformanceProfile(
            algorithm_name=self.current_profile['algorithm_name'],
            execution_time=execution_time,
            memory_peak_mb=memory_peak,
            cpu_utilization=cpu_utilization,
            cache_efficiency=cache_efficiency,
            quantum_operations=self.current_profile['quantum_operations'],
            parallel_efficiency=min(1.0, cpu_utilization / 100.0)
        )
        
        self.profiles.append(profile)
        self.current_profile = None
        
        return profile
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance profiles."""
        if not self.profiles:
            return {}
        
        return {
            'total_algorithms': len(self.profiles),
            'avg_execution_time': np.mean([p.execution_time for p in self.profiles]),
            'avg_memory_peak': np.mean([p.memory_peak_mb for p in self.profiles]),
            'avg_cpu_utilization': np.mean([p.cpu_utilization for p in self.profiles]),
            'avg_cache_efficiency': np.mean([p.cache_efficiency for p in self.profiles]),
            'total_quantum_operations': sum([p.quantum_operations for p in self.profiles])
        }


class AdaptiveQuantumOptimizer:
    """Adaptive optimizer for quantum algorithms."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_config = {
            'n_qubits': 8,
            'coherence_threshold': 0.7,
            'entanglement_threshold': 0.5,
            'parallel_processes': 4,
            'cache_size': 1000
        }
        
    def optimize_parameters(self, 
                          performance_profile: PerformanceProfile,
                          target_execution_time: float = 10.0,
                          target_memory_mb: float = 500.0) -> Dict[str, Any]:
        """Optimize parameters based on performance profile."""
        new_config = self.current_config.copy()
        
        # Adjust based on execution time
        if performance_profile.execution_time > target_execution_time:
            # Too slow - reduce complexity
            new_config['n_qubits'] = max(4, new_config['n_qubits'] - 1)
            new_config['parallel_processes'] = min(8, new_config['parallel_processes'] + 1)
        elif performance_profile.execution_time < target_execution_time * 0.5:
            # Too fast - can increase complexity
            new_config['n_qubits'] = min(12, new_config['n_qubits'] + 1)
        
        # Adjust based on memory usage
        if performance_profile.memory_peak_mb > target_memory_mb:
            # Too much memory - reduce cache
            new_config['cache_size'] = max(100, int(new_config['cache_size'] * 0.8))
        elif performance_profile.memory_peak_mb < target_memory_mb * 0.5:
            # Low memory usage - can increase cache
            new_config['cache_size'] = min(5000, int(new_config['cache_size'] * 1.2))
        
        # Adjust based on cache efficiency
        if performance_profile.cache_efficiency < 0.3:
            # Poor cache performance - increase cache size
            new_config['cache_size'] = min(5000, int(new_config['cache_size'] * 1.5))
        
        # Store optimization decision
        self.optimization_history.append({
            'timestamp': time.time(),
            'old_config': self.current_config.copy(),
            'new_config': new_config.copy(),
            'performance_trigger': {
                'execution_time': performance_profile.execution_time,
                'memory_peak_mb': performance_profile.memory_peak_mb,
                'cache_efficiency': performance_profile.cache_efficiency
            }
        })
        
        self.current_config = new_config
        return new_config
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        
        if len(self.optimization_history) > 3:
            recent_history = self.optimization_history[-3:]
            
            # Check for oscillating parameters
            n_qubits_changes = [h['new_config']['n_qubits'] - h['old_config']['n_qubits'] 
                              for h in recent_history]
            if len(set(np.sign(n_qubits_changes))) > 1:
                recommendations.append("Consider fixing n_qubits to avoid oscillation")
            
            # Check for consistent memory pressure
            memory_increases = [h['new_config']['cache_size'] < h['old_config']['cache_size'] 
                              for h in recent_history]
            if all(memory_increases):
                recommendations.append("Consistent memory pressure detected - consider algorithmic optimization")
        
        return recommendations


class QuantumBenchmarkSuite:
    """Comprehensive benchmarking suite for quantum algorithms."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.cache = QuantumCache()
        self.profiler = QuantumPerformanceProfiler()
        self.optimizer = AdaptiveQuantumOptimizer()
        
        self.benchmark_results = []
    
    def benchmark_algorithm(self,
                          algorithm_factory: callable,
                          algorithm_name: str,
                          test_datasets: List[pd.DataFrame],
                          n_runs: int = 5) -> Dict[str, Any]:
        """Comprehensive benchmark of quantum algorithm."""
        algorithm_results = []
        
        for run in range(n_runs):
            for dataset_idx, dataset in enumerate(test_datasets):
                # Start profiling
                self.profiler.start_profiling(f"{algorithm_name}_run_{run}")
                
                # Create algorithm instance
                algorithm = algorithm_factory()
                
                # Run algorithm with caching
                start_time = time.time()
                
                try:
                    # Check cache first
                    cache_key = f"{algorithm_name}_{dataset_idx}_{run}"
                    cached_result = self.cache.get(cache_key, dataset.shape, dataset.columns.tolist())
                    
                    if cached_result is not None:
                        result = cached_result
                        cache_hit = True
                    else:
                        # Record quantum operations during execution
                        original_fit = algorithm.fit
                        original_predict = algorithm.predict
                        
                        def tracked_fit(*args, **kwargs):
                            self.profiler.record_quantum_operation()
                            return original_fit(*args, **kwargs)
                        
                        def tracked_predict(*args, **kwargs):
                            self.profiler.record_quantum_operation()
                            return original_predict(*args, **kwargs)
                        
                        algorithm.fit = tracked_fit
                        algorithm.predict = tracked_predict
                        
                        # Execute algorithm
                        algorithm.fit(dataset)
                        result = algorithm.predict(dataset)
                        
                        # Cache result
                        computation_time = time.time() - start_time
                        self.cache.put(result, computation_time, cache_key, 
                                     dataset.shape, dataset.columns.tolist())
                        cache_hit = False
                    
                    execution_time = time.time() - start_time
                    
                    # End profiling
                    cache_stats = self.cache.get_stats()
                    profile = self.profiler.end_profiling(cache_stats['hit_rate'])
                    
                    # Record results
                    algorithm_results.append({
                        'run': run,
                        'dataset_idx': dataset_idx,
                        'dataset_shape': dataset.shape,
                        'execution_time': execution_time,
                        'cache_hit': cache_hit,
                        'performance_profile': profile,
                        'result': result
                    })
                    
                    # Adaptive optimization
                    if run > 0 and run % 2 == 0:  # Optimize every 2 runs
                        new_config = self.optimizer.optimize_parameters(profile)
                        # Could apply new_config to algorithm_factory
                    
                except Exception as e:
                    algorithm_results.append({
                        'run': run,
                        'dataset_idx': dataset_idx,
                        'error': str(e),
                        'failed': True
                    })
        
        # Analyze results
        successful_results = [r for r in algorithm_results if not r.get('failed', False)]
        
        if successful_results:
            benchmark_summary = {
                'algorithm_name': algorithm_name,
                'total_runs': len(algorithm_results),
                'successful_runs': len(successful_results),
                'avg_execution_time': np.mean([r['execution_time'] for r in successful_results]),
                'std_execution_time': np.std([r['execution_time'] for r in successful_results]),
                'cache_hit_rate': np.mean([r['cache_hit'] for r in successful_results]),
                'avg_memory_peak': np.mean([r['performance_profile'].memory_peak_mb for r in successful_results]),
                'avg_quantum_operations': np.mean([r['performance_profile'].quantum_operations for r in successful_results]),
                'optimization_recommendations': self.optimizer.get_optimization_recommendations(),
                'detailed_results': algorithm_results
            }
        else:
            benchmark_summary = {
                'algorithm_name': algorithm_name,
                'total_runs': len(algorithm_results),
                'successful_runs': 0,
                'error': 'All runs failed'
            }
        
        self.benchmark_results.append(benchmark_summary)
        return benchmark_summary
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.benchmark_results:
            return "No benchmark results available"
        
        report = "QUANTUM ALGORITHM PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall statistics
        total_algorithms = len(self.benchmark_results)
        successful_algorithms = sum(1 for r in self.benchmark_results if r.get('successful_runs', 0) > 0)
        
        report += f"Total Algorithms Tested: {total_algorithms}\n"
        report += f"Successful Algorithms: {successful_algorithms}\n"
        report += f"Success Rate: {successful_algorithms/total_algorithms*100:.1f}%\n\n"
        
        # Cache performance
        cache_stats = self.cache.get_stats()
        report += "CACHE PERFORMANCE\n"
        report += "-" * 20 + "\n"
        report += f"Hit Rate: {cache_stats['hit_rate']*100:.1f}%\n"
        report += f"Total Entries: {cache_stats['total_entries']}\n"
        report += f"Memory Used: {cache_stats['memory_used_mb']:.1f} MB\n\n"
        
        # Individual algorithm results
        for result in self.benchmark_results:
            if result.get('successful_runs', 0) > 0:
                report += f"ALGORITHM: {result['algorithm_name']}\n"
                report += "-" * 30 + "\n"
                report += f"Success Rate: {result['successful_runs']}/{result['total_runs']}\n"
                report += f"Avg Execution Time: {result['avg_execution_time']:.3f}s ± {result['std_execution_time']:.3f}s\n"
                report += f"Cache Hit Rate: {result['cache_hit_rate']*100:.1f}%\n"
                report += f"Avg Memory Peak: {result['avg_memory_peak']:.1f} MB\n"
                report += f"Avg Quantum Ops: {result['avg_quantum_operations']:.0f}\n"
                
                if result['optimization_recommendations']:
                    report += "Recommendations:\n"
                    for rec in result['optimization_recommendations']:
                        report += f"  - {rec}\n"
                
                report += "\n"
        
        return report
    
    def export_results(self, format: str = 'json'):
        """Export benchmark results."""
        if format == 'json':
            import json
            output_file = self.output_dir / 'benchmark_results.json'
            
            # Prepare serializable data
            serializable_results = []
            for result in self.benchmark_results:
                serializable_result = result.copy()
                
                # Convert complex objects to serializable format
                if 'detailed_results' in serializable_result:
                    for detail in serializable_result['detailed_results']:
                        if 'performance_profile' in detail:
                            profile = detail['performance_profile']
                            detail['performance_profile'] = {
                                'algorithm_name': profile.algorithm_name,
                                'execution_time': profile.execution_time,
                                'memory_peak_mb': profile.memory_peak_mb,
                                'cpu_utilization': profile.cpu_utilization,
                                'cache_efficiency': profile.cache_efficiency,
                                'quantum_operations': profile.quantum_operations,
                                'parallel_efficiency': profile.parallel_efficiency
                            }
                        if 'result' in detail and hasattr(detail['result'], 'adjacency_matrix'):
                            detail['result'] = {
                                'adjacency_matrix': detail['result'].adjacency_matrix.tolist(),
                                'confidence_scores': detail['result'].confidence_scores.tolist(),
                                'method_used': detail['result'].method_used
                            }
                
                serializable_results.append(serializable_result)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        elif format == 'csv':
            output_file = self.output_dir / 'benchmark_summary.csv'
            
            # Create summary DataFrame
            summary_data = []
            for result in self.benchmark_results:
                if result.get('successful_runs', 0) > 0:
                    summary_data.append({
                        'algorithm_name': result['algorithm_name'],
                        'successful_runs': result['successful_runs'],
                        'total_runs': result['total_runs'],
                        'avg_execution_time': result['avg_execution_time'],
                        'std_execution_time': result['std_execution_time'],
                        'cache_hit_rate': result['cache_hit_rate'],
                        'avg_memory_peak': result['avg_memory_peak'],
                        'avg_quantum_operations': result['avg_quantum_operations']
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(output_file, index=False)


# Global quantum performance instance
_quantum_performance = None

def get_quantum_performance_manager() -> QuantumBenchmarkSuite:
    """Get global quantum performance manager."""
    global _quantum_performance
    if _quantum_performance is None:
        _quantum_performance = QuantumBenchmarkSuite()
    return _quantum_performance