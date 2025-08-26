#!/usr/bin/env python3
"""
Autonomous SDLC - Scalable Demo (Generation 3)
Enhanced with performance optimization, concurrent processing, and auto-scaling
"""

import sys
import os
import logging
import time
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup performance-optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scalable_demo.log')
    ]
)
logger = logging.getLogger(__name__)


class PerformanceCache:
    """High-performance caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with TTL and LRU tracking"""
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        current_time = time.time()
        
        # Check TTL
        if current_time - timestamp > self.ttl_seconds:
            self._evict(key)
            return None
        
        # Update access time for LRU
        self.access_times[key] = current_time
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Store value with automatic LRU eviction if needed"""
        current_time = time.time()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = (value, current_time)
        self.access_times[key] = current_time
    
    def _evict(self, key: str) -> None:
        """Remove specific key from cache"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._evict(lru_key)
    
    def clear_expired(self) -> int:
        """Clear expired items, return count of cleared items"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._evict(key)
        
        return len(expired_keys)


class ResourceManager:
    """Advanced resource management with auto-scaling and optimization"""
    
    def __init__(self):
        self.cpu_count = cpu_count()
        self.memory_monitor = {}
        self.performance_metrics = {}
        self._last_gc_time = time.time()
        
    def get_optimal_worker_count(self, task_type: str = "cpu_bound") -> int:
        """Determine optimal worker count based on task type and system load"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if task_type == "cpu_bound":
                # For CPU-bound tasks, use fewer workers if system is under load
                if cpu_percent > 80:
                    return max(1, self.cpu_count // 2)
                elif cpu_percent > 50:
                    return max(2, int(self.cpu_count * 0.75))
                else:
                    return self.cpu_count
                    
            elif task_type == "io_bound":
                # For I/O-bound tasks, can use more workers
                if memory_percent > 85:
                    return self.cpu_count
                else:
                    return min(self.cpu_count * 2, 16)
            
            return self.cpu_count
            
        except ImportError:
            return self.cpu_count
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage with garbage collection and monitoring"""
        import gc
        
        current_time = time.time()
        if current_time - self._last_gc_time > 60:  # GC every minute
            collected = gc.collect()
            self._last_gc_time = current_time
            
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                
                optimization_results = {
                    "garbage_collected": collected,
                    "memory_rss_mb": memory_info.rss / (1024 * 1024),
                    "memory_vms_mb": memory_info.vms / (1024 * 1024),
                    "optimization_time": current_time
                }
                
                self.memory_monitor[current_time] = optimization_results
                return optimization_results
                
            except ImportError:
                return {"garbage_collected": collected, "psutil_unavailable": True}
        
        return {"status": "no_optimization_needed"}
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """Provide resource optimization recommendations"""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            recommendations = []
            
            if cpu_percent > 90:
                recommendations.append("Consider reducing concurrent tasks - high CPU usage")
            elif cpu_percent < 20:
                recommendations.append("System underutilized - can increase parallelism")
            
            if memory.percent > 85:
                recommendations.append("High memory usage - enable aggressive caching cleanup")
            elif memory.percent < 30:
                recommendations.append("Low memory usage - can increase cache size")
            
            if disk.percent > 90:
                recommendations.append("Low disk space - enable result compression")
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "recommendations": recommendations,
                "optimal_workers": self.get_optimal_worker_count()
            }
            
        except ImportError:
            return {"status": "monitoring_unavailable", "default_workers": self.cpu_count}


class ScalableCausalDiscovery:
    """Scalable causal discovery with performance optimization and concurrent processing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.cache = PerformanceCache(
            max_size=self.config["cache_size"],
            ttl_seconds=self.config["cache_ttl"]
        )
        self.resource_manager = ResourceManager()
        self.performance_metrics = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default scalable configuration"""
        return {
            "cache_size": 1000,
            "cache_ttl": 3600,
            "enable_parallel": True,
            "enable_caching": True,
            "enable_optimization": True,
            "batch_size": 1000,
            "max_concurrent_tasks": cpu_count(),
            "memory_limit_mb": 4096,
            "enable_compression": True,
            "auto_scaling": True
        }
    
    def _cache_key(self, data_shape: Tuple[int, int], algorithm: str, params: Dict) -> str:
        """Generate cache key for results"""
        import hashlib
        key_data = f"{data_shape}_{algorithm}_{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _async_data_processing(self, data_chunks: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Asynchronous data processing for large datasets"""
        
        def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            """Process individual data chunk"""
            # Simulate processing
            processed = chunk.copy()
            
            # Handle missing values
            numeric_cols = processed.select_dtypes(include=[np.number]).columns
            processed[numeric_cols] = processed[numeric_cols].fillna(processed[numeric_cols].mean())
            
            # Normalize
            processed[numeric_cols] = (processed[numeric_cols] - processed[numeric_cols].mean()) / processed[numeric_cols].std()
            
            return processed
        
        # Process chunks concurrently
        loop = asyncio.get_event_loop()
        max_workers = self.resource_manager.get_optimal_worker_count("cpu_bound")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, process_chunk, chunk)
                for chunk in data_chunks
            ]
            
            processed_chunks = await asyncio.gather(*tasks)
        
        return processed_chunks
    
    def _parallel_causal_discovery(self, data_chunks: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Run causal discovery on multiple data chunks in parallel"""
        
        # Process chunks in parallel using ThreadPoolExecutor instead of ProcessPoolExecutor
        # to avoid pickle issues with local functions
        max_workers = self.resource_manager.get_optimal_worker_count("cpu_bound")
        
        def discover_chunk_wrapper(chunk_info):
            return self._discover_single_chunk(chunk_info)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_data = list(enumerate(data_chunks))
            results = list(executor.map(discover_chunk_wrapper, chunk_data))
        
        return results
    
    def _discover_single_chunk(self, chunk_data: Tuple[int, pd.DataFrame]) -> Dict[str, Any]:
        """Discover causal relationships for a single chunk - separate method to avoid pickle issues"""
        chunk_id, chunk = chunk_data
        
        try:
            from algorithms.base import SimpleLinearCausalModel
            
            # Check cache first
            cache_key = self._cache_key(chunk.shape, "SimpleLinear", {"threshold": 0.3})
            cached_result = self.cache.get(cache_key) if self.config["enable_caching"] else None
            
            if cached_result:
                logger.info(f"Cache hit for chunk {chunk_id}")
                return {"chunk_id": chunk_id, "result": cached_result, "cached": True}
            
            # Run discovery
            model = SimpleLinearCausalModel(threshold=0.3)
            result = model.fit_discover(chunk)
            
            # Convert result to serializable format
            serializable_result = {
                "method": result.method_used,
                "edges_found": int(result.adjacency_matrix.sum()),
                "adjacency_matrix": result.adjacency_matrix.tolist(),
                "confidence_scores": result.confidence_scores.tolist(),
                "metadata": result.metadata
            }
            
            # Cache result
            if self.config["enable_caching"]:
                self.cache.put(cache_key, serializable_result)
            
            return {"chunk_id": chunk_id, "result": serializable_result, "cached": False}
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} processing failed: {e}")
            return {"chunk_id": chunk_id, "error": str(e)}
    
    def _optimize_performance(self, data_size: Tuple[int, int]) -> Dict[str, Any]:
        """Apply performance optimizations based on data characteristics"""
        
        optimization_results = {
            "memory_optimization": self.resource_manager.optimize_memory_usage(),
            "cache_cleanup": self.cache.clear_expired(),
            "resource_recommendations": self.resource_manager.get_resource_recommendations()
        }
        
        # Auto-scale configuration based on data size
        if self.config["auto_scaling"]:
            n_samples, n_features = data_size
            
            if n_samples > 10000:
                # Large dataset optimizations
                self.config["batch_size"] = min(2000, n_samples // 10)
                self.config["max_concurrent_tasks"] = min(cpu_count() * 2, 16)
                logger.info("Applied large dataset optimizations")
                
            elif n_samples < 500:
                # Small dataset optimizations
                self.config["batch_size"] = n_samples
                self.config["max_concurrent_tasks"] = 2
                logger.info("Applied small dataset optimizations")
        
        return optimization_results
    
    def scalable_causal_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute scalable causal discovery with performance optimization"""
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting scalable causal discovery on data: {data.shape}")
            
            # Performance optimization
            optimization_results = self._optimize_performance(data.shape)
            
            # Split data into chunks for parallel processing
            n_samples = len(data)
            batch_size = min(self.config["batch_size"], n_samples)
            
            if n_samples > batch_size and self.config["enable_parallel"]:
                # Large dataset: chunk and process in parallel
                data_chunks = [
                    data.iloc[i:i+batch_size].copy()
                    for i in range(0, n_samples, batch_size)
                ]
                
                logger.info(f"Processing {len(data_chunks)} chunks in parallel")
                
                # Async preprocessing if needed
                if len(data_chunks) > 1:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    processed_chunks = loop.run_until_complete(
                        self._async_data_processing(data_chunks)
                    )
                    loop.close()
                else:
                    processed_chunks = data_chunks
                
                # Parallel causal discovery
                chunk_results = self._parallel_causal_discovery(processed_chunks)
                
                # Aggregate results
                total_edges = sum(
                    result.get("result", {}).get("edges_found", 0) 
                    for result in chunk_results 
                    if "result" in result
                )
                
                successful_chunks = [r for r in chunk_results if "result" in r]
                failed_chunks = [r for r in chunk_results if "error" in r]
                cached_chunks = [r for r in successful_chunks if r.get("cached", False)]
                
                aggregated_result = {
                    "method": "ScalableParallelDiscovery",
                    "total_chunks": len(data_chunks),
                    "successful_chunks": len(successful_chunks),
                    "failed_chunks": len(failed_chunks),
                    "cached_chunks": len(cached_chunks),
                    "total_edges_found": total_edges,
                    "chunk_results": chunk_results,
                    "processing_mode": "parallel_chunked"
                }
                
            else:
                # Small dataset: process directly
                from algorithms.base import SimpleLinearCausalModel
                
                model = SimpleLinearCausalModel(threshold=0.3)
                result = model.fit_discover(data)
                
                aggregated_result = {
                    "method": result.method_used,
                    "edges_found": int(result.adjacency_matrix.sum()),
                    "adjacency_matrix": result.adjacency_matrix.tolist(),
                    "confidence_scores": result.confidence_scores.tolist(),
                    "metadata": result.metadata,
                    "processing_mode": "single_threaded"
                }
            
            execution_time = time.time() - start_time
            
            # Compile comprehensive results
            scalable_results = {
                "causal_result": aggregated_result,
                "performance_metrics": {
                    "execution_time": execution_time,
                    "data_shape": data.shape,
                    "processing_mode": aggregated_result.get("processing_mode"),
                    "cache_hit_rate": len([r for r in chunk_results if r.get("cached")]) / max(len(chunk_results), 1) if "chunk_results" in aggregated_result else 0,
                    "parallel_efficiency": min(len(chunk_results) / max(execution_time, 1), 100) if "chunk_results" in aggregated_result else 1
                },
                "optimization_results": optimization_results,
                "resource_usage": self.resource_manager.get_resource_recommendations(),
                "cache_stats": {
                    "cache_size": len(self.cache.cache),
                    "cache_capacity": self.cache.max_size,
                    "expired_cleared": optimization_results["cache_cleanup"]
                },
                "status": "success"
            }
            
            logger.info(f"Scalable causal discovery completed in {execution_time:.2f}s")
            return scalable_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Scalable causal discovery failed: {e}")
            
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
                "data_shape": data.shape,
                "optimization_results": optimization_results if 'optimization_results' in locals() else {},
                "timestamp": time.time()
            }


def main():
    """Main scalable demo execution"""
    print("⚡ Autonomous SDLC - Scalable Demo (Generation 3)")
    print("=" * 60)
    
    try:
        # Initialize scalable discovery system
        scalable_config = {
            "cache_size": 2000,
            "cache_ttl": 1800,
            "enable_parallel": True,
            "enable_caching": True,
            "enable_optimization": True,
            "batch_size": 500,
            "max_concurrent_tasks": cpu_count(),
            "auto_scaling": True
        }
        
        discovery_system = ScalableCausalDiscovery(scalable_config)
        
        print("✅ Scalable discovery system initialized")
        print(f"   CPU cores: {cpu_count()}")
        print(f"   Cache size: {scalable_config['cache_size']}")
        print(f"   Batch size: {scalable_config['batch_size']}")
        
        # Generate larger test dataset for scalability demonstration
        from utils.data_processing import DataProcessor
        processor = DataProcessor()
        
        # Generate moderately large dataset
        data = processor.generate_synthetic_data(
            n_samples=2500,  # Larger dataset
            n_variables=12,   # More variables
            noise_level=0.25,
            random_state=42
        )
        
        print(f"✅ Generated large test dataset: {data.shape}")
        print(f"   Memory usage: ~{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Execute scalable causal discovery
        print("\n⚡ Executing scalable causal discovery...")
        start_time = time.time()
        results = discovery_system.scalable_causal_discovery(data)
        total_time = time.time() - start_time
        
        # Display comprehensive results
        if results["status"] == "success":
            print("\n🎉 Scalable causal discovery completed successfully!")
            
            causal_results = results["causal_result"]
            perf_metrics = results["performance_metrics"]
            
            print(f"   Processing mode: {causal_results.get('processing_mode', 'unknown')}")
            
            if causal_results.get("processing_mode") == "parallel_chunked":
                print(f"   Total chunks: {causal_results['total_chunks']}")
                print(f"   Successful: {causal_results['successful_chunks']}")
                print(f"   Failed: {causal_results['failed_chunks']}")
                print(f"   Cached: {causal_results['cached_chunks']}")
                print(f"   Total edges: {causal_results['total_edges_found']}")
            else:
                print(f"   Edges found: {causal_results['edges_found']}")
            
            print(f"\n📊 Performance Metrics:")
            print(f"   Execution time: {perf_metrics['execution_time']:.2f}s")
            print(f"   Cache hit rate: {perf_metrics['cache_hit_rate']:.1%}")
            print(f"   Parallel efficiency: {perf_metrics['parallel_efficiency']:.1f}")
            
            print(f"\n🔧 Resource Usage:")
            resource_usage = results["resource_usage"]
            print(f"   CPU usage: {resource_usage.get('cpu_usage', 'N/A'):.1f}%")
            print(f"   Memory usage: {resource_usage.get('memory_usage', 'N/A'):.1f}%")
            print(f"   Optimal workers: {resource_usage.get('optimal_workers', 'N/A')}")
            
            if resource_usage.get("recommendations"):
                print(f"\n💡 System Recommendations:")
                for rec in resource_usage["recommendations"][:3]:
                    print(f"   • {rec}")
            
            cache_stats = results["cache_stats"]
            print(f"\n📦 Cache Statistics:")
            print(f"   Cache utilization: {cache_stats['cache_size']}/{cache_stats['cache_capacity']}")
            print(f"   Expired entries cleared: {cache_stats['expired_cleared']}")
            
            # Save detailed results
            results_file = Path("scalable_discovery_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {results_file}")
            
            return True
            
        else:
            print(f"\n❌ Scalable causal discovery failed:")
            print(f"   Error: {results['error']}")
            print(f"   Execution time: {results['execution_time']:.2f}s")
            return False
            
    except Exception as e:
        logger.error(f"Main scalable demo execution failed: {e}")
        print(f"\n❌ Demo failed with unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)