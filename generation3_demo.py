#!/usr/bin/env python3
"""Generation 3 performance optimization demonstration."""

import pandas as pd
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import performance components
from utils.data_processing import DataProcessor
from utils.performance import AdaptiveCache, ConcurrentProcessor, PerformanceOptimizer
from algorithms.optimized import OptimizedCausalModel, AdaptiveScalingManager
from utils.monitoring import PerformanceMonitor
from utils.logging_config import get_logger

logger = get_logger("generation3_demo")


def main():
    """Demonstrate Generation 3 performance optimization features."""
    print("🚀 Generation 3: PERFORMANCE & SCALING DEMO")
    print("=" * 50)
    
    # 1. ADAPTIVE CACHING
    print("\n1. 📦 ADAPTIVE CACHING")
    cache = AdaptiveCache(max_size=50, max_memory_mb=100, default_ttl=60)
    
    # Simulate cache operations
    data_processor = DataProcessor()
    test_data = data_processor.generate_synthetic_data(n_samples=500, n_variables=5)
    
    # Cache some computation results
    print("Caching computation results...")
    cache.put("correlation_matrix", test_data.corr())
    cache.put("data_stats", test_data.describe())
    cache.put("sample_data", test_data.head(100))
    
    # Test cache hits and misses
    start_time = time.time()
    cached_corr = cache.get("correlation_matrix")
    cache_time = time.time() - start_time
    
    start_time = time.time()
    fresh_corr = test_data.corr()
    compute_time = time.time() - start_time
    
    print(f"Cache retrieval: {cache_time*1000:.2f}ms")
    print(f"Fresh computation: {compute_time*1000:.2f}ms")
    print(f"Speedup: {compute_time/cache_time:.1f}x")
    
    # Display cache statistics
    cache_stats = cache.get_stats()
    print(f"Cache stats: {cache_stats['hit_rate']:.1%} hit rate, "
          f"{cache_stats['size']}/{cache_stats['max_size']} entries, "
          f"{cache_stats['memory_usage_bytes']/1024/1024:.1f}MB")
    
    # 2. CONCURRENT PROCESSING
    print("\n2. ⚡ CONCURRENT PROCESSING")
    concurrent_processor = ConcurrentProcessor(max_workers=4)
    
    # Generate larger dataset for meaningful parallel processing
    large_data = data_processor.generate_synthetic_data(n_samples=2000, n_variables=12)
    
    # Compare serial vs parallel correlation computation
    print("Computing correlation matrices...")
    
    # Serial computation
    start_time = time.time()
    serial_corr = large_data.corr()
    serial_time = time.time() - start_time
    
    # Parallel computation
    start_time = time.time()
    parallel_corr = concurrent_processor.parallel_correlation_matrix(large_data)
    parallel_time = time.time() - start_time
    
    print(f"Serial computation: {serial_time:.3f}s")
    print(f"Parallel computation: {parallel_time:.3f}s")
    print(f"Speedup: {serial_time/parallel_time:.1f}x")
    print(f"Results match: {np.allclose(serial_corr.values, parallel_corr.values, rtol=1e-10)}")
    
    # 3. OPTIMIZED CAUSAL MODEL
    print("\n3. 🔧 OPTIMIZED CAUSAL DISCOVERY")
    
    # Create different model configurations for comparison
    models = {
        "Basic": OptimizedCausalModel(
            enable_caching=False,
            enable_parallel=False,
            threshold=0.3
        ),
        "Cached": OptimizedCausalModel(
            enable_caching=True,
            enable_parallel=False,
            threshold=0.3
        ),
        "Parallel": OptimizedCausalModel(
            enable_caching=False,
            enable_parallel=True,
            threshold=0.3,
            max_workers=4
        ),
        "Turbo": OptimizedCausalModel(
            enable_caching=True,
            enable_parallel=True,
            threshold=0.3,
            max_workers=4
        )
    }
    
    # Benchmark each configuration
    benchmark_data = data_processor.generate_synthetic_data(n_samples=1500, n_variables=10)
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name} model...")
        
        start_time = time.time()
        model.fit(benchmark_data)
        fit_time = time.time() - start_time
        
        start_time = time.time()
        discovery_result = model.discover()
        discover_time = time.time() - start_time
        
        total_time = fit_time + discover_time
        results[name] = {
            'fit_time': fit_time,
            'discover_time': discover_time,
            'total_time': total_time,
            'n_edges': discovery_result.metadata['n_edges']
        }
        
        # Get model performance stats
        perf_stats = model.get_performance_stats()
        cache_hit_rate = perf_stats.get('cache_hit_rate', 0)
        
        print(f"  Fit: {fit_time:.3f}s, Discover: {discover_time:.3f}s, Total: {total_time:.3f}s")
        print(f"  Edges found: {discovery_result.metadata['n_edges']}, Cache hit rate: {cache_hit_rate:.1%}")
        
        if name == "Turbo":
            print(f"  🔥 Turbo mode: {perf_stats.get('parallel_operations', 0)} parallel operations")
    
    # Show performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    baseline_time = results['Basic']['total_time']
    for name, stats in results.items():
        speedup = baseline_time / stats['total_time']
        print(f"  {name:8s}: {stats['total_time']:.3f}s ({speedup:.1f}x speedup)")
    
    # 4. PERFORMANCE MONITORING
    print("\n4. 📈 PERFORMANCE MONITORING")
    monitor = PerformanceMonitor(enable_detailed_monitoring=True)
    
    # Monitor a complex operation
    metrics = monitor.start_monitoring("large_scale_discovery")
    
    # Simulate large-scale operation
    very_large_data = data_processor.generate_synthetic_data(n_samples=3000, n_variables=15)
    turbo_model = models["Turbo"]
    turbo_model.fit(very_large_data)
    final_result = turbo_model.discover()
    
    monitor.add_custom_metric("data_size_mb", very_large_data.memory_usage(deep=True).sum() / 1024 / 1024)
    monitor.add_custom_metric("edges_discovered", final_result.metadata['n_edges'])
    
    final_metrics = monitor.stop_monitoring()
    
    print(f"Large-scale operation metrics:")
    print(f"  Duration: {final_metrics.duration:.3f}s")
    print(f"  Memory delta: {final_metrics.memory_usage_end - final_metrics.memory_usage_start:.1f}MB")
    print(f"  Data processed: {final_metrics.custom_metrics['data_size_mb']:.1f}MB")
    print(f"  Edges discovered: {final_metrics.custom_metrics['edges_discovered']}")
    
    # 5. AUTO-SCALING SIMULATION
    print("\n5. 🔄 ADAPTIVE SCALING")
    scaling_manager = AdaptiveScalingManager()
    
    # Register model for scaling
    scaling_manager.register_model("main_model", turbo_model)
    
    # Simulate different load conditions
    scenarios = [
        {"name": "Low Load", "response_time": 0.5, "memory_usage": 0.3, "cpu_usage": 0.2, "queue_depth": 5},
        {"name": "Normal Load", "response_time": 1.2, "memory_usage": 0.5, "cpu_usage": 0.4, "queue_depth": 15},
        {"name": "High Load", "response_time": 3.5, "memory_usage": 0.85, "cpu_usage": 0.8, "queue_depth": 75},
        {"name": "Overload", "response_time": 8.0, "memory_usage": 0.95, "cpu_usage": 0.9, "queue_depth": 150}
    ]
    
    for scenario in scenarios:
        metrics = {k: v for k, v in scenario.items() if k != "name"}
        scaling_decision = scaling_manager.check_and_scale("main_model", metrics)
        
        print(f"  {scenario['name']:12s}: {scaling_decision['action']:10s} - {scaling_decision['reason']}")
    
    # 6. BENCHMARK SUITE
    print("\n6. 🏁 COMPREHENSIVE BENCHMARK")
    
    # Create fresh turbo model for benchmark
    benchmark_model = OptimizedCausalModel(
        enable_caching=True,
        enable_parallel=True,
        auto_optimize=True
    )
    benchmark_model.enable_turbo_mode()
    
    # Run benchmark across different data sizes
    data_sizes = [(500, 8), (1000, 10), (1500, 12), (2000, 15)]
    benchmark_results = benchmark_model.benchmark_performance(data_sizes, n_runs=2)
    
    print("Benchmark Results:")
    summary = benchmark_results.groupby(['n_samples', 'n_features']).agg({
        'total_time': ['mean', 'std'],
        'n_edges': 'mean',
        'memory_mb': 'mean'
    }).round(3)
    
    print(summary)
    
    # 7. CACHE PERFORMANCE ANALYSIS
    print("\n7. 💾 CACHE PERFORMANCE ANALYSIS")
    
    # Test cache with different access patterns
    cache_test = AdaptiveCache(max_size=20, default_ttl=30)
    
    # Sequential access pattern
    print("Testing sequential access pattern...")
    for i in range(25):  # More items than cache size
        cache_test.put(f"item_{i}", f"data_{i}")
    
    # Mixed access pattern
    hit_count = 0
    total_requests = 50
    for i in range(total_requests):
        key = f"item_{i % 15}"  # Access recent items more often
        if cache_test.get(key) is not None:
            hit_count += 1
        else:
            cache_test.put(key, f"data_{key}")
    
    final_cache_stats = cache_test.get_stats()
    print(f"Final cache performance:")
    print(f"  Hit rate: {final_cache_stats['hit_rate']:.1%}")
    print(f"  Total evictions: {final_cache_stats['total_evictions']}")
    print(f"  Memory pressure evictions: {final_cache_stats['memory_pressure_evictions']}")
    
    # 8. MEMORY AND CPU OPTIMIZATION
    print("\n8. 🧠 MEMORY & CPU OPTIMIZATION")
    
    # Compare memory usage patterns
    memory_efficient_model = OptimizedCausalModel(
        enable_caching=True,
        cache_ttl=60,  # Shorter TTL for memory efficiency
        max_workers=2  # Fewer workers to reduce memory overhead
    )
    
    # Process same data with different configurations
    test_data_large = data_processor.generate_synthetic_data(n_samples=2500, n_variables=20)
    
    print(f"Processing large dataset: {test_data_large.shape}")
    print(f"Dataset size: {test_data_large.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
    
    # Memory-efficient processing
    start_memory = memory_efficient_model.get_performance_stats().get('cache_memory_usage_mb', 0)
    memory_efficient_model.fit(test_data_large)
    result_efficient = memory_efficient_model.discover()
    end_memory = memory_efficient_model.get_performance_stats().get('cache_memory_usage_mb', 0)
    
    print(f"Memory-efficient processing:")
    print(f"  Cache memory usage: {end_memory - start_memory:.1f}MB")
    print(f"  Processing completed successfully")
    print(f"  Edges discovered: {result_efficient.metadata['n_edges']}")
    
    print("\n🎉 GENERATION 3 PERFORMANCE DEMO COMPLETED!")
    print("\n🚀 Features Demonstrated:")
    print("  ✅ Adaptive caching with LRU eviction")
    print("  ✅ Concurrent processing with threading")
    print("  ✅ Performance monitoring and metrics")
    print("  ✅ Auto-scaling based on load conditions")
    print("  ✅ Comprehensive benchmarking suite")
    print("  ✅ Memory and CPU optimization")
    print("  ✅ Turbo mode for maximum performance")
    print("  ✅ Cache performance analysis")
    
    print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
    print(f"  Best speedup achieved: {max(baseline_time / r['total_time'] for r in results.values()):.1f}x")
    print(f"  Cache efficiency: {final_cache_stats['hit_rate']:.1%} hit rate")
    print(f"  Largest dataset processed: {test_data_large.shape}")
    print(f"  Parallel operations utilized: Multi-threaded correlation computation")


if __name__ == "__main__":
    main()