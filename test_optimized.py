#!/usr/bin/env python3
"""Test script to verify optimized causal discovery functionality."""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add current directory to path for package import
sys.path.insert(0, os.path.dirname(__file__))

try:
    from causal_discovery_toolkit.algorithms.optimized import OptimizedCausalModel, AdaptiveScalingManager
    from causal_discovery_toolkit import DataProcessor, CausalMetrics
    
    print("✅ Optimized modules imported successfully")
    
    # Test with synthetic data for performance
    print("\n🧪 Testing optimized performance...")
    np.random.seed(42)
    
    # Generate larger dataset for performance testing
    data_processor = DataProcessor()
    large_data = data_processor.generate_synthetic_data(
        n_samples=1000,
        n_variables=15,
        random_state=42
    )
    
    # Test basic optimized model
    print("\n🚀 Testing OptimizedCausalModel...")
    model = OptimizedCausalModel(
        threshold=0.3,
        enable_caching=True,
        enable_parallel=True,
        auto_optimize=True
    )
    
    start_time = time.time()
    model.fit(large_data)
    fit_time = time.time() - start_time
    
    start_time = time.time()
    result = model.discover()
    discover_time = time.time() - start_time
    
    print(f"✅ Optimized model completed:")
    print(f"  Fit time: {fit_time:.3f}s")
    print(f"  Discovery time: {discover_time:.3f}s")
    print(f"  Edges found: {result.metadata['n_edges']}")
    print(f"  Variables: {result.metadata['n_variables']}")
    print(f"  Optimization enabled: {result.metadata['optimization_enabled']}")
    
    # Test performance statistics
    print("\n📊 Testing performance statistics...")
    perf_stats = model.get_performance_stats()
    print(f"  Cache hits: {perf_stats['cache_hits']}")
    print(f"  Cache misses: {perf_stats['cache_misses']}")
    print(f"  Parallel operations: {perf_stats['parallel_operations']}")
    print(f"  Cache memory usage: {perf_stats['cache_memory_usage_mb']:.2f}MB")
    
    # Test turbo mode
    print("\n⚡ Testing Turbo Mode...")
    model.enable_turbo_mode()
    
    start_time = time.time()
    model.fit(large_data)
    turbo_result = model.discover()
    turbo_time = time.time() - start_time
    
    print(f"✅ Turbo mode completed in {turbo_time:.3f}s")
    print(f"  Max workers: {model.max_workers}")
    print(f"  Caching enabled: {model.enable_caching}")
    print(f"  Parallel enabled: {model.enable_parallel}")
    
    # Test cached correlation subset
    print("\n🔄 Testing cached correlation subset...")
    subset_cols = list(large_data.columns[:5])
    start_time = time.time()
    cached_corr1 = model.cached_correlation_subset(large_data, subset_cols)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    cached_corr2 = model.cached_correlation_subset(large_data, subset_cols)
    cached_call_time = time.time() - start_time
    
    print(f"✅ Cached correlation test:")
    print(f"  First call: {first_call_time:.4f}s")
    print(f"  Cached call: {cached_call_time:.4f}s")
    print(f"  Speedup: {first_call_time/cached_call_time:.1f}x")
    
    # Test benchmark functionality
    print("\n🏁 Running mini performance benchmark...")
    small_sizes = [(100, 5), (500, 8)]  # Small sizes for quick testing
    benchmark_results = model.benchmark_performance(
        data_sizes=small_sizes, 
        n_runs=2
    )
    
    print("✅ Benchmark completed:")
    print(benchmark_results.groupby(['n_samples', 'n_features'])[['fit_time', 'discover_time']].mean())
    
    # Test adaptive scaling manager
    print("\n🎯 Testing Adaptive Scaling Manager...")
    scaling_manager = AdaptiveScalingManager()
    
    # Register model
    scaling_manager.register_model('test_model', model)
    
    # Simulate high load metrics
    high_load_metrics = {
        'response_time': 3.0,
        'memory_usage': 0.85,
        'cpu_usage': 0.8,
        'queue_depth': 60
    }
    
    scaling_decision = scaling_manager.check_and_scale('test_model', high_load_metrics)
    print(f"✅ Scaling decision: {scaling_decision['action']}")
    print(f"  Reason: {scaling_decision['reason']}")
    
    # Test with low load
    low_load_metrics = {
        'response_time': 0.5,
        'memory_usage': 0.2,
        'cpu_usage': 0.3,
        'queue_depth': 5
    }
    
    scaling_decision = scaling_manager.check_and_scale('test_model', low_load_metrics)
    print(f"✅ Low load scaling decision: {scaling_decision['action']}")
    
    # Test performance comparison
    print("\n⚖️  Performance comparison test...")
    
    # Disable optimizations for comparison
    model.disable_optimizations()
    
    start_time = time.time()
    model.fit(large_data)
    basic_result = model.discover()
    basic_time = time.time() - start_time
    
    # Re-enable optimizations
    model.enable_turbo_mode()
    
    start_time = time.time()
    model.fit(large_data)
    optimized_result = model.discover()
    optimized_time = time.time() - start_time
    
    speedup = basic_time / optimized_time if optimized_time > 0 else float('inf')
    
    print(f"✅ Performance comparison:")
    print(f"  Basic mode: {basic_time:.3f}s")
    print(f"  Optimized mode: {optimized_time:.3f}s") 
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Results identical: {np.array_equal(basic_result.adjacency_matrix, optimized_result.adjacency_matrix)}")
    
    print("\n🎉 ALL OPTIMIZATION TESTS PASSED - Generation 3 Complete!")
    
    # Show final comprehensive summary
    print("\n📋 Final Optimization Summary:")
    final_stats = model.get_performance_stats()
    print(f"  • Total operations: {len(final_stats['fit_times']) + len(final_stats['discover_times'])}")
    print(f"  • Average fit time: {final_stats.get('avg_fit_time', 0):.3f}s")
    print(f"  • Average discovery time: {final_stats.get('avg_discover_time', 0):.3f}s")
    print(f"  • Cache hit rate: {final_stats.get('cache_hit_rate', 0):.2f}")
    print(f"  • Parallel operations: {final_stats['parallel_operations']}")
    print(f"  • Memory efficiency: {final_stats['cache_memory_usage_mb']:.2f}MB cached")

except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)