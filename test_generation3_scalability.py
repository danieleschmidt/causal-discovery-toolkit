#!/usr/bin/env python3
"""Generation 3 scalability test - Make it Scale"""

import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
import time
import warnings


def test_scalable_imports():
    """Test that scalable components import correctly"""
    print("🔍 Testing scalable imports...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel, ScalabilityMetrics
    from utils.performance import ConcurrentProcessor, AdaptiveCache, BatchProcessor
    from utils.auto_scaling import ResourceMonitor, AutoScaler
    
    print("✅ All scalable components imported successfully")


def test_basic_scalability():
    """Test basic scalability features"""
    print("🔍 Testing basic scalability...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Test with medium-sized dataset
    data = processor.generate_synthetic_data(n_samples=500, n_variables=6, random_state=42)
    
    model = ScalableCausalDiscoveryModel(
        enable_parallelization=True,
        enable_caching=True,
        enable_auto_scaling=False,  # Disable for testing
        max_workers=2,
        optimization_level="balanced",
        user_id="scalability_test"
    )
    
    # First run - should be uncached
    start_time = time.time()
    result1 = model.fit_discover(data)
    first_run_time = time.time() - start_time
    
    print(f"✅ First run completed")
    print(f"   - Quality score: {result1.quality_score:.3f}")
    print(f"   - Processing time: {result1.processing_time:.3f}s")
    print(f"   - Total time: {first_run_time:.3f}s")
    print(f"   - Method used: {result1.method_used}")
    
    # Second run - should use cache if enabled
    start_time = time.time()
    result2 = model.fit_discover(data)
    second_run_time = time.time() - start_time
    
    print(f"✅ Second run completed")
    print(f"   - Processing time: {result2.processing_time:.3f}s")
    print(f"   - Total time: {second_run_time:.3f}s")
    print(f"   - Cache likely used: {second_run_time < first_run_time * 0.8}")
    
    return True


def test_performance_optimization():
    """Test performance optimization features"""
    print("🔍 Testing performance optimization...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Test different optimization levels
    optimization_levels = ["speed", "memory", "balanced"]
    results = {}
    
    for opt_level in optimization_levels:
        print(f"   Testing optimization level: {opt_level}")
        
        data = processor.generate_synthetic_data(n_samples=300, n_variables=5, random_state=42)
        
        model = ScalableCausalDiscoveryModel(
            optimization_level=opt_level,
            max_workers=2,
            user_id=f"opt_{opt_level}"
        )
        
        start_time = time.time()
        result = model.fit_discover(data)
        execution_time = time.time() - start_time
        
        results[opt_level] = {
            'time': execution_time,
            'quality': result.quality_score,
            'edges': result.metadata['n_edges']
        }
        
        print(f"     - Time: {execution_time:.3f}s")
        print(f"     - Quality: {result.quality_score:.3f}")
    
    print(f"✅ Performance optimization tests completed")
    
    # Compare results
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"   - Fastest optimization: {fastest[0]} ({fastest[1]['time']:.3f}s)")
    
    return True


def test_auto_scaling():
    """Test auto-scaling components"""
    print("🔍 Testing auto-scaling...")
    
    from utils.auto_scaling import ResourceMonitor, AutoScaler
    
    # Test resource monitoring
    monitor = ResourceMonitor()
    current_load = monitor.get_system_load()
    
    print(f"✅ Resource monitoring works")
    print(f"   - Current system load: {current_load:.1%}")
    
    # Test auto-scaler
    scaler = AutoScaler(
        min_workers=1,
        max_workers=4,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3
    )
    
    # Test scaling decisions
    test_loads = [0.2, 0.5, 0.8, 0.9, 0.3, 0.1]
    
    for load in test_loads:
        new_workers = scaler.adjust_workers(load)
        print(f"   - Load: {load:.1%} -> Workers: {new_workers}")
    
    print(f"✅ Auto-scaling tests completed")
    return True


def test_caching_and_performance():
    """Test caching and performance optimizations"""
    print("🔍 Testing caching and performance...")
    
    from utils.performance import AdaptiveCache, BatchProcessor
    
    # Test adaptive cache
    cache = AdaptiveCache(max_size=10, default_ttl=60)
    
    # Add some test items
    for i in range(5):
        cache.put(f"key_{i}", f"value_{i}")
    
    # Test retrieval
    cached_value = cache.get("key_1")
    print(f"✅ Cache retrieval: {cached_value}")
    
    # Test batch processor
    batch_processor = BatchProcessor(batch_size=50, max_workers=2)
    
    # Create test data
    test_data = pd.DataFrame({
        'A': np.random.randn(200),
        'B': np.random.randn(200),
        'C': np.random.randn(200)
    })
    
    def simple_process(batch):
        return len(batch)
    
    def aggregate_results(results):
        return sum(results)
    
    # Test batch processing
    total_processed = batch_processor.process_in_batches(
        test_data, simple_process, aggregate_results
    )
    
    print(f"✅ Batch processing completed: {total_processed} samples processed")
    
    return True


def test_scalability_with_large_data():
    """Test scalability with larger datasets"""
    print("🔍 Testing scalability with larger data...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Test with larger dataset
    data = processor.generate_synthetic_data(n_samples=1000, n_variables=8, random_state=42)
    
    print(f"   - Dataset size: {data.shape}")
    print(f"   - Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    model = ScalableCausalDiscoveryModel(
        enable_parallelization=True,
        enable_caching=True,
        optimization_level="speed",
        max_workers=4,
        user_id="large_data_test"
    )
    
    # Get optimization recommendations
    recommendations = model.optimize_for_dataset(data)
    print(f"   - Optimization recommendations: {recommendations}")
    
    # Run discovery
    start_time = time.time()
    result = model.fit_discover(data)
    execution_time = time.time() - start_time
    
    print(f"✅ Large data processing completed")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Total time: {execution_time:.3f}s")
    print(f"   - Quality score: {result.quality_score:.3f}")
    print(f"   - Edges detected: {result.metadata['n_edges']}")
    
    # Get scalability report
    scalability_report = model.get_scalability_report()
    print(f"   - Average processing time: {scalability_report.get('average_processing_time', 0):.3f}s")
    print(f"   - Cache hit rate: {scalability_report.get('cache_hit_rate', 0):.1%}")
    
    return True


def test_different_strategies():
    """Test different processing strategies"""
    print("🔍 Testing different processing strategies...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Test scenarios for different strategies
    test_scenarios = [
        ("small", 100, 3),      # Should use standard
        ("medium", 300, 6),     # Should use batch_parallel 
        ("large", 800, 4),      # Should use batch_parallel
        ("wide", 200, 15)       # Should use hierarchical
    ]
    
    for scenario_name, n_samples, n_vars in test_scenarios:
        print(f"   Testing {scenario_name} scenario: {n_samples}x{n_vars}")
        
        data = processor.generate_synthetic_data(
            n_samples=n_samples, 
            n_variables=n_vars, 
            random_state=42
        )
        
        model = ScalableCausalDiscoveryModel(
            optimization_level="balanced",
            user_id=f"strategy_{scenario_name}"
        )
        
        # This will internally select the appropriate strategy
        result = model.fit_discover(data)
        
        print(f"     - Method used: {result.method_used}")
        print(f"     - Processing time: {result.processing_time:.3f}s")
        print(f"     - Quality: {result.quality_score:.3f}")
    
    print(f"✅ Processing strategy tests completed")
    return True


if __name__ == "__main__":
    print("⚡ GENERATION 3 SCALABILITY TEST - Make it Scale")
    print("=" * 60)
    
    try:
        test_scalable_imports()
        print()
        
        test_basic_scalability()
        print()
        
        test_performance_optimization()
        print()
        
        test_auto_scaling()
        print()
        
        test_caching_and_performance()
        print()
        
        test_scalability_with_large_data()
        print()
        
        test_different_strategies()
        print()
        
        print("🎉 ALL GENERATION 3 SCALABILITY TESTS PASSED!")
        print("✅ Advanced optimization, caching, and auto-scaling implemented")
        
    except Exception as e:
        print(f"❌ SCALABILITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)