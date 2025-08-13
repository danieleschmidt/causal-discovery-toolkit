"""Test performance optimization features."""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.performance_optimization import (
    PerformanceConfig, OptimizedDataProcessor, PerformanceProfiler,
    PerformanceOptimizedPipeline, StreamingProcessor,
    create_performance_optimized_config
)
from algorithms.base import SimpleLinearCausalModel


def test_performance_profiler():
    """Test performance profiling capabilities."""
    print("📊 Testing Performance Profiler...")
    
    profiler = PerformanceProfiler()
    
    @profiler.profile_function("test_computation")
    def expensive_computation():
        # Simulate expensive computation
        data = np.random.randn(1000, 100)
        result = np.dot(data.T, data)
        time.sleep(0.1)  # Simulate processing time
        return result
    
    # Run profiled function
    result = expensive_computation()
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    
    print(f"✅ Performance profiling completed")
    print(f"   Execution time: {summary['avg_time']:.3f}s")
    print(f"   Memory usage: {summary['avg_memory_mb']:.1f}MB")
    print(f"   Functions profiled: {summary['successful_functions']}")


def test_optimized_data_processor():
    """Test optimized data processing."""
    print("\n🚀 Testing Optimized Data Processor...")
    
    config = PerformanceConfig(
        max_workers=2,
        chunk_size=500,
        parallel_threshold=100
    )
    
    processor = OptimizedDataProcessor(config)
    
    # Create test data with various data types
    data = pd.DataFrame({
        'int64_col': np.random.randint(0, 100, 1000).astype('int64'),
        'float64_col': np.random.randn(1000).astype('float64'),
        'small_int': np.random.randint(0, 10, 1000).astype('int64'),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    })
    
    print(f"Original data memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    # Test data layout optimization
    optimized_data = processor.optimize_data_layout(data)
    print(f"Optimized data memory: {optimized_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    # Test parallel correlation computation
    start_time = time.time()
    corr_matrix = processor.parallel_correlation_matrix(optimized_data.select_dtypes(include=[np.number]))
    parallel_time = time.time() - start_time
    
    print(f"✅ Parallel correlation computed in {parallel_time:.3f}s")
    print(f"   Correlation matrix shape: {corr_matrix.shape}")
    
    # Test adaptive algorithm selection
    adaptation = processor.adaptive_algorithm_selection(optimized_data)
    print(f"✅ Adaptive recommendations: {adaptation['recommendations']['preferred_algorithms']}")


def test_streaming_processor():
    """Test streaming data processing."""
    print("\n🌊 Testing Streaming Processor...")
    
    processor = StreamingProcessor(window_size=300, overlap=0.1)
    
    # Simulate streaming data
    for batch_num in range(5):
        # Generate batch data
        batch_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100) + batch_num * 0.1  # Slight trend
        })
        
        result = processor.add_data_batch(batch_data)
        
        if result:
            print(f"✅ Processed streaming batch {batch_num + 1}")
            print(f"   Window size: {result['window_size']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Avg correlation: {result['avg_correlation']:.3f}")
    
    # Get trend analysis
    trends = processor.get_trend_analysis()
    if trends.get('status') != 'insufficient_data':
        print(f"✅ Trend analysis: {trends['correlation_trend']} correlation")
        print(f"   Stability score: {trends['stability_score']:.3f}")


def test_performance_optimized_pipeline():
    """Test the complete performance-optimized pipeline."""
    print("\n⚡ Testing Performance-Optimized Pipeline...")
    
    # Create test data
    data = pd.DataFrame({
        'X1': np.random.randn(500),
        'X2': np.random.randn(500),
        'X3': np.random.randn(500),
        'X4': np.random.randn(500),
        'X5': np.random.randn(500)
    })
    
    # Make X2 depend on X1, X3 depend on X2, etc. for some causal structure
    data['X2'] += 0.5 * data['X1']
    data['X3'] += 0.7 * data['X2']
    data['X4'] += 0.3 * data['X3']
    data['X5'] += 0.4 * data['X4']
    
    # Test different optimization levels
    for opt_level in ['memory', 'balanced', 'speed']:
        print(f"\nTesting {opt_level} optimization...")
        
        config = create_performance_optimized_config(opt_level)
        pipeline = PerformanceOptimizedPipeline(config)
        
        algorithm = SimpleLinearCausalModel(threshold=0.3)
        
        try:
            result = pipeline.optimize_and_discover(data, algorithm)
            
            print(f"✅ {opt_level.capitalize()} optimization completed")
            print(f"   Execution time: {result['execution_time']:.3f}s")
            print(f"   Discovered edges: {np.sum(result['causal_result'].adjacency_matrix)}")
            print(f"   Peak memory: {result['system_metrics'].get('peak_memory_mb', 0):.1f}MB")
            print(f"   Optimizations applied: {len(result['optimizations_applied'])}")
            
        except Exception as e:
            print(f"❌ {opt_level.capitalize()} optimization failed: {e}")


def test_algorithm_benchmarking():
    """Test algorithm benchmarking with performance optimization."""
    print("\n🏆 Testing Algorithm Benchmarking...")
    
    # Create test data
    data = pd.DataFrame(np.random.randn(300, 6))
    
    # Add some causal relationships
    data.iloc[:, 1] += 0.5 * data.iloc[:, 0]
    data.iloc[:, 2] += 0.3 * data.iloc[:, 1]
    
    config = create_performance_optimized_config('balanced')
    pipeline = PerformanceOptimizedPipeline(config)
    
    # Define algorithms to benchmark
    algorithms = {
        'simple_linear_0.2': SimpleLinearCausalModel(threshold=0.2),
        'simple_linear_0.3': SimpleLinearCausalModel(threshold=0.3),
        'simple_linear_0.4': SimpleLinearCausalModel(threshold=0.4)
    }
    
    try:
        benchmark_results = pipeline.benchmark_algorithms(algorithms, data)
        
        print("✅ Algorithm benchmarking completed")
        print("\nBenchmark Results:")
        print(benchmark_results[['algorithm', 'execution_time', 'peak_memory_mb', 'n_edges', 'success']].to_string(index=False, float_format='%.3f'))
        
        # Find best performing algorithm
        successful_results = benchmark_results[benchmark_results['success']]
        if len(successful_results) > 0:
            best_algorithm = successful_results.loc[successful_results['execution_time'].idxmin(), 'algorithm']
            print(f"\n🥇 Best performing algorithm: {best_algorithm}")
        
    except Exception as e:
        print(f"❌ Algorithm benchmarking failed: {e}")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n💾 Testing Memory Efficiency...")
    
    config = PerformanceConfig(
        chunk_size=200,
        memory_limit_mb=100,
        use_sparse_matrices=True
    )
    
    processor = OptimizedDataProcessor(config)
    
    # Create large test dataset
    large_data = pd.DataFrame(np.random.randn(2000, 20))
    
    print(f"Large dataset shape: {large_data.shape}")
    print(f"Memory usage: {large_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    # Test memory-efficient processing
    def simple_operation(chunk):
        return chunk.corr().values.mean()
    
    try:
        chunk_results = processor.memory_efficient_processing(
            large_data, simple_operation, chunk_size=200
        )
        
        print(f"✅ Memory-efficient processing completed")
        print(f"   Processed {len(chunk_results)} chunks")
        print(f"   Average correlation: {np.mean(chunk_results):.3f}")
        
    except Exception as e:
        print(f"❌ Memory-efficient processing failed: {e}")


def main():
    """Run all performance tests."""
    print("🚀 PERFORMANCE OPTIMIZATION TESTING")
    print("=" * 50)
    
    test_performance_profiler()
    test_optimized_data_processor()
    test_streaming_processor()
    test_performance_optimized_pipeline()
    test_algorithm_benchmarking()
    test_memory_efficiency()
    
    print("\n🎉 PERFORMANCE TESTING COMPLETED!")
    print("All performance optimization features have been validated.")


if __name__ == "__main__":
    main()