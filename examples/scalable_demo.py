"""Demonstration of scalable and high-performance causal discovery."""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add path for direct algorithm imports
sys.path.append('/root/repo/src/algorithms')
sys.path.append('/root/repo/src/utils')

# Import algorithms with fallback handling
try:
    from base import SimpleLinearCausalModel
    from distributed_discovery import DistributedCausalDiscovery, MemoryEfficientDiscovery, StreamingCausalDiscovery
    from auto_scaling import AutoScaler, WorkloadEstimator, ResourceMonitor
    
    # Import utilities
    import importlib.util
    spec = importlib.util.spec_from_file_location("metrics", "/root/repo/src/utils/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    CausalMetrics = metrics.CausalMetrics
    
    SCALABLE_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some scalable features unavailable: {e}")
    SCALABLE_FEATURES_AVAILABLE = False


def generate_large_dataset(n_samples=5000, n_variables=8, complexity="medium"):
    """Generate large synthetic dataset for scalability testing."""
    np.random.seed(42)
    
    if complexity == "simple":
        # Simple linear relationships
        data = {}
        x0 = np.random.normal(0, 1, n_samples)
        data['X0'] = x0
        
        for i in range(1, n_variables):
            # Each variable depends on previous with some noise
            data[f'X{i}'] = 0.7 * data[f'X{i-1}'] + np.random.normal(0, 0.5, n_samples)
            
    elif complexity == "medium":
        # Mix of linear and non-linear relationships
        data = {}
        data['X0'] = np.random.normal(0, 1, n_samples)
        data['X1'] = 0.8 * data['X0'] + np.random.normal(0, 0.3, n_samples)
        data['X2'] = np.sin(data['X1']) + np.random.normal(0, 0.4, n_samples)
        data['X3'] = 0.6 * data['X1'] + 0.4 * data['X2'] + np.random.normal(0, 0.3, n_samples)
        
        # Add more variables
        for i in range(4, n_variables):
            parent_idx = np.random.choice(i)
            data[f'X{i}'] = 0.5 * data[f'X{parent_idx}'] + np.random.normal(0, 0.4, n_samples)
            
    elif complexity == "high":
        # Complex multi-parent relationships
        data = {}
        data['X0'] = np.random.normal(0, 1, n_samples)
        data['X1'] = np.random.normal(0, 1, n_samples)
        
        for i in range(2, n_variables):
            # Each variable can have multiple parents
            n_parents = min(np.random.poisson(1.5) + 1, i)
            parent_indices = np.random.choice(i, n_parents, replace=False)
            
            value = np.zeros(n_samples)
            for j, parent_idx in enumerate(parent_indices):
                weight = np.random.normal(0.5, 0.2)
                value += weight * data[f'X{parent_idx}']
            
            # Add non-linearity occasionally
            if np.random.random() > 0.7:
                value = np.tanh(value)
            
            data[f'X{i}'] = value + np.random.normal(0, 0.3, n_samples)
    
    return pd.DataFrame(data)


def demo_resource_monitoring():
    """Demo resource monitoring and workload estimation."""
    if not SCALABLE_FEATURES_AVAILABLE:
        print("\n⚠️ Scalable features not available - skipping resource monitoring demo")
        return
        
    print("\n📊 Resource Monitoring Demo")
    print("-" * 40)
    
    # Initialize resource monitor
    monitor = ResourceMonitor(monitoring_interval=0.5)
    monitor.start_monitoring()
    
    # Generate workload
    data = generate_large_dataset(n_samples=2000, n_variables=6)
    
    print(f"   📈 Started resource monitoring...")
    print(f"   🔍 Generated dataset: {data.shape}")
    
    # Get current metrics
    time.sleep(1)  # Allow monitoring to collect data
    current_metrics = monitor.get_current_metrics()
    
    print(f"   💾 Memory: {current_metrics.memory_used_gb:.2f}GB used, "
          f"{current_metrics.memory_available_gb:.2f}GB available ({current_metrics.memory_percent:.1f}%)")
    print(f"   🖥️ CPU: {current_metrics.cpu_percent:.1f}%")
    
    # Workload estimation
    estimator = WorkloadEstimator()
    workload = estimator.estimate_workload(data, "SimpleLinearCausalModel", {'threshold': 0.3})
    
    print(f"   ⚙️ Workload estimation:")
    print(f"     - Data size: {workload.data_size_mb:.1f}MB")
    print(f"     - Complexity score: {workload.complexity_score:.0e}")
    print(f"     - Est. runtime: {workload.estimated_runtime_seconds:.1f}s")
    print(f"     - Est. memory: {workload.memory_requirement_gb:.2f}GB")
    
    monitor.stop_monitoring()
    return workload


def demo_auto_scaling():
    """Demo auto-scaling capabilities."""
    if not SCALABLE_FEATURES_AVAILABLE:
        print("\n⚠️ Auto-scaling features not available - skipping demo")
        return
        
    print("\n🚀 Auto-Scaling Demo") 
    print("-" * 40)
    
    # Generate different sized datasets
    datasets = [
        ("Small", generate_large_dataset(500, 4, "simple")),
        ("Medium", generate_large_dataset(2000, 6, "medium")),
        ("Large", generate_large_dataset(5000, 8, "high"))
    ]
    
    autoscaler = AutoScaler(scaling_strategy="adaptive")
    
    for size_name, data in datasets:
        print(f"\n   📊 {size_name} Dataset ({data.shape}):")
        
        # Estimate workload
        workload = autoscaler.workload_estimator.estimate_workload(
            data, "SimpleLinearCausalModel", {'threshold': 0.3}
        )
        
        # Get optimal configuration
        config = autoscaler.get_optimal_configuration(workload)
        
        print(f"     - Optimal processes: {config['n_processes']}")
        print(f"     - Memory limit: {config['memory_limit_gb']:.2f}GB")
        print(f"     - Chunking: {'Yes' if config['enable_chunking'] else 'No'}")
        if config['enable_chunking']:
            print(f"     - Chunk size: {config['chunk_size']}")
        print(f"     - Parallel execution: {'Yes' if config['parallel_execution'] else 'No'}")
    
    autoscaler.cleanup()


def demo_distributed_processing():
    """Demo distributed causal discovery."""
    if not SCALABLE_FEATURES_AVAILABLE:
        print("\n⚠️ Distributed processing not available - skipping demo")
        return
        
    print("\n🌐 Distributed Processing Demo")
    print("-" * 40)
    
    # Generate large dataset
    data = generate_large_dataset(n_samples=3000, n_variables=6, complexity="medium")
    
    print(f"   📊 Dataset: {data.shape}")
    
    try:
        # Create distributed discovery
        distributed_model = DistributedCausalDiscovery(
            base_model_class=SimpleLinearCausalModel,
            chunk_size=800,
            overlap_size=50,
            n_processes=2,  # Use 2 processes for demo
            aggregation_method="weighted_average",
            threshold=0.3
        )
        
        print(f"   ⚙️ Configured distributed processing:")
        print(f"     - Chunk size: 800 samples")
        print(f"     - Overlap: 50 samples") 
        print(f"     - Processes: 2")
        print(f"     - Aggregation: weighted_average")
        
        # Run distributed discovery
        print(f"   🚀 Running distributed causal discovery...")
        start_time = time.time()
        
        distributed_result = distributed_model.fit_discover(data)
        
        runtime = time.time() - start_time
        
        print(f"   ✅ Completed in {runtime:.2f} seconds")
        print(f"   📈 Discovered {np.sum(distributed_result.adjacency_matrix)} causal edges")
        print(f"   🧩 Processed {distributed_result.metadata['n_chunks']} chunks")
        print(f"   ⏱️ Total processing time: {distributed_result.metadata['total_processing_time']:.2f}s")
        
        # Compare with regular processing
        print(f"\n   🔄 Comparing with regular processing...")
        regular_model = SimpleLinearCausalModel(threshold=0.3)
        
        start_time = time.time()
        regular_result = regular_model.fit_discover(data)
        regular_runtime = time.time() - start_time
        
        print(f"   📊 Regular processing: {regular_runtime:.2f}s")
        print(f"   📈 Regular discovered: {np.sum(regular_result.adjacency_matrix)} edges")
        
        # Performance comparison
        if runtime < regular_runtime:
            speedup = regular_runtime / runtime
            print(f"   🏆 Distributed is {speedup:.1f}x faster!")
        else:
            slowdown = runtime / regular_runtime  
            print(f"   📉 Distributed is {slowdown:.1f}x slower (overhead for small dataset)")
        
        return distributed_result
        
    except Exception as e:
        print(f"   ❌ Distributed processing failed: {str(e)}")
        return None


def demo_memory_efficient_processing():
    """Demo memory-efficient processing."""
    if not SCALABLE_FEATURES_AVAILABLE:
        print("\n⚠️ Memory-efficient processing not available - skipping demo")
        return
        
    print("\n💾 Memory-Efficient Processing Demo")
    print("-" * 40)
    
    # Generate large dataset that would stress memory
    large_data = generate_large_dataset(n_samples=8000, n_variables=10, complexity="medium")
    
    print(f"   📊 Large dataset: {large_data.shape}")
    print(f"   💾 Memory usage: {large_data.memory_usage(deep=True).sum() / (1024*1024):.1f}MB")
    
    try:
        # Create memory-efficient discovery with small budget
        memory_model = MemoryEfficientDiscovery(
            base_model_class=SimpleLinearCausalModel,
            memory_budget_gb=0.5,  # Limit to 500MB
            chunk_strategy="adaptive",
            compression_enabled=True,
            threshold=0.3
        )
        
        print(f"   ⚙️ Memory-efficient config:")
        print(f"     - Memory budget: 500MB")
        print(f"     - Chunking: adaptive")
        print(f"     - Compression: enabled")
        
        # Run memory-efficient discovery
        print(f"   🚀 Running memory-efficient discovery...")
        start_time = time.time()
        
        memory_result = memory_model.fit_discover(large_data)
        
        runtime = time.time() - start_time
        
        print(f"   ✅ Completed in {runtime:.2f} seconds")
        print(f"   📈 Discovered {np.sum(memory_result.adjacency_matrix)} causal edges")
        print(f"   🧩 Used chunking: {'Yes' if memory_result.metadata.get('chunked_processing', False) else 'No'}")
        
        if memory_result.metadata.get('chunked_processing', False):
            print(f"   📦 Chunk size: {memory_result.metadata['chunk_size']} samples")
            print(f"   🔢 Number of chunks: {memory_result.metadata['n_chunks']}")
        
        return memory_result
        
    except Exception as e:
        print(f"   ❌ Memory-efficient processing failed: {str(e)}")
        return None


def demo_streaming_processing():
    """Demo streaming causal discovery."""
    if not SCALABLE_FEATURES_AVAILABLE:
        print("\n⚠️ Streaming processing not available - skipping demo")
        return
        
    print("\n📡 Streaming Processing Demo")
    print("-" * 40)
    
    try:
        # Generate initial dataset
        initial_data = generate_large_dataset(n_samples=1000, n_variables=5, complexity="simple")
        
        # Create streaming discovery
        streaming_model = StreamingCausalDiscovery(
            base_model_class=SimpleLinearCausalModel,
            window_size=800,
            update_frequency=200,
            decay_factor=0.9,
            threshold=0.3
        )
        
        print(f"   📊 Initial data: {initial_data.shape}")
        print(f"   ⚙️ Streaming config:")
        print(f"     - Window size: 800 samples")
        print(f"     - Update frequency: 200 samples")
        print(f"     - Decay factor: 0.9")
        
        # Initialize with initial data
        print(f"   🚀 Initializing streaming model...")
        streaming_model.fit(initial_data)
        initial_result = streaming_model.discover()
        
        print(f"   📈 Initial discovery: {np.sum(initial_result.adjacency_matrix)} edges")
        
        # Simulate streaming updates
        print(f"   📡 Simulating data stream updates...")
        
        for i in range(3):
            # Generate new batch of data
            new_batch = generate_large_dataset(n_samples=250, n_variables=5, complexity="simple")
            
            print(f"     Update {i+1}: Adding {len(new_batch)} samples...")
            
            # Update model
            updated_result = streaming_model.update(new_batch)
            
            if updated_result:
                print(f"     📈 Updated discovery: {np.sum(updated_result.adjacency_matrix)} edges")
                print(f"     📊 Buffer size: {updated_result.metadata['buffer_size']} samples")
            else:
                print(f"     ⏳ No update triggered (waiting for more samples)")
        
        # Get final result
        final_result = streaming_model.discover()
        print(f"\n   🏁 Final streaming result: {np.sum(final_result.adjacency_matrix)} edges")
        print(f"   🔄 Total updates: {final_result.metadata['n_updates']}")
        
        return final_result
        
    except Exception as e:
        print(f"   ❌ Streaming processing failed: {str(e)}")
        return None


def main():
    """Run scalable causal discovery demo."""
    print("🚀 Scalable & High-Performance Causal Discovery Demo")
    print("=" * 70)
    
    # Demo 1: Resource monitoring
    workload_info = demo_resource_monitoring()
    
    # Demo 2: Auto-scaling
    demo_auto_scaling()
    
    # Demo 3: Distributed processing
    distributed_result = demo_distributed_processing()
    
    # Demo 4: Memory-efficient processing
    memory_result = demo_memory_efficient_processing()
    
    # Demo 5: Streaming processing
    streaming_result = demo_streaming_processing()
    
    # Summary
    print(f"\n📋 Scalability Demo Summary")
    print("-" * 40)
    
    if workload_info:
        print(f"   Resource Monitoring: ✅ Functional")
    else:
        print(f"   Resource Monitoring: ❌ Failed/Unavailable")
    
    if distributed_result:
        print(f"   Distributed Processing: ✅ Functional")
    else:
        print(f"   Distributed Processing: ❌ Failed/Unavailable")
    
    if memory_result:
        print(f"   Memory-Efficient Processing: ✅ Functional")
    else:
        print(f"   Memory-Efficient Processing: ❌ Failed/Unavailable")
    
    if streaming_result:
        print(f"   Streaming Processing: ✅ Functional")
    else:
        print(f"   Streaming Processing: ❌ Failed/Unavailable")
    
    print(f"\n💡 Key Scalability Features Demonstrated:")
    
    if SCALABLE_FEATURES_AVAILABLE:
        print("   • Automatic resource monitoring and workload estimation")
        print("   • Dynamic auto-scaling based on system resources")
        print("   • Distributed processing with chunk-based parallelization")
        print("   • Memory-efficient processing for large datasets")
        print("   • Streaming causal discovery for real-time data")
        print("   • Adaptive parameter tuning and resource management")
    else:
        print("   • Limited scalability features (basic functionality only)")
    
    print(f"\n🎯 Scalability Benefits:")
    print("   • Handle datasets from 100s to 100,000s of samples")
    print("   • Automatic optimization based on available resources")
    print("   • Real-time adaptation to changing system conditions")
    print("   • Memory-conscious processing for resource-constrained environments")
    
    print(f"\n🚀 Scalable causal discovery demo completed!")


if __name__ == "__main__":
    main()