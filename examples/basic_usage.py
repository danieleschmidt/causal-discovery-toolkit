"""Basic usage example of the causal discovery toolkit."""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.base import SimpleLinearCausalModel
from utils.data_processing import DataProcessor
from utils.metrics import CausalMetrics
from experiments.benchmark import CausalBenchmark


def main():
    """Demonstrate basic usage of the causal discovery toolkit."""
    print("🧪 Causal Discovery Toolkit - Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    data_processor = DataProcessor()
    
    # Generate synthetic data with known causal structure
    print("\n1. Generating synthetic data...")
    data = data_processor.generate_synthetic_data(
        n_samples=500,
        n_variables=4,
        noise_level=0.2,
        random_state=42
    )
    print(f"Generated data shape: {data.shape}")
    print(f"Variables: {list(data.columns)}")
    
    # True causal structure (X1 -> X2 -> X3 -> X4)
    n_vars = len(data.columns)
    true_adjacency = np.zeros((n_vars, n_vars))
    for i in range(n_vars - 1):
        true_adjacency[i, i + 1] = 1
    print(f"True edges: {np.sum(true_adjacency)}")
    
    # Clean and standardize data
    print("\n2. Preprocessing data...")
    data_clean = data_processor.clean_data(data)
    data_standardized = data_processor.standardize(data_clean)
    print("Data cleaned and standardized")
    
    # Initialize and run causal discovery
    print("\n3. Running causal discovery...")
    model = SimpleLinearCausalModel(threshold=0.3)
    result = model.fit_discover(data_standardized)
    
    print(f"Method used: {result.method_used}")
    print(f"Discovered edges: {np.sum(result.adjacency_matrix)}")
    print(f"Variables: {result.metadata['variable_names']}")
    
    # Evaluate results
    print("\n4. Evaluating results...")
    metrics = CausalMetrics.evaluate_discovery(
        true_adjacency,
        result.adjacency_matrix,
        result.confidence_scores
    )
    
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Structural Hamming Distance: {metrics['structural_hamming_distance']}")
    
    # Display adjacency matrices
    print("\n5. Adjacency Matrix Comparison:")
    print("True Adjacency Matrix:")
    print(true_adjacency.astype(int))
    print("\\nDiscovered Adjacency Matrix:")
    print(result.adjacency_matrix.astype(int))
    
    # Run quick benchmark
    print("\n6. Running quick benchmark...")
    benchmark = CausalBenchmark()
    benchmark_results = benchmark.run_synthetic_benchmark(
        models=[SimpleLinearCausalModel(threshold=0.3)],
        n_samples_list=[200, 500],
        n_variables_list=[3, 5],
        noise_levels=[0.1, 0.3],
        n_runs=2
    )
    
    summary = benchmark.summarize_results(benchmark_results)
    print(f"Benchmark success rate: {summary['success_rate']:.1%}")
    print(f"Mean runtime: {summary['mean_runtime']:.3f} seconds")
    
    # Show sample of detailed results
    print("\n7. Sample benchmark results:")
    sample_results = benchmark_results.head(3)[['model', 'n_samples', 'n_variables', 
                                               'noise_level', 'f1_score', 'runtime_seconds']]
    print(sample_results.to_string(index=False))
    
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("- Try different thresholds or models")
    print("- Test with your own data")
    print("- Run comprehensive benchmarks")


if __name__ == "__main__":
    main()