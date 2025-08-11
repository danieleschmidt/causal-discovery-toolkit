"""Advanced algorithms demonstration for causal discovery toolkit."""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add both package locations to path for robust import resolution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from causal_discovery_toolkit import SimpleLinearCausalModel, DataProcessor, CausalMetrics
    from causal_discovery_toolkit.algorithms.bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
    from causal_discovery_toolkit.algorithms.information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
except ImportError:
    # Fallback to direct imports
    from algorithms.base import SimpleLinearCausalModel
    from algorithms.bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
    from algorithms.information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
    from utils.data_processing import DataProcessor
    from utils.metrics import CausalMetrics


def generate_complex_causal_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate complex synthetic data with known causal structure."""
    np.random.seed(random_state)
    
    # Complex causal structure: X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4, X2 -> X5
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + np.random.normal(0, 0.5, n_samples)
    x3 = -0.6 * x1 + np.random.normal(0, 0.4, n_samples) 
    x4 = 0.5 * x2 + 0.7 * x3 + np.random.normal(0, 0.3, n_samples)
    x5 = 0.9 * x2 + np.random.normal(0, 0.2, n_samples)
    
    return pd.DataFrame({
        'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5
    })


def generate_true_adjacency() -> np.ndarray:
    """Generate the true adjacency matrix for the complex data."""
    # X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4, X2 -> X5
    adjacency = np.zeros((5, 5))
    adjacency[0, 1] = 1  # X1 -> X2
    adjacency[0, 2] = 1  # X1 -> X3
    adjacency[1, 3] = 1  # X2 -> X4
    adjacency[2, 3] = 1  # X3 -> X4
    adjacency[1, 4] = 1  # X2 -> X5
    return adjacency


def evaluate_algorithm(model, data: pd.DataFrame, true_adjacency: np.ndarray, 
                      algorithm_name: str) -> dict:
    """Evaluate a causal discovery algorithm."""
    print(f"\n🔬 Testing {algorithm_name}...")
    
    start_time = time.time()
    try:
        result = model.fit_discover(data)
        runtime = time.time() - start_time
        
        # Evaluate performance
        metrics = CausalMetrics.evaluate_discovery(
            true_adjacency, 
            result.adjacency_matrix,
            result.confidence_scores
        )
        
        print(f"   ✅ Success - Runtime: {runtime:.3f}s")
        print(f"   📊 Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        print(f"   🎯 Edges: {np.sum(result.adjacency_matrix)}/{np.sum(true_adjacency)} (discovered/true)")
        
        return {
            'algorithm': algorithm_name,
            'success': True,
            'runtime': runtime,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'n_edges_discovered': int(np.sum(result.adjacency_matrix)),
            'n_edges_true': int(np.sum(true_adjacency)),
            'shd': metrics['structural_hamming_distance']
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"   ❌ Failed - {str(e)[:60]}...")
        return {
            'algorithm': algorithm_name,
            'success': False,
            'runtime': runtime,
            'error': str(e)
        }


def main():
    """Demonstrate advanced causal discovery algorithms."""
    print("🧪 Advanced Causal Discovery Algorithms Demo")
    print("=" * 60)
    
    # Generate complex synthetic data
    print("\n1. Generating complex synthetic data...")
    data = generate_complex_causal_data(n_samples=500, random_state=42)
    true_adjacency = generate_true_adjacency()
    
    print(f"   📊 Data shape: {data.shape}")
    print(f"   🎯 True causal edges: {np.sum(true_adjacency)}")
    print(f"   📈 Variables: {list(data.columns)}")
    
    # Preprocess data
    data_processor = DataProcessor()
    data_clean = data_processor.standardize(data_processor.clean_data(data))
    
    # Initialize algorithms to test
    algorithms = [
        (SimpleLinearCausalModel(threshold=0.3), "Simple Linear"),
        (BayesianNetworkDiscovery(score_method='bic', max_parents=2, use_bootstrap=False), "Bayesian Network (BIC)"),
        (BayesianNetworkDiscovery(score_method='aic', max_parents=2, use_bootstrap=False), "Bayesian Network (AIC)"),
        (ConstraintBasedDiscovery(alpha=0.05, max_conditioning_set_size=2), "Constraint-Based (PC-like)"),
        (MutualInformationDiscovery(threshold=0.1, n_bins=8, use_conditional_mi=True), "Mutual Information"),
        (TransferEntropyDiscovery(threshold=0.05, lag=1, n_bins=6), "Transfer Entropy"),
    ]
    
    print(f"\n2. Testing {len(algorithms)} algorithms...")
    
    # Test all algorithms
    results = []
    for model, name in algorithms:
        result = evaluate_algorithm(model, data_clean, true_adjacency, name)
        results.append(result)
    
    # Summarize results
    print(f"\n3. 📋 Algorithm Comparison Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<25} {'Status':<10} {'F1 Score':<10} {'Runtime':<10} {'Edges':<10}")
    print("-" * 80)
    
    successful_results = []
    
    for result in results:
        if result['success']:
            status = "✅ Pass"
            f1_score = f"{result['f1_score']:.3f}"
            runtime = f"{result['runtime']:.3f}s"
            edges = f"{result['n_edges_discovered']}/{result['n_edges_true']}"
            successful_results.append(result)
        else:
            status = "❌ Fail"
            f1_score = "N/A"
            runtime = f"{result['runtime']:.3f}s"
            edges = "N/A"
        
        print(f"{result['algorithm']:<25} {status:<10} {f1_score:<10} {runtime:<10} {edges:<10}")
    
    if successful_results:
        print(f"\n4. 🏆 Performance Rankings (by F1 Score):")
        successful_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        for i, result in enumerate(successful_results):
            print(f"   {i+1}. {result['algorithm']}: F1={result['f1_score']:.3f}, "
                  f"Precision={result['precision']:.3f}, Recall={result['recall']:.3f}")
        
        # Best algorithm analysis
        best_algorithm = successful_results[0]
        print(f"\n5. 🎯 Best Algorithm: {best_algorithm['algorithm']}")
        print(f"   Performance: F1={best_algorithm['f1_score']:.3f}, "
              f"Precision={best_algorithm['precision']:.3f}, Recall={best_algorithm['recall']:.3f}")
        print(f"   Efficiency: {best_algorithm['runtime']:.3f}s runtime")
        print(f"   Discovery: {best_algorithm['n_edges_discovered']}/{best_algorithm['n_edges_true']} edges found")
    
    # Generate temporal data for Transfer Entropy
    print(f"\n6. 🕐 Temporal Causal Discovery Example:")
    print("   Generating time series with X1 -> X2 causal relationship...")
    
    n_time = 200
    x1_temporal = np.random.normal(0, 1, n_time)
    x2_temporal = np.zeros(n_time)
    
    # X2 depends on X1 with lag 1
    for t in range(1, n_time):
        x2_temporal[t] = 0.7 * x1_temporal[t-1] + 0.3 * x2_temporal[t-1] + np.random.normal(0, 0.2)
    
    temporal_data = pd.DataFrame({'X1': x1_temporal, 'X2': x2_temporal})
    temporal_true = np.array([[0, 1], [0, 0]])  # X1 -> X2
    
    # Test Transfer Entropy
    te_model = TransferEntropyDiscovery(threshold=0.02, lag=1, n_bins=5)
    te_result = evaluate_algorithm(te_model, temporal_data, temporal_true, "Transfer Entropy (Temporal)")
    
    print(f"\n7. 💡 Research Insights:")
    print("   • Bayesian Networks excel at discovering direct causal relationships")
    print("   • Constraint-based methods provide good balance of precision and recall")
    print("   • Information-theoretic approaches handle non-linear relationships")  
    print("   • Transfer Entropy captures temporal causality effectively")
    print("   • Algorithm choice should depend on data characteristics and assumptions")
    
    print(f"\n✅ Advanced algorithms demonstration completed!")
    print(f"\n📖 Next Steps:")
    print("   - Try algorithms on your real-world datasets")
    print("   - Experiment with hyperparameter tuning")
    print("   - Combine multiple algorithms for ensemble methods")
    print("   - Consider domain knowledge for algorithm selection")


if __name__ == "__main__":
    main()