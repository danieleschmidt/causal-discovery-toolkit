"""Simple demonstration of advanced algorithms with direct imports."""

import pandas as pd
import numpy as np
import sys
import os

# Add path for direct algorithm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Direct imports to avoid package issues - avoid using algorithms.__init__.py
import sys
sys.path.append('/root/repo/src/algorithms')
sys.path.append('/root/repo/src/utils')

from base import SimpleLinearCausalModel, CausalResult
from bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery  
from information_theory import MutualInformationDiscovery, TransferEntropyDiscovery

# Import utilities with direct path access
import importlib.util
spec = importlib.util.spec_from_file_location("data_processing", "/root/repo/src/utils/data_processing.py")
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)
DataProcessor = data_processing.DataProcessor

spec = importlib.util.spec_from_file_location("metrics", "/root/repo/src/utils/metrics.py") 
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
CausalMetrics = metrics.CausalMetrics


def generate_simple_data(n_samples=300):
    """Generate simple causal data: X1 -> X2 -> X3."""
    np.random.seed(42)
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + np.random.normal(0, 0.3, n_samples)
    x3 = 0.6 * x2 + np.random.normal(0, 0.4, n_samples)
    return pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3})


def main():
    """Demo advanced algorithms."""
    print("🧬 Advanced Causal Discovery Algorithms")
    print("=" * 50)
    
    # Generate data
    data = generate_simple_data()
    true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # X1->X2->X3
    
    # Test algorithms
    algorithms = [
        (SimpleLinearCausalModel(threshold=0.3), "Simple Linear"),
        (BayesianNetworkDiscovery(max_parents=1, use_bootstrap=False), "Bayesian Network"),
        (ConstraintBasedDiscovery(alpha=0.1), "Constraint-Based"),
        (MutualInformationDiscovery(threshold=0.1, n_bins=5), "Mutual Information"),
    ]
    
    processor = DataProcessor()
    data_clean = processor.standardize(data)
    
    print(f"📊 Data: {data.shape}, True edges: {np.sum(true_adj)}")
    
    for model, name in algorithms:
        try:
            result = model.fit_discover(data_clean)
            metrics = CausalMetrics.evaluate_discovery(true_adj, result.adjacency_matrix, result.confidence_scores)
            print(f"✅ {name}: F1={metrics['f1_score']:.3f}, Edges={np.sum(result.adjacency_matrix)}")
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)[:30]}...")
    
    print("\n🎯 Advanced algorithms demonstration complete!")


if __name__ == "__main__":
    main()