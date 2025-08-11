"""Demonstration of robust and ensemble causal discovery."""

import pandas as pd
import numpy as np
import sys
import os

# Add path for direct algorithm imports
sys.path.append('/root/repo/src/algorithms')
sys.path.append('/root/repo/src/utils')

# Import algorithms with fallback handling
try:
    from base import SimpleLinearCausalModel, CausalResult
    from robust_ensemble import RobustEnsembleDiscovery, AdaptiveEnsembleDiscovery
    from bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
    from information_theory import MutualInformationDiscovery
    
    # Import utilities
    import importlib.util
    spec = importlib.util.spec_from_file_location("metrics", "/root/repo/src/utils/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    CausalMetrics = metrics.CausalMetrics
    
    ROBUST_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some robust features unavailable: {e}")
    ROBUST_FEATURES_AVAILABLE = False


def generate_challenging_data(n_samples=400):
    """Generate challenging data with noise and non-linearities."""
    np.random.seed(42)
    
    # Base variables
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.7 * x1 + np.random.normal(0, 0.3, n_samples)
    
    # Non-linear relationship 
    x3 = np.sin(0.5 * x2) + np.random.normal(0, 0.2, n_samples)
    
    # Variable with outliers
    x4 = 0.6 * x3 + np.random.normal(0, 0.4, n_samples)
    outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    x4[outlier_indices] += np.random.normal(0, 3, len(outlier_indices))
    
    # Some missing values
    x5 = 0.4 * x1 + 0.3 * x4 + np.random.normal(0, 0.3, n_samples)
    missing_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    x5[missing_indices] = np.nan
    
    return pd.DataFrame({
        'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5
    })


def generate_true_structure():
    """True causal structure for challenging data."""
    # X1->X2->X3->X4, X1->X5, X4->X5
    adj = np.zeros((5, 5))
    adj[0, 1] = 1  # X1 -> X2
    adj[1, 2] = 1  # X2 -> X3
    adj[2, 3] = 1  # X3 -> X4
    adj[0, 4] = 1  # X1 -> X5
    adj[3, 4] = 1  # X4 -> X5
    return adj


def demo_basic_robustness():
    """Demo basic robustness features."""
    print("\n🛡️ Basic Robustness Demo")
    print("-" * 40)
    
    # Generate challenging data
    data = generate_challenging_data()
    true_adj = generate_true_structure()
    
    print(f"📊 Generated challenging dataset:")
    print(f"   - Shape: {data.shape}")
    print(f"   - Missing values: {data.isnull().sum().sum()}")
    print(f"   - Outliers in X4: ~5% of values")
    print(f"   - Non-linear X2->X3 relationship")
    
    # Test basic algorithms on challenging data
    algorithms = [
        (SimpleLinearCausalModel(threshold=0.3), "Simple Linear"),
    ]
    
    if ROBUST_FEATURES_AVAILABLE:
        algorithms.extend([
            (BayesianNetworkDiscovery(max_parents=2, use_bootstrap=False), "Bayesian Network"),
            (ConstraintBasedDiscovery(alpha=0.1), "Constraint-Based"),
            (MutualInformationDiscovery(threshold=0.1, n_bins=6), "Mutual Information"),
        ])
    
    # Test each algorithm
    results = []
    for model, name in algorithms:
        try:
            # Handle missing values
            data_clean = data.fillna(data.mean())
            
            result = model.fit_discover(data_clean)
            
            if ROBUST_FEATURES_AVAILABLE:
                metrics_result = CausalMetrics.evaluate_discovery(
                    true_adj, result.adjacency_matrix, result.confidence_scores
                )
                f1_score = metrics_result['f1_score']
            else:
                f1_score = "N/A"
            
            results.append((name, "✅", f1_score, np.sum(result.adjacency_matrix)))
            print(f"   ✅ {name}: F1={f1_score}, Edges={np.sum(result.adjacency_matrix)}")
            
        except Exception as e:
            results.append((name, "❌", "N/A", "N/A"))
            print(f"   ❌ {name}: Failed - {str(e)[:40]}...")
    
    return results


def demo_ensemble_methods():
    """Demo ensemble methods."""
    if not ROBUST_FEATURES_AVAILABLE:
        print("\n⚠️ Robust ensemble features not available - skipping ensemble demo")
        return
    
    print("\n🤖 Ensemble Methods Demo")
    print("-" * 40)
    
    data = generate_challenging_data()
    data_clean = data.fillna(data.mean())
    true_adj = generate_true_structure()
    
    # Create robust ensemble
    ensemble = RobustEnsembleDiscovery(
        ensemble_method="weighted_vote",
        consensus_threshold=0.4,
        parallel_execution=False  # For demo simplicity
    )
    
    # Add base models with different strengths
    base_models = [
        (SimpleLinearCausalModel(threshold=0.3), "SimpleLinear", 1.0),
        (BayesianNetworkDiscovery(max_parents=2), "BayesianNet", 1.2),
        (ConstraintBasedDiscovery(alpha=0.1), "ConstraintBased", 1.1),
        (MutualInformationDiscovery(threshold=0.1, n_bins=6), "MutualInfo", 0.9),
    ]
    
    for model, name, weight in base_models:
        ensemble.add_base_model(model, name, weight)
    
    print(f"   🔧 Configured ensemble with {len(base_models)} base models")
    
    try:
        # Run ensemble
        ensemble_result = ensemble.fit_discover(data_clean)
        
        # Evaluate
        metrics_result = CausalMetrics.evaluate_discovery(
            true_adj, ensemble_result.adjacency_matrix, ensemble_result.confidence_scores
        )
        
        print(f"   ✅ Ensemble completed successfully")
        print(f"   📊 Performance: F1={metrics_result['f1_score']:.3f}, "
              f"Precision={metrics_result['precision']:.3f}, Recall={metrics_result['recall']:.3f}")
        print(f"   🎯 Discovered {np.sum(ensemble_result.adjacency_matrix)} edges "
              f"(true: {np.sum(true_adj)})")
        print(f"   🤝 Base models used: {len(ensemble_result.individual_results)}")
        
        return ensemble_result
        
    except Exception as e:
        print(f"   ❌ Ensemble failed: {str(e)}")
        return None


def demo_adaptive_ensemble():
    """Demo adaptive ensemble."""
    if not ROBUST_FEATURES_AVAILABLE:
        print("\n⚠️ Adaptive ensemble features not available - skipping adaptive demo")
        return
    
    print("\n🧠 Adaptive Ensemble Demo")
    print("-" * 40)
    
    data = generate_challenging_data()
    data_clean = data.fillna(data.mean())
    
    try:
        # Create adaptive ensemble
        adaptive_ensemble = AdaptiveEnsembleDiscovery(
            ensemble_method="weighted_vote",
            consensus_threshold=0.4
        )
        
        print(f"   🔍 Analyzing data characteristics...")
        
        # Fit (will automatically select optimal methods)
        adaptive_result = adaptive_ensemble.fit_discover(data_clean)
        
        print(f"   ✅ Adaptive ensemble completed")
        print(f"   🎯 Selected {len(adaptive_ensemble.base_models)} methods based on data analysis")
        print(f"   📊 Discovered {np.sum(adaptive_result.adjacency_matrix)} causal edges")
        
        # Show selected methods
        if hasattr(adaptive_ensemble, 'base_models'):
            methods = [name for _, name, _ in adaptive_ensemble.base_models]
            print(f"   🔧 Auto-selected methods: {methods}")
        
        return adaptive_result
        
    except Exception as e:
        print(f"   ❌ Adaptive ensemble failed: {str(e)}")
        return None


def main():
    """Run robust causal discovery demo."""
    print("🛡️ Robust & Ensemble Causal Discovery Demo")
    print("=" * 60)
    
    # Demo 1: Basic robustness
    basic_results = demo_basic_robustness()
    
    # Demo 2: Ensemble methods
    ensemble_result = demo_ensemble_methods()
    
    # Demo 3: Adaptive ensemble
    adaptive_result = demo_adaptive_ensemble()
    
    # Summary
    print(f"\n📋 Demo Summary")
    print("-" * 30)
    
    if basic_results:
        successful_basic = sum(1 for _, status, _, _ in basic_results if status == "✅")
        print(f"   Basic Methods: {successful_basic}/{len(basic_results)} successful")
    
    if ensemble_result:
        print(f"   Robust Ensemble: ✅ Successful")
    else:
        print(f"   Robust Ensemble: ❌ Failed/Unavailable")
    
    if adaptive_result:
        print(f"   Adaptive Ensemble: ✅ Successful")
    else:
        print(f"   Adaptive Ensemble: ❌ Failed/Unavailable")
    
    print(f"\n💡 Key Robust Features Demonstrated:")
    print("   • Handling missing values and outliers")
    print("   • Non-linear relationship detection")
    if ROBUST_FEATURES_AVAILABLE:
        print("   • Multi-method ensemble voting")
        print("   • Adaptive method selection")
        print("   • Robust validation and error recovery")
    else:
        print("   • Basic robustness (limited features available)")
    
    print(f"\n🚀 Robust causal discovery demo completed!")


if __name__ == "__main__":
    main()