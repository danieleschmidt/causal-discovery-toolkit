#!/usr/bin/env python3
"""Test script to verify fixed imports work correctly."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_algorithm_imports():
    """Test basic algorithm imports."""
    print("Testing basic algorithm imports...")
    
    from algorithms.base import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
    print("✓ Base algorithm classes imported")
    
    from algorithms.robust import RobustSimpleLinearCausalModel
    print("✓ Robust algorithm classes imported")
    
    from algorithms.optimized import OptimizedCausalModel
    print("✓ Optimized algorithm classes imported")
    
    return True

def test_advanced_algorithm_imports():
    """Test advanced algorithm imports."""
    print("\nTesting advanced algorithm imports...")
    
    from algorithms.distributed_discovery import DistributedCausalDiscovery
    print("✓ Distributed discovery imported")
    
    from algorithms.robust_ensemble import RobustEnsembleDiscovery
    print("✓ Robust ensemble imported")
    
    return True

def test_utility_imports():
    """Test utility imports."""
    print("\nTesting utility imports...")
    
    from utils.data_processing import DataProcessor
    print("✓ Data processor imported")
    
    from utils.metrics import CausalMetrics
    print("✓ Metrics imported")
    
    from utils.validation import DataValidator
    print("✓ Validation imported")
    
    return True

def test_advanced_utility_imports():
    """Test advanced utility imports."""
    print("\nTesting advanced utility imports...")
    
    from utils.error_recovery import resilient_causal_discovery
    print("✓ Error recovery imported")
    
    from utils.robust_validation import RobustValidationSuite
    print("✓ Robust validation imported")
    
    from utils.auto_scaling import AutoScaler
    print("✓ Auto scaling imported")
    
    return True

def test_experiments_imports():
    """Test experiments imports."""
    print("\nTesting experiments imports...")
    
    from experiments.benchmark import CausalBenchmark
    print("✓ Benchmark imported")
    
    return True

def test_basic_functionality():
    """Test basic functionality works."""
    print("\nTesting basic functionality...")
    
    import numpy as np
    import pandas as pd
    from algorithms.base import SimpleLinearCausalModel
    
    # Create simple test data
    np.random.seed(42)
    data = pd.DataFrame({
        'X': np.random.randn(100),
        'Y': np.random.randn(100),
        'Z': np.random.randn(100)
    })
    
    # Create and test model
    model = SimpleLinearCausalModel()
    model.fit(data)
    result = model.discover()
    
    print(f"✓ Basic model works - discovered {result.adjacency_matrix.shape} adjacency matrix")
    print(f"✓ Method used: {result.method_used}")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_algorithm_imports()
        test_advanced_algorithm_imports()
        test_utility_imports()
        test_advanced_utility_imports()
        test_experiments_imports()
        test_basic_functionality()
        
        print("\n" + "="*50)
        print("🎉 ALL IMPORT TESTS PASSED!")
        print("🎉 BASIC FUNCTIONALITY WORKS!")
        print("🎉 PACKAGE IS READY TO USE!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)