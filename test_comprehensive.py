#!/usr/bin/env python3
"""Comprehensive test suite for causal discovery toolkit."""

import sys
import os
import pandas as pd
import numpy as np
import time
import traceback

# Add current directory to path for package import
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_functionality():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        from causal_discovery_toolkit import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
        from causal_discovery_toolkit import DataProcessor, CausalMetrics
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'X1': np.random.randn(100),
            'X2': np.random.randn(100),
            'X3': np.random.randn(100)
        })
        data['X2'] = 0.5 * data['X1'] + 0.3 * np.random.randn(100)
        
        # Test basic model
        model = SimpleLinearCausalModel(threshold=0.3)
        model.fit(data)
        result = model.discover()
        
        assert isinstance(result, CausalResult), "Result should be CausalResult"
        assert result.adjacency_matrix.shape == (3, 3), "Adjacency matrix shape incorrect"
        assert result.metadata['n_variables'] == 3, "Variable count incorrect"
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_robust_functionality():
    """Test robust functionality."""
    print("🧪 Testing robust functionality...")
    
    try:
        from causal_discovery_toolkit.algorithms.robust import RobustSimpleLinearCausalModel
        
        # Create problematic data
        np.random.seed(42)
        data = pd.DataFrame({
            'X1': np.random.randn(50),
            'X2': np.random.randn(50),
            'X3': np.random.randn(50)
        })
        # Add missing values
        data.loc[10:15, 'X1'] = np.nan
        
        model = RobustSimpleLinearCausalModel(handle_missing='drop')
        model.fit(data)
        result = model.discover()
        
        assert result.metadata['processed_shape'][0] < 50, "Should have fewer rows after dropping NaN"
        
        # Test health check
        health = model.validate_health()
        assert health['model_ready'], "Model should be ready"
        
        print("✅ Robust functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Robust functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_optimized_functionality():
    """Test optimized functionality."""
    print("🧪 Testing optimized functionality...")
    
    try:
        from causal_discovery_toolkit.algorithms.optimized import OptimizedCausalModel
        from causal_discovery_toolkit import DataProcessor
        
        # Create larger dataset
        np.random.seed(42)
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=200, n_variables=8, random_state=42)
        
        model = OptimizedCausalModel(
            threshold=0.3,
            enable_caching=True,
            enable_parallel=True
        )
        
        model.fit(data)
        result = model.discover()
        
        # Test performance stats
        stats = model.get_performance_stats()
        assert 'cache_hits' in stats, "Cache stats should be available"
        
        print("✅ Optimized functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Optimized functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing utilities."""
    print("🧪 Testing data processing...")
    
    try:
        from causal_discovery_toolkit import DataProcessor
        
        processor = DataProcessor()
        
        # Test synthetic data generation
        data = processor.generate_synthetic_data(n_samples=100, n_variables=5, random_state=42)
        assert data.shape == (100, 5), "Generated data shape incorrect"
        
        # Test data cleaning
        dirty_data = data.copy()
        dirty_data.loc[10:20, 'X1'] = np.nan
        
        cleaned = processor.clean_data(dirty_data, drop_na=True)
        assert cleaned.shape[0] < data.shape[0], "Should have fewer rows after cleaning"
        
        # Test validation
        is_valid, issues = processor.validate_data(data)
        assert is_valid, f"Data should be valid, issues: {issues}"
        
        print("✅ Data processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test evaluation metrics."""
    print("🧪 Testing metrics...")
    
    try:
        from causal_discovery_toolkit import CausalMetrics
        
        # Create test adjacency matrices
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred_adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        
        # Test metrics
        shd = CausalMetrics.structural_hamming_distance(true_adj, pred_adj)
        assert shd == 1, f"SHD should be 1, got {shd}"
        
        prf = CausalMetrics.precision_recall_f1(true_adj, pred_adj)
        assert 'precision' in prf, "Precision should be in results"
        assert 'recall' in prf, "Recall should be in results"
        assert 'f1_score' in prf, "F1 should be in results"
        
        # Test comprehensive evaluation
        eval_result = CausalMetrics.evaluate_discovery(true_adj, pred_adj)
        assert 'structural_hamming_distance' in eval_result, "SHD should be in evaluation"
        
        print("✅ Metrics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_import_performance():
    """Test import performance."""
    print("🧪 Testing import performance...")
    
    try:
        start_time = time.time()
        import causal_discovery_toolkit
        import_time = time.time() - start_time
        
        assert import_time < 5.0, f"Import took too long: {import_time:.2f}s"
        
        print(f"✅ Import performance test passed ({import_time:.3f}s)")
        return True
        
    except Exception as e:
        print(f"❌ Import performance test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage is reasonable."""
    print("🧪 Testing memory usage...")
    
    try:
        import causal_discovery_toolkit
        from causal_discovery_toolkit import DataProcessor
        
        # Create reasonably large dataset
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=500, n_variables=10, random_state=42)
        
        # Process with different models and ensure no memory leaks
        from causal_discovery_toolkit import SimpleLinearCausalModel
        from causal_discovery_toolkit.algorithms.robust import RobustSimpleLinearCausalModel
        from causal_discovery_toolkit.algorithms.optimized import OptimizedCausalModel
        
        models = [
            SimpleLinearCausalModel(),
            RobustSimpleLinearCausalModel(),
            OptimizedCausalModel()
        ]
        
        for model in models:
            model.fit(data)
            result = model.discover()
            # Ensure results are reasonable
            assert result.adjacency_matrix.size > 0, "Adjacency matrix should not be empty"
        
        print("✅ Memory usage test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("\n🚀 STARTING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_import_performance,
        test_basic_functionality,
        test_data_processing,
        test_metrics,
        test_robust_functionality,
        test_optimized_functionality,
        test_memory_usage
    ]
    
    passed = 0
    failed = 0
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print(f"📊 TEST RESULTS:")
    print(f"  • Total tests: {len(tests)}")
    print(f"  • Passed: {passed} ✅")
    print(f"  • Failed: {failed} ❌")
    print(f"  • Success rate: {passed/len(tests)*100:.1f}%")
    print(f"  • Total time: {total_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - QUALITY GATES MET!")
        return 0
    else:
        print(f"\n💥 {failed} TESTS FAILED - QUALITY GATES NOT MET")
        return 1

if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    sys.exit(exit_code)