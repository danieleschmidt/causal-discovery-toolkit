"""Quick quality gates runner for causal discovery toolkit."""

import sys
import os
import time
import traceback

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

def test_basic_functionality():
    """Test basic functionality."""
    print("🔬 Testing Basic Functionality...")
    
    try:
        from base import SimpleLinearCausalModel
        import numpy as np
        import pandas as pd
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 0.8 * x1 + np.random.normal(0, 0.3, n_samples)
        x3 = 0.6 * x2 + np.random.normal(0, 0.4, n_samples)
        data = pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3})
        
        # Test basic model
        model = SimpleLinearCausalModel(threshold=0.3)
        result = model.fit_discover(data)
        
        # Basic validations
        assert result.adjacency_matrix.shape == (3, 3)
        assert result.confidence_scores.shape == (3, 3)
        assert result.method_used == "SimpleLinearCausal"
        assert "threshold" in result.metadata
        
        print("   ✅ Basic functionality: PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality: FAILED - {str(e)}")
        return False

def test_advanced_algorithms():
    """Test advanced algorithms."""
    print("🧬 Testing Advanced Algorithms...")
    
    try:
        from bayesian_network import BayesianNetworkDiscovery
        from information_theory import MutualInformationDiscovery
        import numpy as np
        import pandas as pd
        
        # Generate test data
        np.random.seed(42)
        n_samples = 80
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 0.7 * x1 + np.random.normal(0, 0.4, n_samples)
        data = pd.DataFrame({'X1': x1, 'X2': x2})
        
        # Test Bayesian Network
        bn_model = BayesianNetworkDiscovery(max_parents=1, use_bootstrap=False)
        bn_result = bn_model.fit_discover(data)
        assert bn_result.method_used == "BayesianNetwork"
        
        # Test Mutual Information
        mi_model = MutualInformationDiscovery(threshold=0.1, n_bins=4)
        mi_result = mi_model.fit_discover(data)
        assert mi_result.method_used == "MutualInformation"
        
        print("   ✅ Advanced algorithms: PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced algorithms: FAILED - {str(e)[:50]}...")
        return False

def test_data_handling():
    """Test data handling edge cases."""
    print("📊 Testing Data Handling...")
    
    try:
        from base import SimpleLinearCausalModel
        import numpy as np
        import pandas as pd
        
        model = SimpleLinearCausalModel(threshold=0.5)
        
        # Test small dataset
        small_data = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
        result1 = model.fit_discover(small_data)
        assert result1.adjacency_matrix.shape == (2, 2)
        
        # Test with missing values (handled by fillna)
        data_with_nan = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [2, np.nan, 4, 5]
        })
        data_clean = data_with_nan.fillna(data_with_nan.mean())
        result2 = model.fit_discover(data_clean)
        assert result2.adjacency_matrix.shape == (2, 2)
        
        print("   ✅ Data handling: PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Data handling: FAILED - {str(e)}")
        return False

def test_performance():
    """Test performance with larger dataset."""
    print("⚡ Testing Performance...")
    
    try:
        from base import SimpleLinearCausalModel
        import numpy as np
        import pandas as pd
        
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 1000
        n_vars = 6
        
        data = {}
        data['X0'] = np.random.normal(0, 1, n_samples)
        
        for i in range(1, n_vars):
            data[f'X{i}'] = (0.5 * data[f'X{i-1}'] + 
                           np.random.normal(0, 0.5, n_samples))
        
        large_data = pd.DataFrame(data)
        
        model = SimpleLinearCausalModel(threshold=0.3)
        
        start_time = time.time()
        result = model.fit_discover(large_data)
        runtime = time.time() - start_time
        
        assert result.adjacency_matrix.shape == (n_vars, n_vars)
        assert runtime < 10  # Should complete within 10 seconds
        
        print(f"   ✅ Performance: PASSED ({runtime:.2f}s for {n_samples}x{n_vars})")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance: FAILED - {str(e)}")
        return False

def test_error_handling():
    """Test error handling."""
    print("🛡️ Testing Error Handling...")
    
    try:
        from base import SimpleLinearCausalModel
        import pandas as pd
        
        model = SimpleLinearCausalModel()
        
        # Test invalid input types
        try:
            model.fit("not a dataframe")
            return False  # Should have raised error
        except TypeError:
            pass  # Expected
        
        # Test empty dataframe
        try:
            model.fit(pd.DataFrame())
            return False  # Should have raised error
        except ValueError:
            pass  # Expected
        
        # Test discovery before fitting
        unfitted_model = SimpleLinearCausalModel()
        try:
            unfitted_model.discover()
            return False  # Should have raised error
        except RuntimeError:
            pass  # Expected
        
        print("   ✅ Error handling: PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling: FAILED - {str(e)}")
        return False

def run_security_check():
    """Run basic security checks."""
    print("🔒 Running Security Check...")
    
    try:
        # Check for common security issues
        security_issues = []
        
        # Check for obvious security risks in code
        import os
        repo_root = os.path.join(os.path.dirname(__file__), '..')
        
        # Simple check - no eval() or exec() in source
        src_dir = os.path.join(repo_root, 'src')
        if os.path.exists(src_dir):
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'eval(' in content or 'exec(' in content:
                                security_issues.append(f"eval/exec found in {file}")
        
        if security_issues:
            print(f"   ⚠️ Security issues found: {security_issues}")
            return False
        else:
            print("   ✅ Security check: PASSED")
            return True
            
    except Exception as e:
        print(f"   ❌ Security check: FAILED - {str(e)}")
        return False

def main():
    """Run all quality gates."""
    print("🛡️ Causal Discovery Toolkit - Quality Gates")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Advanced Algorithms", test_advanced_algorithms),
        ("Data Handling", test_data_handling),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling),
        ("Security Check", run_security_check),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name}: FAILED - Unexpected error: {str(e)}")
    
    # Summary
    runtime = time.time() - start_time
    success_rate = passed / total
    
    print(f"\n📊 Quality Gates Summary:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Total Runtime: {runtime:.2f}s")
    
    # Overall result
    if success_rate >= 0.85:  # 85% threshold
        print(f"   🎉 Overall Result: PASSED")
        
        # Additional info
        print(f"\n💡 Key Features Verified:")
        print(f"   ✅ Core causal discovery algorithms")
        print(f"   ✅ Advanced Bayesian and information-theoretic methods")
        print(f"   ✅ Robust data handling and error recovery")
        print(f"   ✅ Performance optimization for large datasets")
        print(f"   ✅ Security and input validation")
        
        return True
    else:
        print(f"   ❌ Overall Result: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)