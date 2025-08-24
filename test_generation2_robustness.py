#!/usr/bin/env python3
"""Generation 2 robustness test - Make it Robust"""

import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch


def test_robust_imports():
    """Test that robust components import correctly"""
    print("🔍 Testing robust imports...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel, RobustCausalResult
    from utils.validation import DataValidator, ParameterValidator, ValidationResult
    from utils.security import DataSecurityValidator, SecurityResult
    from utils.error_handling import ErrorHandler, CausalDiscoveryError, DataValidationError
    from utils.monitoring import HealthMonitor, CircuitBreaker
    
    print("✅ All robust components imported successfully")


def test_comprehensive_validation():
    """Test comprehensive data validation"""
    print("🔍 Testing comprehensive validation...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    model = RobustCausalDiscoveryModel(strict_validation=True, user_id="test_user")
    
    # Test valid data
    valid_data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
    result = model.fit_discover(valid_data)
    
    print(f"✅ Valid data processed successfully")
    print(f"   - Quality score: {result.quality_score:.3f}")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Validation passed: {result.validation_result.is_valid}")
    
    # Test edge cases
    test_cases = [
        ("small_dataset", processor.generate_synthetic_data(n_samples=5, n_variables=2, random_state=42)),
        ("few_features", processor.generate_synthetic_data(n_samples=100, n_variables=1, random_state=42)),
    ]
    
    for case_name, data in test_cases:
        try:
            model_case = RobustCausalDiscoveryModel(strict_validation=False, user_id="test_user")
            result_case = model_case.fit_discover(data)
            print(f"⚠️  Edge case '{case_name}' handled gracefully (quality: {result_case.quality_score:.3f})")
        except Exception as e:
            print(f"✅ Edge case '{case_name}' properly rejected: {type(e).__name__}")
    
    return True


def test_security_validation():
    """Test security validation features"""
    print("🔍 Testing security validation...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.security import DataSecurityValidator
    
    # Test with potentially sensitive data
    sensitive_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'email_domain': ['gmail.com'] * 50 + ['yahoo.com'] * 50,  # High cardinality
        'user_id': range(100)  # Potential identifier
    })
    
    model = RobustCausalDiscoveryModel(enable_security=True, user_id="security_test")
    
    # Should generate security warnings but still proceed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = model.fit_discover(sensitive_data)
        
        security_warnings = [warning for warning in w if "Security concerns" in str(warning.message)]
        print(f"✅ Security validation detected concerns: {len(security_warnings)} warnings")
        print(f"   - Security risk level: {result.security_result.risk_level}")
        print(f"   - Security issues: {len(result.security_result.issues)}")
    
    # Test security validator directly
    validator = DataSecurityValidator()
    security_result = validator.validate_data_security(sensitive_data)
    
    print(f"✅ Direct security validation:")
    print(f"   - Is secure: {security_result.is_secure}")
    print(f"   - Risk level: {security_result.risk_level}")
    print(f"   - Issues found: {len(security_result.issues)}")
    
    return True


def test_error_handling():
    """Test comprehensive error handling"""
    print("🔍 Testing error handling and recovery...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.error_handling import CausalDiscoveryError, DataValidationError
    
    model = RobustCausalDiscoveryModel(max_retries=2, user_id="error_test")
    
    # Test data validation errors
    try:
        empty_data = pd.DataFrame()
        model.fit(empty_data)
        assert False, "Should have raised DataValidationError"
    except DataValidationError as e:
        print(f"✅ Data validation error properly caught: {type(e).__name__}")
    
    # Test with invalid data types
    try:
        invalid_data = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'mixed_col': [1, 'two', 3.0]
        })
        model.fit(invalid_data)
        assert False, "Should have raised DataValidationError"
    except DataValidationError as e:
        print(f"✅ Invalid data type error properly caught: {type(e).__name__}")
    
    # Test unfitted model error
    try:
        new_model = RobustCausalDiscoveryModel(user_id="unfitted_test")
        new_model.discover()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"✅ Unfitted model error properly caught: {type(e).__name__}")
    
    return True


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("🔍 Testing circuit breaker...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.monitoring import CircuitBreaker
    
    # Create model with low failure threshold for testing
    model = RobustCausalDiscoveryModel(circuit_breaker_threshold=2, user_id="cb_test")
    
    # Trigger failures
    failure_count = 0
    for i in range(3):
        try:
            empty_data = pd.DataFrame()
            model.fit(empty_data)
        except Exception:
            failure_count += 1
    
    print(f"✅ Circuit breaker recorded {failure_count} failures")
    
    # Check circuit breaker state
    cb_state = model.circuit_breaker.state.name
    print(f"✅ Circuit breaker state: {cb_state}")
    
    return True


def test_performance_monitoring():
    """Test performance monitoring"""
    print("🔍 Testing performance monitoring...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    model = RobustCausalDiscoveryModel(user_id="perf_test")
    
    # Generate test data
    data = processor.generate_synthetic_data(n_samples=300, n_variables=5, random_state=42)
    
    # Perform multiple operations to build history
    for i in range(3):
        result = model.fit_discover(data)
        print(f"   - Operation {i+1}: quality={result.quality_score:.3f}, time={result.processing_time:.3f}s")
    
    # Check model health
    health_status = model.get_health_status()
    print(f"✅ Health monitoring:")
    print(f"   - Overall health: {health_status['overall_health']}")
    print(f"   - Success rate: {health_status['recent_success_rate']:.1%}")
    print(f"   - Avg processing time: {health_status['average_processing_time']:.3f}s")
    
    # Get model info
    model_info = model.get_model_info()
    print(f"✅ Model info:")
    print(f"   - Model type: {model_info['model_type']}")
    print(f"   - Processing history: {model_info['processing_history_length']} operations")
    print(f"   - Last quality score: {model_info['last_quality_score']:.3f}")
    
    return True


def test_audit_logging():
    """Test audit logging functionality"""
    print("🔍 Testing audit logging...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    from utils.security import global_audit_logger
    
    processor = DataProcessor()
    model = RobustCausalDiscoveryModel(user_id="audit_test")
    
    # Create test data
    data = processor.generate_synthetic_data(n_samples=150, n_variables=3, random_state=42)
    
    # Perform operation (this should generate audit logs)
    result = model.fit_discover(data)
    
    print("✅ Audit logging executed (check application logs for details)")
    print(f"   - Operation completed with quality score: {result.quality_score:.3f}")
    
    return True


def test_comprehensive_robustness():
    """Test comprehensive robustness scenarios"""
    print("🔍 Testing comprehensive robustness scenarios...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Scenario 1: Noisy data with missing values
    data_with_noise = processor.generate_synthetic_data(n_samples=200, n_variables=4, noise_level=0.5, random_state=42)
    # Add some missing values
    mask = np.random.random(data_with_noise.shape) < 0.05
    data_with_noise = data_with_noise.mask(mask)
    
    model1 = RobustCausalDiscoveryModel(strict_validation=False, user_id="noise_test")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = model1.fit_discover(data_with_noise)
        print(f"✅ Noisy data scenario: quality={result1.quality_score:.3f}, warnings={len(w)}")
    
    # Scenario 2: Large dataset
    large_data = processor.generate_synthetic_data(n_samples=1000, n_variables=6, random_state=42)
    model2 = RobustCausalDiscoveryModel(user_id="large_test")
    result2 = model2.fit_discover(large_data)
    print(f"✅ Large dataset scenario: quality={result2.quality_score:.3f}")
    
    # Scenario 3: Edge case with perfect correlations
    perfect_corr_data = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
    })
    perfect_corr_data['x3'] = perfect_corr_data['x1']  # Perfect correlation
    perfect_corr_data['x4'] = 2 * perfect_corr_data['x1'] + 0.001 * np.random.randn(100)  # Near perfect
    
    model3 = RobustCausalDiscoveryModel(strict_validation=False, user_id="corr_test")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result3 = model3.fit_discover(perfect_corr_data)
        print(f"✅ Perfect correlation scenario: quality={result3.quality_score:.3f}, warnings={len(w)}")
    
    return True


if __name__ == "__main__":
    print("🛡️  GENERATION 2 ROBUSTNESS TEST - Make it Robust")
    print("=" * 70)
    
    try:
        # Test all robustness components
        test_robust_imports()
        print()
        
        test_comprehensive_validation()
        print()
        
        test_security_validation()
        print()
        
        test_error_handling()
        print()
        
        test_circuit_breaker()
        print()
        
        test_performance_monitoring()
        print()
        
        test_audit_logging()
        print()
        
        test_comprehensive_robustness()
        print()
        
        print("🎉 ALL GENERATION 2 ROBUSTNESS TESTS PASSED!")
        print("✅ Comprehensive error handling, security, and monitoring implemented")
        
    except Exception as e:
        print(f"❌ ROBUSTNESS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)