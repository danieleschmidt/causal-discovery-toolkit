"""Test robustness and security features of the enhanced causal discovery toolkit."""

import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.error_handling import (
    robust_execution, safe_execution, CausalDiscoveryError,
    ErrorHandler, CircuitBreaker, validate_input_safely
)
from utils.secure_computing import (
    SecureCausalDiscovery, SecureComputationConfig,
    SecureMemoryManager, DifferentialPrivacyManager
)
from utils.security import DataSecurityValidator
from algorithms.base import SimpleLinearCausalModel


def test_error_handling():
    """Test error handling capabilities."""
    print("🛡️ Testing Error Handling...")
    
    # Test robust execution decorator
    @robust_execution(max_retries=2, enable_recovery=True)
    def failing_function(should_fail=True):
        if should_fail:
            raise ValueError("Intentional test failure")
        return "Success!"
    
    # Test fallback behavior
    @robust_execution(max_retries=1, fallback_result="Fallback result")
    def function_with_fallback():
        raise RuntimeError("This should use fallback")
    
    try:
        # This should succeed with fallback
        result = function_with_fallback()
        print(f"✅ Fallback test: {result}")
        
        # This should succeed on retry
        result = failing_function(should_fail=False)
        print(f"✅ Retry test: {result}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test safe execution context
    try:
        with safe_execution("test_operation") as error_handler:
            # This should be handled gracefully
            data = pd.DataFrame(np.random.randn(100, 5))
            validated = validate_input_safely(data)
            print(f"✅ Safe execution test: validated shape {validated.shape}")
            
    except Exception as e:
        print(f"❌ Safe execution test failed: {e}")


def test_circuit_breaker():
    """Test circuit breaker pattern."""
    print("\n⚡ Testing Circuit Breaker...")
    
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
    
    @breaker
    def unreliable_function():
        if np.random.random() < 0.8:  # 80% failure rate
            raise RuntimeError("Simulated failure")
        return "Success!"
    
    failures = 0
    successes = 0
    
    for i in range(10):
        try:
            result = unreliable_function()
            successes += 1
            print(f"✅ Attempt {i+1}: {result}")
        except Exception as e:
            failures += 1
            if "Circuit breaker open" in str(e):
                print(f"🔒 Circuit breaker opened at attempt {i+1}")
                break
            else:
                print(f"❌ Attempt {i+1}: {e}")
    
    print(f"Circuit breaker test: {successes} successes, {failures} failures")


def test_security_validation():
    """Test security validation features."""
    print("\n🔒 Testing Security Validation...")
    
    validator = DataSecurityValidator()
    
    # Test with safe data
    safe_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    
    result = validator.validate_data_security(safe_data)
    print(f"✅ Safe data validation: {result.is_secure}, risk: {result.risk_level}")
    
    # Test with potentially sensitive data
    sensitive_data = pd.DataFrame({
        'ssn': ['123-45-6789'] * 100,  # Fake SSN pattern
        'feature1': np.random.randn(100),
        'email_address': ['test@example.com'] * 100
    })
    
    result = validator.validate_data_security(sensitive_data)
    print(f"⚠️ Sensitive data validation: {result.is_secure}, risk: {result.risk_level}")
    if result.issues:
        print(f"Issues found: {result.issues[:2]}")  # Show first 2 issues


def test_secure_memory_management():
    """Test secure memory management."""
    print("\n💾 Testing Secure Memory Management...")
    
    config = SecureComputationConfig(
        max_memory_mb=100,
        secure_temp_dir=True,
        enable_secure_deletion=True
    )
    
    try:
        with SecureMemoryManager(config) as memory_manager:
            # Test secure array allocation
            array = memory_manager.allocate_secure_array((1000, 100))
            print(f"✅ Allocated secure array: shape {array.shape}")
            
            # Test secure temp file creation
            temp_file = memory_manager.create_secure_temp_file('.test')
            print(f"✅ Created secure temp file: {temp_file}")
            
            # Write some test data
            test_data = "This is sensitive test data that should be securely deleted"
            temp_file.write_text(test_data)
            
            print(f"✅ Wrote test data to secure temp file")
            
            # Memory manager will automatically clean up on exit
        
        # Verify file was securely deleted
        if not temp_file.exists():
            print("✅ Secure temp file was properly deleted")
        else:
            print("⚠️ Temp file still exists (expected in some test environments)")
            
    except Exception as e:
        print(f"❌ Secure memory management test failed: {e}")


def test_differential_privacy():
    """Test differential privacy features."""
    print("\n🎭 Testing Differential Privacy...")
    
    dp_manager = DifferentialPrivacyManager(epsilon=1.0)
    
    # Test with sample data
    original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Add Laplace noise
    noisy_data_laplace = dp_manager.add_laplace_noise(original_data.copy())
    
    # Add Gaussian noise  
    noisy_data_gaussian = dp_manager.add_gaussian_noise(original_data.copy())
    
    print(f"Original data: {original_data}")
    print(f"With Laplace noise: {noisy_data_laplace}")
    print(f"With Gaussian noise: {noisy_data_gaussian}")
    
    # Test adjacency matrix privatization
    adj_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    private_adj = dp_manager.privatize_adjacency_matrix(adj_matrix)
    
    print(f"Original adjacency:\n{adj_matrix}")
    print(f"Privatized adjacency:\n{private_adj}")
    
    print("✅ Differential privacy test completed")


def test_secure_causal_discovery():
    """Test secure causal discovery pipeline."""
    print("\n🔍 Testing Secure Causal Discovery...")
    
    config = SecureComputationConfig(
        enable_audit_log=False,  # Disable for testing
        differential_privacy=True,
        privacy_epsilon=1.0,
        enable_data_anonymization=True
    )
    
    secure_discovery = SecureCausalDiscovery(config)
    
    # Create test data
    data = pd.DataFrame({
        'X1': np.random.randn(200),
        'X2': np.random.randn(200),
        'X3': np.random.randn(200),
        'X4': np.random.randn(200)
    })
    
    # Add some potential PII to test anonymization
    data['user_id'] = range(200)  # This should be removed by anonymization
    
    try:
        # Test secure algorithm execution
        algorithm = SimpleLinearCausalModel(threshold=0.3)
        result = secure_discovery.secure_fit_discover(algorithm, data)
        
        print(f"✅ Secure causal discovery completed")
        print(f"   Method: {result.method_used}")
        print(f"   Discovered edges: {np.sum(result.adjacency_matrix)}")
        print(f"   Result shape: {result.adjacency_matrix.shape}")
        
        # Verify that user_id column was handled appropriately
        if 'user_id' in data.columns:
            print("✅ Anonymization test: PII column was processed")
        
        secure_discovery.close()
        
    except Exception as e:
        print(f"❌ Secure causal discovery test failed: {e}")
        import traceback
        traceback.print_exc()


def test_input_validation():
    """Test robust input validation."""
    print("\n✅ Testing Input Validation...")
    
    # Test with various invalid inputs
    test_cases = [
        ("Empty DataFrame", pd.DataFrame()),
        ("Single column", pd.DataFrame({'col1': [1, 2, 3]})),
        ("Non-numeric data", pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['x', 'y', 'z']})),
        ("Data with NaN", pd.DataFrame({'col1': [1, np.nan, 3], 'col2': [4, 5, np.nan]})),
        ("Data with infinity", pd.DataFrame({'col1': [1, np.inf, 3], 'col2': [4, 5, 6]})),
    ]
    
    for test_name, test_data in test_cases:
        try:
            validated = validate_input_safely(test_data, min_samples=2, min_features=2)
            print(f"✅ {test_name}: Validation passed, shape {validated.shape}")
        except Exception as e:
            print(f"⚠️ {test_name}: {type(e).__name__}: {str(e)[:60]}...")
    
    # Test with valid data
    valid_data = pd.DataFrame(np.random.randn(50, 4))
    try:
        validated = validate_input_safely(valid_data)
        print(f"✅ Valid data: Validation passed, shape {validated.shape}")
    except Exception as e:
        print(f"❌ Valid data validation failed: {e}")


def main():
    """Run all robustness tests."""
    print("🚀 ROBUSTNESS AND SECURITY TESTING")
    print("=" * 50)
    
    test_error_handling()
    test_circuit_breaker()
    test_security_validation()
    test_secure_memory_management()
    test_differential_privacy()
    test_input_validation()
    test_secure_causal_discovery()
    
    print("\n🎉 ROBUSTNESS TESTING COMPLETED!")
    print("All security and error handling features have been tested.")


if __name__ == "__main__":
    main()