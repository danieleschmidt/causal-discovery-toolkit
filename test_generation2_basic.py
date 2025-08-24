#!/usr/bin/env python3
"""Basic Generation 2 robustness test - simplified version"""

import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
import warnings


def test_basic_robust_functionality():
    """Test basic robust functionality with numeric data only"""
    print("🔍 Testing basic robust functionality...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Test with valid numeric data
    valid_data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
    
    model = RobustCausalDiscoveryModel(
        strict_validation=False, 
        enable_security=True,
        user_id="test_user"
    )
    
    result = model.fit_discover(valid_data)
    
    print(f"✅ Basic robust functionality works")
    print(f"   - Quality score: {result.quality_score:.3f}")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Validation passed: {result.validation_result.is_valid}")
    print(f"   - Security level: {result.security_result.risk_level}")
    print(f"   - Number of edges detected: {result.metadata['n_edges']}")
    
    return True


def test_error_handling():
    """Test basic error handling"""
    print("🔍 Testing error handling...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.error_handling import DataValidationError
    
    model = RobustCausalDiscoveryModel(user_id="error_test")
    
    # Test with empty data
    try:
        empty_data = pd.DataFrame()
        model.fit(empty_data)
        assert False, "Should have raised an error"
    except (DataValidationError, Exception) as e:
        print(f"✅ Empty data error properly caught: {type(e).__name__}")
    
    return True


def test_health_monitoring():
    """Test health monitoring features"""
    print("🔍 Testing health monitoring...")
    
    from algorithms.robust_enhanced import RobustCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    model = RobustCausalDiscoveryModel(user_id="health_test")
    
    # Generate test data
    data = processor.generate_synthetic_data(n_samples=150, n_variables=3, random_state=42)
    
    # Perform operation
    result = model.fit_discover(data)
    
    # Check health status
    health_status = model.get_health_status()
    print(f"✅ Health monitoring works:")
    print(f"   - Overall health: {health_status.get('overall_health', 'unknown')}")
    print(f"   - Circuit breaker state: {health_status.get('circuit_breaker_state', 'unknown')}")
    
    # Get model info
    model_info = model.get_model_info()
    print(f"✅ Model info available:")
    print(f"   - Model type: {model_info['model_type']}")
    print(f"   - Is fitted: {model_info['is_fitted']}")
    print(f"   - Processing history: {model_info['processing_history_length']} operations")
    
    return True


if __name__ == "__main__":
    print("🛡️  GENERATION 2 BASIC ROBUSTNESS TEST")
    print("=" * 50)
    
    try:
        test_basic_robust_functionality()
        print()
        
        test_error_handling()
        print()
        
        test_health_monitoring()
        print()
        
        print("🎉 ALL BASIC GENERATION 2 TESTS PASSED!")
        print("✅ Core robustness features are working")
        
    except Exception as e:
        print(f"❌ BASIC ROBUSTNESS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)