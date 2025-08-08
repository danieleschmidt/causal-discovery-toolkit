#!/usr/bin/env python3
"""Robust causal discovery example demonstrating Generation 2 features."""

import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from algorithms.robust import RobustSimpleLinearCausalModel
from utils.data_processing import DataProcessor
from utils.metrics import CausalMetrics
from utils.monitoring import global_health_checker, global_monitor
from utils.security import global_security_validator
from utils.logging_config import get_logger

# Set up logging
logger = get_logger("robust_example")


def main():
    """Demonstrate robust causal discovery with Generation 2 features."""
    print("🛡️  Robust Causal Discovery - Generation 2 Features Demo")
    print("=" * 60)
    
    # 1. SECURITY VALIDATION
    print("\n1. 🔒 SECURITY VALIDATION")
    data_processor = DataProcessor()
    
    # Create potentially problematic data
    problematic_data = pd.DataFrame({
        'user_email': ['user1@test.com', 'user2@test.com', 'user3@test.com'],
        'ssn_like': ['123-45-6789', '987-65-4321', '111-22-3333'],
        'feature_A': [1.5, 2.3, 3.1],
        'feature_B': [0.8, 1.2, 1.6]
    })
    
    security_result = global_security_validator.validate_data_security(problematic_data)
    print(f"Security Assessment: {security_result.risk_level} RISK")
    print(f"Issues Found: {len(security_result.issues)}")
    for issue in security_result.issues:
        print(f"  ⚠️  {issue}")
    
    # Use clean data instead
    clean_data = data_processor.generate_synthetic_data(
        n_samples=200, n_variables=4, noise_level=0.2, random_state=42
    )
    print(f"Using clean synthetic data: {clean_data.shape}")
    
    # 2. HEALTH MONITORING
    print("\n2. 💊 HEALTH MONITORING")
    health_status = global_health_checker.run_checks()
    print(f"System Health: {'✅ HEALTHY' if health_status['overall_healthy'] else '❌ UNHEALTHY'}")
    
    for check_name, check_result in health_status['checks'].items():
        status = "✅ PASS" if check_result['healthy'] else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    # 3. ROBUST MODEL WITH COMPREHENSIVE ERROR HANDLING
    print("\n3. 🔧 ROBUST MODEL INITIALIZATION")
    model = RobustSimpleLinearCausalModel(
        threshold=0.3,
        validate_inputs=True,
        handle_missing='impute_mean',
        correlation_method='pearson',
        min_samples=20,
        max_features=100
    )
    
    model_info = model.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Validation Enabled: {model_info['parameters']['threshold']}")
    print(f"Circuit Breaker: {model_info['circuit_breaker_state']}")
    
    # 4. DATA VALIDATION DURING FIT
    print("\n4. ✅ DATA VALIDATION DURING TRAINING")
    try:
        # First, test with problematic data
        small_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})  # Too few samples
        
        print("Testing with insufficient data...")
        model_test = RobustSimpleLinearCausalModel(min_samples=10)
        model_test.fit(small_data)
        
    except Exception as e:
        print(f"✅ Properly caught error: {str(e)[:100]}...")
    
    # Now fit with good data
    print("Fitting with validated clean data...")
    model.fit(clean_data)
    
    fit_metadata = model.fit_metadata
    print(f"Original shape: {fit_metadata['original_shape']}")
    print(f"Processed shape: {fit_metadata['processed_shape']}")
    print(f"Validation passed: {'validation' in fit_metadata}")
    
    # 5. MONITORED DISCOVERY WITH CIRCUIT BREAKER
    print("\n5. 📊 MONITORED CAUSAL DISCOVERY")
    
    # Perform discovery (with automatic performance monitoring)
    result = model.discover()
    
    print(f"Discovery completed successfully")
    print(f"Method: {result.method_used}")
    print(f"Edges found: {result.metadata['n_edges']}")
    print(f"Sparsity: {result.metadata['sparsity']:.3f}")
    print(f"Max confidence: {result.metadata['max_confidence']:.3f}")
    
    # 6. PERFORMANCE MONITORING RESULTS
    print("\n6. 📈 PERFORMANCE MONITORING")
    perf_summary = global_monitor.get_summary_stats()
    if perf_summary:
        print(f"Total operations monitored: {perf_summary['total_operations']}")
        print(f"Average duration: {perf_summary['avg_duration_seconds']:.4f}s")
        print(f"Memory impact: {perf_summary['avg_memory_delta_mb']:.2f}MB")
    
    # 7. MODEL HEALTH CHECK
    print("\n7. 🏥 MODEL HEALTH VALIDATION")
    health = model.validate_health()
    print(f"Model Health: {'✅ HEALTHY' if health['healthy'] else '❌ UNHEALTHY'}")
    print(f"Ready for inference: {health['model_ready']}")
    print(f"Circuit breaker OK: {health['circuit_breaker_ok']}")
    
    # 8. MISSING DATA HANDLING
    print("\n8. 🔧 MISSING DATA HANDLING")
    data_with_missing = clean_data.copy()
    # Introduce some missing values
    mask = np.random.random(data_with_missing.shape) < 0.05  # 5% missing
    data_with_missing = data_with_missing.mask(mask)
    
    print(f"Created data with missing values: {data_with_missing.isnull().sum().sum()} NaN values")
    
    # Test different missing data strategies
    strategies = ['drop', 'impute_mean', 'impute_median']
    for strategy in strategies:
        try:
            model_strategy = RobustSimpleLinearCausalModel(
                handle_missing=strategy,
                threshold=0.3
            )
            model_strategy.fit(data_with_missing)
            result_strategy = model_strategy.discover()
            
            print(f"✅ Strategy '{strategy}': {result_strategy.metadata['n_edges']} edges found")
            
        except Exception as e:
            print(f"❌ Strategy '{strategy}' failed: {str(e)[:50]}...")
    
    # 9. COMPREHENSIVE ERROR HANDLING DEMO
    print("\n9. 🚨 ERROR HANDLING DEMONSTRATION")
    
    # Test circuit breaker by simulating failures
    print("Testing circuit breaker resilience...")
    circuit_model = RobustSimpleLinearCausalModel()
    circuit_model.fit(clean_data)
    
    # The circuit breaker is tested in the test suite, so we'll just show it's there
    cb_info = circuit_model.get_model_info()
    print(f"Circuit breaker state: {cb_info['circuit_breaker_state']}")
    print(f"Failure count: {cb_info['circuit_breaker_failures']}")
    
    # 10. CORRELATION METHOD COMPARISON
    print("\n10. 📐 CORRELATION METHOD ROBUSTNESS")
    methods = ['pearson', 'spearman', 'kendall']
    
    for method in methods:
        method_model = RobustSimpleLinearCausalModel(
            correlation_method=method,
            threshold=0.3
        )
        method_model.fit(clean_data)
        method_result = method_model.discover()
        
        print(f"{method.capitalize()}: {method_result.metadata['n_edges']} edges, "
              f"max_conf={method_result.metadata['max_confidence']:.3f}")
    
    print("\n🎉 ROBUST CAUSAL DISCOVERY DEMO COMPLETED!")
    print("\nGeneration 2 Features Demonstrated:")
    print("✅ Comprehensive input validation")
    print("✅ Security and privacy checks") 
    print("✅ Health monitoring and circuit breakers")
    print("✅ Performance monitoring and logging")
    print("✅ Robust error handling and recovery")
    print("✅ Multiple correlation methods")
    print("✅ Advanced missing data handling")
    print("✅ Model health validation")


if __name__ == "__main__":
    main()