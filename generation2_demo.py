#!/usr/bin/env python3
"""Generation 2 robustness demonstration."""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import individual components to avoid complex dependency chain
from utils.data_processing import DataProcessor
from utils.validation import DataValidator, ParameterValidator
from utils.security import DataSecurityValidator
from utils.monitoring import PerformanceMonitor, HealthChecker
from utils.logging_config import get_logger

logger = get_logger("generation2_demo")


def main():
    """Demonstrate Generation 2 robustness features."""
    print("🛡️  Generation 2: ROBUSTNESS & SECURITY DEMO")
    print("=" * 50)
    
    # 1. Security Validation
    print("\n1. 🔒 SECURITY VALIDATION")
    security_validator = DataSecurityValidator()
    
    # Create potentially problematic data
    risky_data = pd.DataFrame({
        'email_col': ['user1@company.com', 'user2@company.com'],
        'ssn_like': ['123-45-6789', '987-65-4321'],
        'normal_feature': [1.2, 2.3]
    })
    
    security_result = security_validator.validate_data_security(risky_data)
    print(f"Security Risk Level: {security_result.risk_level}")
    print(f"Issues Found: {len(security_result.issues)}")
    for issue in security_result.issues:
        print(f"  ⚠️  {issue}")
    
    print("\nRecommendations:")
    for rec in security_result.recommendations:
        print(f"  💡 {rec}")
    
    # 2. Data Validation
    print("\n2. ✅ DATA VALIDATION")
    data_validator = DataValidator(strict=False)
    
    # Create clean synthetic data
    data_processor = DataProcessor()
    clean_data = data_processor.generate_synthetic_data(
        n_samples=100, n_variables=4, random_state=42
    )
    
    validation_result = data_validator.validate_input_data(clean_data)
    print(f"Data Valid: {validation_result.is_valid}")
    print(f"Warnings: {len(validation_result.warnings)}")
    print(f"Sample Size: {validation_result.metadata['n_samples']}")
    print(f"Feature Count: {validation_result.metadata['n_features']}")
    print(f"Memory Usage: {validation_result.metadata['memory_usage_mb']:.2f}MB")
    
    # 3. Parameter Validation
    print("\n3. 🔧 PARAMETER VALIDATION")
    
    # Test threshold validation
    valid_threshold = ParameterValidator.validate_threshold(0.3)
    invalid_threshold = ParameterValidator.validate_threshold(1.5)
    
    print(f"Threshold 0.3 valid: {valid_threshold.is_valid}")
    print(f"Threshold 1.5 valid: {invalid_threshold.is_valid}")
    if invalid_threshold.errors:
        print(f"  Error: {invalid_threshold.errors[0]}")
    
    # Test sample size validation
    sample_validation = ParameterValidator.validate_sample_size(100, 5)
    print(f"Sample size validation: {sample_validation.is_valid}")
    print(f"  Ratio: {sample_validation.metadata['ratio']:.1f} samples per feature")
    
    # 4. Performance Monitoring
    print("\n4. 📊 PERFORMANCE MONITORING")
    monitor = PerformanceMonitor()
    
    # Monitor a dummy operation
    metrics = monitor.start_monitoring("data_processing")
    
    # Simulate some work
    processed_data = data_processor.standardize(clean_data)
    monitor.add_custom_metric("rows_processed", len(processed_data))
    
    final_metrics = monitor.stop_monitoring()
    
    print(f"Operation Duration: {final_metrics.duration:.4f}s")
    print(f"Memory Delta: {final_metrics.memory_usage_end - final_metrics.memory_usage_start:.1f}MB")
    print(f"Rows Processed: {final_metrics.custom_metrics['rows_processed']}")
    
    # 5. Health Checks
    print("\n5. 💊 HEALTH MONITORING")
    health_checker = HealthChecker()
    
    # Run default system health checks
    health_result = health_checker.run_checks()
    
    print(f"Overall Health: {'✅ HEALTHY' if health_result['overall_healthy'] else '❌ UNHEALTHY'}")
    
    for check_name, result in health_result['checks'].items():
        status = "✅ PASS" if result['healthy'] else "❌ FAIL"
        duration = result['duration_seconds']
        print(f"  {status} {check_name} ({duration:.3f}s)")
    
    # Get system information
    sys_info = health_checker.get_system_info()
    print(f"\nSystem Info:")
    print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
    print(f"  Memory: {sys_info.get('memory_total_gb', 0):.1f}GB total")
    print(f"  CPU Usage: {sys_info.get('cpu_usage_percent', 0):.1f}%")
    print(f"  Memory Usage: {sys_info.get('memory_usage_percent', 0):.1f}%")
    
    # 6. Missing Data Handling Demo
    print("\n6. 🔧 MISSING DATA HANDLING")
    
    # Create data with missing values
    data_with_missing = clean_data.copy()
    # Introduce 10% missing values randomly
    mask = np.random.random(data_with_missing.shape) < 0.1
    data_with_missing = data_with_missing.mask(mask)
    
    missing_before = data_with_missing.isnull().sum().sum()
    print(f"Missing values introduced: {missing_before}")
    
    # Test different handling strategies
    strategies = {
        'drop': lambda df: df.dropna(),
        'mean_fill': lambda df: df.fillna(df.mean()),
        'median_fill': lambda df: df.fillna(df.median())
    }
    
    for name, strategy in strategies.items():
        try:
            processed = strategy(data_with_missing)
            missing_after = processed.isnull().sum().sum()
            print(f"  {name}: {len(data_with_missing)} → {len(processed)} rows, {missing_after} NaN remaining")
        except Exception as e:
            print(f"  {name}: Failed - {str(e)}")
    
    # 7. Logging Demo
    print("\n7. 📝 STRUCTURED LOGGING")
    
    logger.info("Starting causal discovery workflow", 
                extra={"operation": "discovery", "data_shape": clean_data.shape})
    logger.warning("High correlation detected", 
                   extra={"correlation": 0.95, "variables": ["X1", "X2"]})
    
    # Demonstrate error logging
    try:
        # Simulate an error
        raise ValueError("Simulated error for logging demo")
    except Exception as e:
        logger.error("Discovery failed", extra={"error_type": type(e).__name__})
    
    print("✅ Structured logs generated (check console output)")
    
    print("\n🎉 GENERATION 2 ROBUSTNESS DEMO COMPLETED!")
    print("\n🛡️  Features Demonstrated:")
    print("  ✅ Security validation and PII detection")
    print("  ✅ Comprehensive data validation") 
    print("  ✅ Parameter validation with warnings")
    print("  ✅ Performance monitoring and metrics")
    print("  ✅ System health checks")
    print("  ✅ Robust missing data handling")
    print("  ✅ Structured logging with context")
    print("  ✅ Error handling and reporting")


if __name__ == "__main__":
    main()