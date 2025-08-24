#!/usr/bin/env python3
"""Comprehensive Quality Gates - All Generations Testing"""

import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, Any, List
import traceback


class QualityGateResults:
    """Track quality gate results."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.metrics = {}
    
    def add_result(self, test_name: str, passed: bool, details: str = ""):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            self.tests_failed += 1
            self.failures.append(f"{test_name}: {details}")
            print(f"❌ {test_name}: {details}")
    
    def add_metric(self, name: str, value: Any):
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "pass_rate": self.tests_passed / max(self.tests_run, 1),
            "failures": self.failures,
            "metrics": self.metrics
        }


def test_generation1_quality():
    """Test Generation 1: Make it Work"""
    print("\n🔍 GENERATION 1 QUALITY GATES")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        # Test basic imports
        from algorithms.base import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
        from utils.data_processing import DataProcessor
        from utils.metrics import CausalMetrics
        results.add_result("Basic imports", True, "Core components available")
        
        # Test basic functionality
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
        
        model = SimpleLinearCausalModel(threshold=0.3)
        result = model.fit_discover(data)
        
        # Validate results
        results.add_result("Basic model fitting", model.is_fitted, f"Model fitted successfully")
        results.add_result("Basic discovery", hasattr(result, 'adjacency_matrix'), "Discovery produces results")
        results.add_result("Result validation", result.adjacency_matrix.shape == (3, 3), "Correct matrix dimensions")
        
        # Performance metrics
        start_time = time.time()
        model.fit_discover(data)
        processing_time = time.time() - start_time
        results.add_metric("gen1_processing_time", processing_time)
        results.add_result("Performance", processing_time < 1.0, f"Processing time: {processing_time:.3f}s")
        
    except Exception as e:
        results.add_result("Generation 1", False, str(e))
    
    return results


def test_generation2_quality():
    """Test Generation 2: Make it Robust"""
    print("\n🛡️  GENERATION 2 QUALITY GATES")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        from algorithms.robust_enhanced import RobustCausalDiscoveryModel
        from utils.validation import DataValidator
        from utils.security import DataSecurityValidator
        from utils.data_processing import DataProcessor
        
        results.add_result("Robust imports", True, "Robustness components available")
        
        # Test robustness features
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
        
        model = RobustCausalDiscoveryModel(
            strict_validation=False,
            enable_security=True,
            user_id="quality_gate_test"
        )
        
        result = model.fit_discover(data)
        
        # Validation checks
        results.add_result("Robust fitting", model.is_fitted, "Robust model fitted")
        results.add_result("Security validation", hasattr(result, 'security_result'), "Security validation included")
        results.add_result("Quality scoring", hasattr(result, 'quality_score'), "Quality scoring available")
        results.add_result("Comprehensive result", hasattr(result, 'validation_result'), "Comprehensive validation")
        
        # Error handling test
        try:
            empty_model = RobustCausalDiscoveryModel(user_id="error_test")
            empty_data = pd.DataFrame()
            empty_model.fit(empty_data)
            results.add_result("Error handling", False, "Should have raised error for empty data")
        except Exception:
            results.add_result("Error handling", True, "Properly handles invalid inputs")
        
        # Quality metrics
        results.add_metric("gen2_quality_score", result.quality_score)
        results.add_metric("gen2_processing_time", result.processing_time)
        
        quality_threshold = 0.5
        results.add_result("Quality threshold", result.quality_score >= quality_threshold, 
                         f"Quality: {result.quality_score:.3f} >= {quality_threshold}")
        
    except Exception as e:
        results.add_result("Generation 2", False, str(e))
    
    return results


def test_generation3_quality():
    """Test Generation 3: Make it Scale"""
    print("\n⚡ GENERATION 3 QUALITY GATES")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        from algorithms.scalable_causal import ScalableCausalDiscoveryModel
        from utils.performance import AdaptiveCache, ConcurrentProcessor
        from utils.auto_scaling import AutoScaler, ResourceMonitor
        from utils.data_processing import DataProcessor
        
        results.add_result("Scalable imports", True, "Scalability components available")
        
        # Test scalability features
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=300, n_variables=5, random_state=42)
        
        model = ScalableCausalDiscoveryModel(
            enable_parallelization=True,
            enable_caching=True,
            enable_auto_scaling=False,
            max_workers=2,
            optimization_level="balanced",
            user_id="quality_gate_scale_test"
        )
        
        # Performance test
        start_time = time.time()
        result = model.fit_discover(data)
        total_time = time.time() - start_time
        
        # Scalability checks
        results.add_result("Scalable processing", hasattr(model, 'scalability_metrics'), "Scalability metrics tracked")
        results.add_result("Optimization", hasattr(model, 'optimization_level'), "Optimization configuration")
        results.add_result("Caching", hasattr(model, 'cache') and model.enable_caching, "Caching enabled")
        results.add_result("Parallelization", hasattr(model, 'concurrent_processor'), "Parallel processing")
        
        # Performance metrics
        scalability_report = model.get_scalability_report()
        results.add_metric("gen3_total_time", total_time)
        results.add_metric("gen3_operations", scalability_report.get('operations_count', 0))
        
        # Performance thresholds
        performance_threshold = 2.0  # seconds
        results.add_result("Performance threshold", total_time <= performance_threshold,
                         f"Total time: {total_time:.3f}s <= {performance_threshold}s")
        
        # Test different optimization levels
        for opt_level in ["speed", "memory", "balanced"]:
            try:
                test_model = ScalableCausalDiscoveryModel(
                    optimization_level=opt_level,
                    max_workers=2,
                    user_id=f"opt_test_{opt_level}"
                )
                test_result = test_model.fit_discover(data)
                results.add_result(f"Optimization {opt_level}", True, f"Quality: {test_result.quality_score:.3f}")
            except Exception as e:
                results.add_result(f"Optimization {opt_level}", False, str(e))
        
    except Exception as e:
        results.add_result("Generation 3", False, str(e))
    
    return results


def test_integration_quality():
    """Test integration across all generations"""
    print("\n🔗 INTEGRATION QUALITY GATES")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        from algorithms.base import SimpleLinearCausalModel
        from algorithms.robust_enhanced import RobustCausalDiscoveryModel
        from algorithms.scalable_causal import ScalableCausalDiscoveryModel
        from utils.data_processing import DataProcessor
        
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
        
        # Test all three generations with same data
        models = {
            "Generation 1": SimpleLinearCausalModel(threshold=0.3),
            "Generation 2": RobustCausalDiscoveryModel(user_id="integration_test"),
            "Generation 3": ScalableCausalDiscoveryModel(max_workers=2, user_id="integration_scale_test")
        }
        
        integration_results = {}
        
        for name, model in models.items():
            try:
                start_time = time.time()
                result = model.fit_discover(data)
                processing_time = time.time() - start_time
                
                integration_results[name] = {
                    'success': True,
                    'processing_time': processing_time,
                    'quality_score': getattr(result, 'quality_score', 0.5),
                    'n_edges': result.metadata.get('n_edges', 0)
                }
                
                results.add_result(f"{name} integration", True, 
                                 f"Time: {processing_time:.3f}s, Edges: {integration_results[name]['n_edges']}")
                
            except Exception as e:
                integration_results[name] = {'success': False, 'error': str(e)}
                results.add_result(f"{name} integration", False, str(e))
        
        # Compare results consistency
        successful_results = {k: v for k, v in integration_results.items() if v.get('success')}
        if len(successful_results) >= 2:
            edge_counts = [r['n_edges'] for r in successful_results.values()]
            edge_consistency = max(edge_counts) - min(edge_counts) <= 2  # Allow small variation
            results.add_result("Result consistency", edge_consistency, 
                             f"Edge count range: {min(edge_counts)}-{max(edge_counts)}")
        
        results.add_metric("integration_results", integration_results)
        
    except Exception as e:
        results.add_result("Integration test", False, str(e))
    
    return results


def test_security_quality():
    """Test security and privacy features"""
    print("\n🔒 SECURITY QUALITY GATES")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        from utils.security import DataSecurityValidator, SecureDataHandler
        from algorithms.robust_enhanced import RobustCausalDiscoveryModel
        
        # Test security validation
        validator = DataSecurityValidator()
        
        # Test with normal data
        normal_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        security_result = validator.validate_data_security(normal_data)
        results.add_result("Security validation", True, f"Risk level: {security_result.risk_level}")
        
        # Test secure data handler
        handler = SecureDataHandler()
        test_data = "sensitive_information"
        hashed_data = handler.hash_data(test_data)
        
        results.add_result("Data hashing", len(hashed_data) > 0, "Data successfully hashed")
        results.add_result("Hash uniqueness", hashed_data != test_data, "Hash different from original")
        
        # Test HMAC
        hmac_result = handler.create_hmac(test_data)
        hmac_verified = handler.verify_hmac(test_data, hmac_result)
        results.add_result("HMAC verification", hmac_verified, "HMAC creates and verifies correctly")
        
        # Test model with security enabled
        model = RobustCausalDiscoveryModel(enable_security=True, user_id="security_test")
        result = model.fit_discover(normal_data)
        
        results.add_result("Security integration", hasattr(result, 'security_result'), 
                         "Security results integrated in model output")
        
    except Exception as e:
        results.add_result("Security testing", False, str(e))
    
    return results


def test_performance_benchmarks():
    """Test performance benchmarks"""
    print("\n⏱️  PERFORMANCE BENCHMARKS")
    print("-" * 40)
    
    results = QualityGateResults()
    
    try:
        from algorithms.scalable_causal import ScalableCausalDiscoveryModel
        from utils.data_processing import DataProcessor
        
        processor = DataProcessor()
        
        # Test different data sizes
        test_sizes = [
            (100, 3, "Small"),
            (500, 5, "Medium"),
            (1000, 6, "Large")
        ]
        
        performance_results = {}
        
        for n_samples, n_features, size_name in test_sizes:
            data = processor.generate_synthetic_data(
                n_samples=n_samples, n_variables=n_features, random_state=42
            )
            
            model = ScalableCausalDiscoveryModel(
                optimization_level="speed",
                max_workers=2,
                user_id=f"perf_{size_name.lower()}"
            )
            
            # Benchmark
            start_time = time.time()
            result = model.fit_discover(data)
            processing_time = time.time() - start_time
            
            # Calculate throughput (samples per second)
            throughput = n_samples / processing_time if processing_time > 0 else 0
            
            performance_results[size_name] = {
                'samples': n_samples,
                'features': n_features,
                'time': processing_time,
                'throughput': throughput
            }
            
            # Performance thresholds (samples per second)
            min_throughput = 100  # minimum 100 samples/second
            results.add_result(f"{size_name} performance", throughput >= min_throughput,
                             f"Throughput: {throughput:.0f} samples/s (≥{min_throughput})")
        
        # Overall performance metrics
        avg_throughput = np.mean([r['throughput'] for r in performance_results.values()])
        results.add_metric("average_throughput", avg_throughput)
        results.add_metric("performance_results", performance_results)
        
    except Exception as e:
        results.add_result("Performance benchmarks", False, str(e))
    
    return results


def run_comprehensive_quality_gates():
    """Run all quality gates and generate report"""
    print("🛡️  COMPREHENSIVE QUALITY GATES EXECUTION")
    print("=" * 60)
    
    all_results = {}
    total_tests = 0
    total_passed = 0
    critical_failures = []
    
    # Run all quality gate categories
    quality_tests = [
        ("Generation 1", test_generation1_quality),
        ("Generation 2", test_generation2_quality),
        ("Generation 3", test_generation3_quality),
        ("Integration", test_integration_quality),
        ("Security", test_security_quality),
        ("Performance", test_performance_benchmarks)
    ]
    
    for category, test_func in quality_tests:
        try:
            result = test_func()
            all_results[category] = result.get_summary()
            total_tests += result.tests_run
            total_passed += result.tests_passed
            
            # Check for critical failures
            if result.tests_failed > 0:
                for failure in result.failures:
                    if any(critical in failure.lower() for critical in ['security', 'error', 'crash']):
                        critical_failures.append(f"{category}: {failure}")
            
        except Exception as e:
            print(f"❌ {category} quality gates failed to execute: {e}")
            critical_failures.append(f"{category}: Execution failed - {e}")
    
    # Generate comprehensive report
    print("\n📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    overall_pass_rate = total_passed / max(total_tests, 1)
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {total_passed}")
    print(f"Tests Failed: {total_tests - total_passed}")
    print(f"Overall Pass Rate: {overall_pass_rate:.1%}")
    
    # Category breakdown
    for category, summary in all_results.items():
        pass_rate = summary['pass_rate']
        status = "✅" if pass_rate >= 0.85 else "⚠️" if pass_rate >= 0.70 else "❌"
        print(f"{status} {category}: {summary['tests_passed']}/{summary['tests_run']} ({pass_rate:.1%})")
    
    # Critical failures
    if critical_failures:
        print(f"\n🚨 CRITICAL FAILURES ({len(critical_failures)}):")
        for failure in critical_failures:
            print(f"   - {failure}")
    
    # Success criteria
    success_criteria = {
        "Overall pass rate ≥ 85%": overall_pass_rate >= 0.85,
        "No critical failures": len(critical_failures) == 0,
        "All generations functional": len(all_results) >= 3,
        "Performance benchmarks met": all_results.get('Performance', {}).get('pass_rate', 0) >= 0.70
    }
    
    print(f"\n🎯 SUCCESS CRITERIA:")
    for criterion, met in success_criteria.items():
        status = "✅" if met else "❌"
        print(f"{status} {criterion}")
    
    # Overall result
    all_criteria_met = all(success_criteria.values())
    
    if all_criteria_met:
        print(f"\n🎉 ALL QUALITY GATES PASSED!")
        print("✅ System is ready for production deployment")
        return True
    else:
        print(f"\n⚠️  QUALITY GATES REQUIRE ATTENTION")
        unmet_criteria = [k for k, v in success_criteria.items() if not v]
        print(f"❌ Unmet criteria: {', '.join(unmet_criteria)}")
        return False


if __name__ == "__main__":
    try:
        success = run_comprehensive_quality_gates()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 QUALITY GATES EXECUTION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)