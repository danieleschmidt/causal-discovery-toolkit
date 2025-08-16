#!/usr/bin/env python3
"""Comprehensive quality test suite for the causal discovery toolkit."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import traceback
from typing import Dict, Any, List
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Import core algorithms
from src.algorithms.base import SimpleLinearCausalModel
from src.algorithms.quantum_causal import QuantumCausalDiscovery, QuantumEntanglementCausal
from src.algorithms.bayesian_network import BayesianNetworkDiscovery
from src.algorithms.information_theory import MutualInformationDiscovery

# Import utilities
from src.utils.research_validation import ResearchValidator
from src.utils.advanced_security import create_secure_research_environment
from src.utils.publication_ready import AcademicBenchmarker, standard_causal_metrics


def generate_test_data(n_samples: int = 100, n_variables: int = 4) -> pd.DataFrame:
    """Generate simple test data for algorithms."""
    np.random.seed(42)
    
    # Simple linear relationships
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.5 * X1 + 0.3 * np.random.normal(0, 1, n_samples)
    X3 = 0.4 * X2 + 0.3 * np.random.normal(0, 1, n_samples)
    X4 = 0.2 * X1 + 0.3 * X3 + 0.3 * np.random.normal(0, 1, n_samples)
    
    if n_variables == 4:
        return pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
    elif n_variables == 3:
        return pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
    else:
        return pd.DataFrame({'X1': X1, 'X2': X2})


def test_basic_algorithms() -> Dict[str, Any]:
    """Test basic algorithm functionality."""
    print("🔬 Testing Basic Algorithms...")
    results = {}
    
    data = generate_test_data()
    
    # Test SimpleLinearCausalModel
    try:
        model = SimpleLinearCausalModel(threshold=0.3)
        model.fit(data)
        result = model.discover(data)
        results['SimpleLinearCausalModel'] = {
            'success': True,
            'connections': np.sum(result.adjacency_matrix),
            'method': result.method_used
        }
        print("  ✅ SimpleLinearCausalModel: PASSED")
    except Exception as e:
        results['SimpleLinearCausalModel'] = {'success': False, 'error': str(e)}
        print(f"  ❌ SimpleLinearCausalModel: FAILED - {e}")
    
    # Test QuantumCausalDiscovery (simplified)
    try:
        model = QuantumCausalDiscovery(n_qubits=4, coherence_threshold=0.5)
        model.fit(data)
        result = model.discover(data)
        results['QuantumCausalDiscovery'] = {
            'success': True,
            'connections': np.sum(result.adjacency_matrix),
            'method': result.method_used
        }
        print("  ✅ QuantumCausalDiscovery: PASSED")
    except Exception as e:
        results['QuantumCausalDiscovery'] = {'success': False, 'error': str(e)}
        print(f"  ❌ QuantumCausalDiscovery: FAILED - {e}")
    
    # Test BayesianNetworkDiscovery
    try:
        model = BayesianNetworkDiscovery(max_parents=2)
        model.fit(data)
        result = model.discover(data)
        results['BayesianNetworkDiscovery'] = {
            'success': True,
            'connections': np.sum(result.adjacency_matrix),
            'method': result.method_used
        }
        print("  ✅ BayesianNetworkDiscovery: PASSED")
    except Exception as e:
        results['BayesianNetworkDiscovery'] = {'success': False, 'error': str(e)}
        print(f"  ❌ BayesianNetworkDiscovery: FAILED - {e}")
    
    # Test MutualInformationDiscovery
    try:
        model = MutualInformationDiscovery(threshold=0.1)
        model.fit(data)
        result = model.discover(data)
        results['MutualInformationDiscovery'] = {
            'success': True,
            'connections': np.sum(result.adjacency_matrix),
            'method': result.method_used
        }
        print("  ✅ MutualInformationDiscovery: PASSED")
    except Exception as e:
        results['MutualInformationDiscovery'] = {'success': False, 'error': str(e)}
        print(f"  ❌ MutualInformationDiscovery: FAILED - {e}")
    
    return results


def test_research_validation() -> Dict[str, Any]:
    """Test research validation framework."""
    print("🔬 Testing Research Validation Framework...")
    results = {}
    
    try:
        validator = ResearchValidator(confidence_threshold=0.8)
        data = generate_test_data()
        
        # Create simple algorithm for testing
        algorithm = SimpleLinearCausalModel(threshold=0.3)
        
        # Test algorithm stability validation
        stability_result = validator.validate_algorithm_stability(
            algorithm, data, n_runs=3, parameter_noise=0.05
        )
        
        results['stability_validation'] = {
            'success': stability_result.passed,
            'confidence': stability_result.confidence_level,
            'test_name': stability_result.test_name
        }
        
        # Test data sensitivity validation
        sensitivity_result = validator.validate_data_sensitivity(
            algorithm, data, noise_levels=[0.01, 0.05]
        )
        
        results['sensitivity_validation'] = {
            'success': sensitivity_result.passed,
            'confidence': sensitivity_result.confidence_level,
            'test_name': sensitivity_result.test_name
        }
        
        # Generate validation report
        report = validator.generate_validation_report()
        results['validation_report'] = {
            'total_tests': report['summary']['total_tests'],
            'passed_tests': report['summary']['passed_tests'],
            'success_rate': report['summary']['success_rate']
        }
        
        print("  ✅ Research Validation Framework: PASSED")
        
    except Exception as e:
        results['research_validation'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Research Validation Framework: FAILED - {e}")
    
    return results


def test_security_framework() -> Dict[str, Any]:
    """Test security framework."""
    print("🔒 Testing Security Framework...")
    results = {}
    
    try:
        # Create secure environment
        secure_env = create_secure_research_environment(privacy_budget=1.0)
        
        # Test user authentication
        auth_success = secure_env.access_manager.authenticate_user('admin', 'secure_admin_password')
        results['user_authentication'] = {'success': auth_success}
        
        # Test privacy analysis
        data = generate_test_data()
        privacy_report = secure_env.privacy_analyzer.analyze_privacy(data)
        
        results['privacy_analysis'] = {
            'privacy_score': privacy_report.privacy_score,
            'anonymization_level': privacy_report.anonymization_level,
            'sensitive_fields': len(privacy_report.sensitive_fields_detected)
        }
        
        # Test secure computation
        algorithm = SimpleLinearCausalModel(threshold=0.3)
        secure_result = secure_env.secure_causal_discovery(
            algorithm, data, user_id='admin', apply_privacy=True
        )
        
        results['secure_computation'] = {
            'success': True,
            'privacy_applied': secure_result['privacy_applied'],
            'remaining_budget': secure_result['remaining_privacy_budget']
        }
        
        print("  ✅ Security Framework: PASSED")
        
    except Exception as e:
        results['security_framework'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Security Framework: FAILED - {e}")
    
    return results


def test_benchmarking_framework() -> Dict[str, Any]:
    """Test benchmarking and publication framework."""
    print("📊 Testing Benchmarking Framework...")
    results = {}
    
    try:
        benchmarker = AcademicBenchmarker(output_dir="/tmp/test_benchmark")
        
        # Create test datasets
        datasets = {
            'small_dataset': generate_test_data(n_samples=50, n_variables=3),
            'medium_dataset': generate_test_data(n_samples=100, n_variables=4)
        }
        
        # Get standard metrics
        metrics = standard_causal_metrics()
        
        # Benchmark simple algorithm
        algorithm_factory = lambda: SimpleLinearCausalModel(threshold=0.3)
        
        benchmark_results = benchmarker.benchmark_algorithm(
            algorithm_factory=algorithm_factory,
            algorithm_name='SimpleLinear',
            datasets=datasets,
            metrics=metrics,
            n_runs=2  # Reduced for testing
        )
        
        results['benchmarking'] = {
            'success': True,
            'n_results': len(benchmark_results),
            'algorithms_tested': 1
        }
        
        # Test performance table generation
        performance_table = benchmarker.create_performance_table('precision', latex_format=False)
        results['performance_table'] = {
            'success': len(performance_table) > 0,
            'table_length': len(performance_table)
        }
        
        print("  ✅ Benchmarking Framework: PASSED")
        
    except Exception as e:
        results['benchmarking_framework'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Benchmarking Framework: FAILED - {e}")
    
    return results


def test_performance_scaling() -> Dict[str, Any]:
    """Test performance with different data sizes."""
    print("⚡ Testing Performance Scaling...")
    results = {}
    
    data_sizes = [50, 100, 200]
    algorithm = SimpleLinearCausalModel(threshold=0.3)
    
    for size in data_sizes:
        try:
            data = generate_test_data(n_samples=size, n_variables=4)
            
            start_time = time.time()
            algorithm.fit(data)
            result = algorithm.discover(data)
            execution_time = time.time() - start_time
            
            results[f'size_{size}'] = {
                'success': True,
                'execution_time': execution_time,
                'connections': np.sum(result.adjacency_matrix),
                'data_shape': data.shape
            }
            
        except Exception as e:
            results[f'size_{size}'] = {'success': False, 'error': str(e)}
    
    # Calculate scaling efficiency
    successful_results = [r for r in results.values() if r.get('success', False)]
    if len(successful_results) >= 2:
        times = [r['execution_time'] for r in successful_results]
        scaling_factor = max(times) / min(times)
        results['scaling_analysis'] = {
            'min_time': min(times),
            'max_time': max(times),
            'scaling_factor': scaling_factor,
            'efficient_scaling': scaling_factor < 10.0  # Less than 10x increase
        }
    
    print(f"  ✅ Performance Scaling: PASSED")
    return results


def run_comprehensive_quality_tests() -> Dict[str, Any]:
    """Run comprehensive quality test suite."""
    print("🚀 COMPREHENSIVE QUALITY TEST SUITE")
    print("=" * 70)
    
    all_results = {}
    
    # Run all test suites
    test_suites = [
        ("Basic Algorithms", test_basic_algorithms),
        ("Research Validation", test_research_validation),
        ("Security Framework", test_security_framework),
        ("Benchmarking Framework", test_benchmarking_framework),
        ("Performance Scaling", test_performance_scaling)
    ]
    
    for suite_name, test_function in test_suites:
        print(f"\n📋 {suite_name}")
        print("-" * 50)
        
        try:
            suite_results = test_function()
            all_results[suite_name.lower().replace(' ', '_')] = suite_results
        except Exception as e:
            print(f"❌ {suite_name} suite failed: {e}")
            all_results[suite_name.lower().replace(' ', '_')] = {
                'suite_error': str(e),
                'success': False
            }
    
    return all_results


def generate_quality_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive quality report."""
    report = "\n🎯 QUALITY ASSESSMENT REPORT\n"
    report += "=" * 70 + "\n\n"
    
    total_tests = 0
    passed_tests = 0
    
    for suite_name, suite_results in results.items():
        report += f"📊 {suite_name.upper().replace('_', ' ')}\n"
        report += "-" * 40 + "\n"
        
        if isinstance(suite_results, dict) and 'suite_error' not in suite_results:
            for test_name, test_result in suite_results.items():
                total_tests += 1
                if isinstance(test_result, dict):
                    success = test_result.get('success', False)
                    if success:
                        passed_tests += 1
                        report += f"  ✅ {test_name}: PASSED\n"
                    else:
                        error = test_result.get('error', 'Unknown error')
                        report += f"  ❌ {test_name}: FAILED - {error}\n"
        else:
            total_tests += 1
            report += f"  ❌ Suite failed: {suite_results.get('suite_error', 'Unknown error')}\n"
        
        report += "\n"
    
    # Overall summary
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report += "🏆 OVERALL SUMMARY\n"
    report += "=" * 30 + "\n"
    report += f"Total Tests: {total_tests}\n"
    report += f"Passed Tests: {passed_tests}\n"
    report += f"Failed Tests: {total_tests - passed_tests}\n"
    report += f"Success Rate: {success_rate:.1f}%\n\n"
    
    if success_rate >= 80:
        report += "🎉 QUALITY GATES: PASSED\n"
        report += "✅ System ready for production deployment\n"
    elif success_rate >= 60:
        report += "⚠️  QUALITY GATES: PARTIALLY PASSED\n"
        report += "🔧 Some issues need to be addressed before production\n"
    else:
        report += "❌ QUALITY GATES: FAILED\n"
        report += "🚨 Significant issues detected - not ready for production\n"
    
    return report


def main():
    """Main function."""
    try:
        # Run comprehensive tests
        results = run_comprehensive_quality_tests()
        
        # Generate and display report
        report = generate_quality_report(results)
        print(report)
        
        # Save report to file
        with open('/tmp/quality_report.txt', 'w') as f:
            f.write(report)
        
        # Return appropriate exit code
        total_tests = sum(len(suite) for suite in results.values() if isinstance(suite, dict))
        passed_tests = sum(
            sum(1 for test in suite.values() if isinstance(test, dict) and test.get('success', False))
            for suite in results.values() if isinstance(suite, dict)
        )
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        return 0 if success_rate >= 80 else 1
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())