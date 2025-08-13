"""Comprehensive demonstration of the enhanced causal discovery toolkit."""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import CausalDiscoveryPipeline, PipelineConfig
from evaluation import CausalDiscoveryEvaluator, BenchmarkConfig
from utils.data_processing import DataProcessor
from utils.validation import DataValidator
from utils.metrics import CausalMetrics


def main():
    """Run comprehensive demonstration of the toolkit."""
    print("🚀 CAUSAL DISCOVERY TOOLKIT - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Initialize components
    data_processor = DataProcessor()
    validator = DataValidator()
    
    # Step 1: Generate and validate synthetic data
    print("\n📊 Step 1: Data Generation and Validation")
    print("-" * 40)
    
    data = data_processor.generate_synthetic_data(
        n_samples=1000,
        n_variables=6,
        noise_level=0.2,
        random_state=42
    )
    
    print(f"Generated data shape: {data.shape}")
    
    # Validate data
    validation_results = validator.validate_dataset(data)
    print(f"Data validation: {'✅ PASSED' if validation_results['is_valid'] else '❌ FAILED'}")
    print(f"Quality score: {validation_results['quality_score']:.2f}")
    
    if validation_results['issues']:
        print("Issues found:")
        for issue in validation_results['issues'][:3]:  # Show first 3
            print(f"  - {issue}")
    
    # Create ground truth (chain structure: X1 -> X2 -> X3 -> X4 -> X5 -> X6)
    n_vars = len(data.columns)
    ground_truth = np.zeros((n_vars, n_vars))
    for i in range(n_vars - 1):
        ground_truth[i, i + 1] = 1
    
    print(f"Ground truth edges: {np.sum(ground_truth)}")
    
    # Step 2: Pipeline-based causal discovery
    print("\n🔍 Step 2: Pipeline-based Causal Discovery")
    print("-" * 40)
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        algorithms=['simple_linear', 'mutual_information', 'bayesian_network'],
        preprocessing_steps=['clean', 'standardize', 'validate'],
        parallel_execution=True,
        max_workers=2,
        validation_enabled=True,
        cross_validation_folds=3,
        bootstrap_samples=50
    )
    
    # Run pipeline
    pipeline = CausalDiscoveryPipeline(pipeline_config)
    pipeline_results = pipeline.run(data, ground_truth)
    
    print(f"✅ Pipeline completed in {pipeline_results.metadata['total_execution_time']:.2f}s")
    print(f"Best algorithm: {pipeline_results.best_result.method_used}")
    print(f"Best F1 score: {pipeline_results.performance_metrics.get(pipeline_results.best_result.method_used.lower(), {}).get('f1_score', 'N/A'):.3f}")
    
    # Show individual algorithm performance
    print("\n📈 Algorithm Performance:")
    for algo, metrics in pipeline_results.performance_metrics.items():
        f1 = metrics.get('f1_score', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        exec_time = pipeline_results.execution_times.get(algo, 0)
        print(f"  {algo:20} F1: {f1:.3f}  P: {precision:.3f}  R: {recall:.3f}  Time: {exec_time:.3f}s")
    
    # Step 3: Comparative evaluation
    print("\n🔬 Step 3: Algorithm Comparison")
    print("-" * 40)
    
    from algorithms.base import SimpleLinearCausalModel
    from algorithms.information_theory import MutualInformationDiscovery
    try:
        from algorithms.bayesian_network import BayesianNetworkDiscovery
        algorithms_to_compare = {
            'Simple Linear': SimpleLinearCausalModel(threshold=0.3),
            'Mutual Information': MutualInformationDiscovery(threshold=0.1),
            'Bayesian Network': BayesianNetworkDiscovery()
        }
    except ImportError:
        algorithms_to_compare = {
            'Simple Linear': SimpleLinearCausalModel(threshold=0.3),
            'Mutual Information': MutualInformationDiscovery(threshold=0.1)
        }
    
    evaluator = CausalDiscoveryEvaluator()
    comparison_df = evaluator.compare_algorithms(algorithms_to_compare, data, ground_truth)
    
    if comparison_df is not None:
        print("Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # Step 4: Scalability analysis
    print("\n📊 Step 4: Scalability Analysis")
    print("-" * 40)
    
    print("Testing scalability of Simple Linear model...")
    scalability_df = evaluator.analyze_scalability(
        SimpleLinearCausalModel(threshold=0.3),
        sample_sizes=[100, 300, 500],
        n_variables_list=[5, 10, 15]
    )
    
    print("Scalability Results (sample):")
    print(scalability_df.head().to_string(index=False, float_format='%.3f'))
    
    # Step 5: Mini benchmark
    print("\n🏆 Step 5: Mini Benchmark")
    print("-" * 40)
    
    # Configure a small benchmark
    benchmark_config = BenchmarkConfig(
        algorithms=['simple_linear', 'mutual_information'],
        data_types=['linear', 'nonlinear'],
        sample_sizes=[200, 500],
        noise_levels=[0.1, 0.3],
        n_variables_list=[5, 10],
        n_repetitions=3,
        parallel_execution=True,
        max_workers=2,
        save_results=False
    )
    
    print("Running mini benchmark (this may take a moment)...")
    benchmark_evaluator = CausalDiscoveryEvaluator(benchmark_config)
    benchmark_results = benchmark_evaluator.run_comprehensive_benchmark()
    
    # Show summary
    successful_results = benchmark_results[benchmark_results['success']]
    if len(successful_results) > 0:
        print("\nBenchmark Summary:")
        summary = successful_results.groupby('algorithm').agg({
            'f1_score': ['mean', 'std'],
            'execution_time': ['mean', 'std']
        }).round(3)
        print(summary)
    
    # Step 6: Advanced metrics analysis
    print("\n📐 Step 6: Advanced Metrics Analysis")
    print("-" * 40)
    
    best_result = pipeline_results.best_result
    comprehensive_metrics = CausalMetrics.evaluate_discovery(
        ground_truth, 
        best_result.adjacency_matrix,
        best_result.confidence_scores
    )
    
    print("Comprehensive Metrics for Best Result:")
    for metric, value in comprehensive_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric:25}: {value:.3f}")
        else:
            print(f"  {metric:25}: {value}")
    
    # Step 7: Results interpretation
    print("\n🎯 Step 7: Results Interpretation")
    print("-" * 40)
    
    print("Ground Truth Adjacency Matrix:")
    print(ground_truth.astype(int))
    print("\nDiscovered Adjacency Matrix:")
    print(best_result.adjacency_matrix.astype(int))
    
    # Edge-by-edge comparison
    true_positives = np.sum((ground_truth == 1) & (best_result.adjacency_matrix == 1))
    false_positives = np.sum((ground_truth == 0) & (best_result.adjacency_matrix == 1))
    false_negatives = np.sum((ground_truth == 1) & (best_result.adjacency_matrix == 0))
    
    print(f"\nEdge Detection Summary:")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    
    # Step 8: Recommendations
    print("\n💡 Step 8: Recommendations")
    print("-" * 40)
    
    f1_score = comprehensive_metrics['f1_score']
    
    if f1_score >= 0.8:
        print("✅ Excellent performance! The discovered causal structure is highly accurate.")
    elif f1_score >= 0.6:
        print("✅ Good performance! Consider tuning parameters for better results.")
        print("💡 Try different algorithms or ensemble methods.")
    elif f1_score >= 0.4:
        print("⚠️  Moderate performance. Consider:")
        print("   - Increasing sample size")
        print("   - Trying different algorithms")
        print("   - Adjusting algorithm parameters")
    else:
        print("⚠️  Low performance. Recommendations:")
        print("   - Check data quality and preprocessing")
        print("   - Increase sample size significantly")
        print("   - Try specialized algorithms for your data type")
        print("   - Consider domain knowledge incorporation")
    
    if validation_results['recommendations']:
        print("\nData Quality Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"   - {rec}")
    
    print("\n🎉 DEMONSTRATION COMPLETED!")
    print("For more advanced usage, see the CLI interface:")
    print("  python -m src.cli discover data.csv --output results.json")
    print("  python -m src.cli benchmark --output benchmark_results/")


if __name__ == "__main__":
    main()