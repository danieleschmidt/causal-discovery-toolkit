"""Command Line Interface for Causal Discovery Toolkit."""

import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    from .pipeline import CausalDiscoveryPipeline, PipelineConfig
    from .evaluation import CausalDiscoveryEvaluator, BenchmarkConfig
    from .utils.data_processing import DataProcessor
    from .utils.validation import DataValidator
    from .algorithms.base import SimpleLinearCausalModel
    from .algorithms.information_theory import MutualInformationDiscovery
    from .algorithms.bayesian_network import BayesianNetworkDiscovery
except ImportError:
    from pipeline import CausalDiscoveryPipeline, PipelineConfig
    from evaluation import CausalDiscoveryEvaluator, BenchmarkConfig
    from utils.data_processing import DataProcessor
    from utils.validation import DataValidator
    from algorithms.base import SimpleLinearCausalModel
    from algorithms.information_theory import MutualInformationDiscovery
    from algorithms.bayesian_network import BayesianNetworkDiscovery


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('causal_discovery.log')
        ]
    )


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif path.suffix.lower() == '.json':
        return pd.read_json(file_path)
    elif path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_ground_truth(file_path: str) -> np.ndarray:
    """Load ground truth adjacency matrix."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    if path.suffix.lower() == '.csv':
        return pd.read_csv(file_path, header=None).values
    elif path.suffix.lower() == '.npy':
        return np.load(file_path)
    elif path.suffix.lower() == '.txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported ground truth format: {path.suffix}")


def save_results(results, output_path: str, format: str = 'json'):
    """Save results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        if hasattr(results, 'best_result'):
            # Pipeline result
            serializable_results = {
                'best_algorithm': results.best_result.method_used,
                'adjacency_matrix': results.best_result.adjacency_matrix.tolist(),
                'confidence_scores': results.best_result.confidence_scores.tolist(),
                'metadata': results.best_result.metadata,
                'execution_times': results.execution_times,
                'performance_metrics': results.performance_metrics
            }
        else:
            # Simple result
            serializable_results = results
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    elif format == 'csv':
        if hasattr(results, 'adjacency_matrix'):
            # Save adjacency matrix
            pd.DataFrame(results.adjacency_matrix).to_csv(
                output_path.with_suffix('_adjacency.csv'), index=False, header=False
            )
            # Save confidence scores
            pd.DataFrame(results.confidence_scores).to_csv(
                output_path.with_suffix('_confidence.csv'), index=False, header=False
            )
        else:
            # Assume it's a DataFrame
            results.to_csv(output_path.with_suffix('.csv'), index=False)


def discover_command(args):
    """Run causal discovery on data."""
    print("🔍 Running Causal Discovery...")
    
    # Load data
    print(f"Loading data from {args.data}")
    data = load_data(args.data)
    print(f"Data shape: {data.shape}")
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        print(f"Loading ground truth from {args.ground_truth}")
        ground_truth = load_ground_truth(args.ground_truth)
    
    # Setup pipeline configuration
    pipeline_config = PipelineConfig(
        algorithms=args.algorithms or ['simple_linear', 'mutual_information'],
        parallel_execution=not args.no_parallel,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout,
        validation_enabled=not args.no_validation
    )
    
    # Run pipeline
    pipeline = CausalDiscoveryPipeline(pipeline_config)
    results = pipeline.run(data, ground_truth)
    
    # Print results summary
    print("\n✅ Causal Discovery Completed!")
    print(f"Best algorithm: {results.best_result.method_used}")
    print(f"Discovered edges: {np.sum(results.best_result.adjacency_matrix)}")
    print(f"Total execution time: {results.metadata['total_execution_time']:.2f}s")
    
    if results.performance_metrics:
        best_metrics = results.performance_metrics.get(results.best_result.method_used.lower(), {})
        if best_metrics:
            print(f"F1 Score: {best_metrics.get('f1_score', 'N/A'):.3f}")
            print(f"Precision: {best_metrics.get('precision', 'N/A'):.3f}")
            print(f"Recall: {best_metrics.get('recall', 'N/A'):.3f}")
    
    # Save results
    if args.output:
        save_results(results, args.output, args.format)
        print(f"Results saved to {args.output}")
    
    return results


def benchmark_command(args):
    """Run comprehensive benchmark."""
    print("🚀 Running Comprehensive Benchmark...")
    
    # Setup benchmark configuration
    benchmark_config = BenchmarkConfig(
        algorithms=args.algorithms or ['simple_linear', 'mutual_information', 'bayesian_network'],
        data_types=args.data_types or ['linear', 'nonlinear'],
        sample_sizes=args.sample_sizes or [100, 500, 1000],
        noise_levels=args.noise_levels or [0.1, 0.3, 0.5],
        n_variables_list=args.n_variables or [5, 10, 20],
        n_repetitions=args.repetitions,
        parallel_execution=not args.no_parallel,
        max_workers=args.max_workers,
        output_dir=args.output or "benchmark_results"
    )
    
    # Run benchmark
    evaluator = CausalDiscoveryEvaluator(benchmark_config)
    results_df = evaluator.run_comprehensive_benchmark()
    
    print(f"\n✅ Benchmark completed! Results saved to {benchmark_config.output_dir}")
    
    return results_df


def validate_command(args):
    """Validate input data."""
    print("🔍 Validating Data...")
    
    # Load data
    data = load_data(args.data)
    print(f"Data shape: {data.shape}")
    
    # Run validation
    validator = DataValidator()
    validation_results = validator.validate_dataset(data)
    
    # Print results
    print(f"\n✅ Validation {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
    print(f"Data quality score: {validation_results['quality_score']:.2f}")
    
    if validation_results['issues']:
        print("\n⚠️  Issues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    if validation_results['recommendations']:
        print("\n💡 Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec}")
    
    return validation_results


def generate_data_command(args):
    """Generate synthetic data for testing."""
    print("🎲 Generating Synthetic Data...")
    
    data_processor = DataProcessor()
    
    # Generate data
    data = data_processor.generate_synthetic_data(
        n_samples=args.n_samples,
        n_variables=args.n_variables,
        noise_level=args.noise_level,
        random_state=args.seed
    )
    
    # Create ground truth (chain structure)
    ground_truth = np.zeros((args.n_variables, args.n_variables))
    for i in range(args.n_variables - 1):
        ground_truth[i, i + 1] = 1
    
    # Save data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_path, index=False)
    np.savetxt(output_path.with_suffix('.truth.csv'), ground_truth, delimiter=',', fmt='%d')
    
    print(f"✅ Synthetic data generated:")
    print(f"  Data: {output_path}")
    print(f"  Ground truth: {output_path.with_suffix('.truth.csv')}")
    print(f"  Shape: {data.shape}")
    print(f"  True edges: {np.sum(ground_truth)}")


def compare_command(args):
    """Compare multiple algorithms on the same dataset."""
    print("🔬 Comparing Algorithms...")
    
    # Load data
    data = load_data(args.data)
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)
    
    # Initialize algorithms
    algorithms = {}
    for algo_name in args.algorithms:
        if algo_name == 'simple_linear':
            algorithms[algo_name] = SimpleLinearCausalModel(threshold=0.3)
        elif algo_name == 'mutual_information':
            algorithms[algo_name] = MutualInformationDiscovery(threshold=0.1)
        elif algo_name == 'bayesian_network':
            algorithms[algo_name] = BayesianNetworkDiscovery()
    
    # Run comparison
    evaluator = CausalDiscoveryEvaluator()
    
    if ground_truth is not None:
        comparison_df = evaluator.compare_algorithms(algorithms, data, ground_truth)
        
        print("\n📊 Algorithm Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Save results
        if args.output:
            comparison_df.to_csv(args.output, index=False)
            print(f"Comparison results saved to {args.output}")
    else:
        print("⚠️  Ground truth required for algorithm comparison")
    
    return comparison_df if ground_truth is not None else None


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Causal Discovery Toolkit - Advanced causal inference and discovery tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run causal discovery on data
  python -m causal_discovery_toolkit discover data.csv --output results.json
  
  # Run comprehensive benchmark
  python -m causal_discovery_toolkit benchmark --output benchmark_results/
  
  # Validate data quality
  python -m causal_discovery_toolkit validate data.csv
  
  # Generate synthetic test data
  python -m causal_discovery_toolkit generate --n-samples 1000 --n-variables 10 --output test_data.csv
  
  # Compare algorithms
  python -m causal_discovery_toolkit compare data.csv --ground-truth truth.csv --algorithms simple_linear mutual_information
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Run causal discovery on data')
    discover_parser.add_argument('data', help='Input data file (CSV, Excel, JSON, Parquet)')
    discover_parser.add_argument('--ground-truth', help='Ground truth adjacency matrix file')
    discover_parser.add_argument('--algorithms', nargs='+', 
                                help='Algorithms to use (simple_linear, mutual_information, bayesian_network, etc.)')
    discover_parser.add_argument('--output', '-o', help='Output file path')
    discover_parser.add_argument('--format', choices=['json', 'csv'], default='json', 
                                help='Output format')
    discover_parser.add_argument('--no-parallel', action='store_true', 
                                help='Disable parallel execution')
    discover_parser.add_argument('--no-validation', action='store_true',
                                help='Disable data validation')
    discover_parser.add_argument('--max-workers', type=int, default=4,
                                help='Maximum number of parallel workers')
    discover_parser.add_argument('--timeout', type=float, default=300.0,
                                help='Timeout in seconds for each algorithm')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmark')
    benchmark_parser.add_argument('--algorithms', nargs='+',
                                 help='Algorithms to benchmark')
    benchmark_parser.add_argument('--data-types', nargs='+',
                                 help='Data types to test (linear, nonlinear, mixed)')
    benchmark_parser.add_argument('--sample-sizes', nargs='+', type=int,
                                 help='Sample sizes to test')
    benchmark_parser.add_argument('--noise-levels', nargs='+', type=float,
                                 help='Noise levels to test')
    benchmark_parser.add_argument('--n-variables', nargs='+', type=int,
                                 help='Number of variables to test')
    benchmark_parser.add_argument('--repetitions', type=int, default=10,
                                 help='Number of repetitions per configuration')
    benchmark_parser.add_argument('--output', '-o', default='benchmark_results',
                                 help='Output directory')
    benchmark_parser.add_argument('--no-parallel', action='store_true',
                                 help='Disable parallel execution')
    benchmark_parser.add_argument('--max-workers', type=int, default=4,
                                 help='Maximum number of parallel workers')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate input data')
    validate_parser.add_argument('data', help='Input data file to validate')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic test data')
    generate_parser.add_argument('--n-samples', type=int, default=1000,
                                help='Number of samples to generate')
    generate_parser.add_argument('--n-variables', type=int, default=10,
                                help='Number of variables to generate')
    generate_parser.add_argument('--noise-level', type=float, default=0.2,
                                help='Noise level (0.0 to 1.0)')
    generate_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
    generate_parser.add_argument('--output', '-o', default='synthetic_data.csv',
                                help='Output file path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple algorithms')
    compare_parser.add_argument('data', help='Input data file')
    compare_parser.add_argument('--ground-truth', help='Ground truth adjacency matrix file')
    compare_parser.add_argument('--algorithms', nargs='+', required=True,
                               help='Algorithms to compare')
    compare_parser.add_argument('--output', '-o', help='Output comparison results file')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.command == 'discover':
            discover_command(args)
        elif args.command == 'benchmark':
            benchmark_command(args)
        elif args.command == 'validate':
            validate_command(args)
        elif args.command == 'generate':
            generate_data_command(args)
        elif args.command == 'compare':
            compare_command(args)
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()