"""Comprehensive evaluation framework for causal discovery algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from .algorithms.base import CausalDiscoveryModel, CausalResult
    from .utils.metrics import CausalMetrics
    from .utils.data_processing import DataProcessor
    from .pipeline import CausalDiscoveryPipeline, PipelineConfig
except ImportError:
    from algorithms.base import CausalDiscoveryModel, CausalResult
    from utils.metrics import CausalMetrics
    from utils.data_processing import DataProcessor
    from pipeline import CausalDiscoveryPipeline, PipelineConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    algorithms: List[str] = None
    data_types: List[str] = None
    sample_sizes: List[int] = None
    noise_levels: List[float] = None
    n_variables_list: List[int] = None
    n_repetitions: int = 10
    timeout_per_run: float = 60.0
    save_results: bool = True
    output_dir: str = "benchmark_results"
    parallel_execution: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ['simple_linear', 'mutual_information', 'bayesian_network']
        if self.data_types is None:
            self.data_types = ['linear', 'nonlinear', 'mixed']
        if self.sample_sizes is None:
            self.sample_sizes = [100, 500, 1000]
        if self.noise_levels is None:
            self.noise_levels = [0.1, 0.3, 0.5]
        if self.n_variables_list is None:
            self.n_variables_list = [5, 10, 20]


@dataclass
class ExperimentResult:
    """Results from a single benchmark experiment."""
    algorithm: str
    data_type: str
    n_samples: int
    n_variables: int
    noise_level: float
    repetition: int
    execution_time: float
    metrics: Dict[str, float]
    causal_result: CausalResult
    ground_truth: np.ndarray
    success: bool
    error_message: Optional[str] = None


class CausalDiscoveryEvaluator:
    """Comprehensive evaluation framework for causal discovery algorithms."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.data_processor = DataProcessor()
        self.results_history = []
        
        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations."""
        print("🚀 Starting comprehensive causal discovery benchmark...")
        print(f"Configuration: {len(self.config.algorithms)} algorithms, "
              f"{len(self.config.data_types)} data types, "
              f"{len(self.config.sample_sizes)} sample sizes")
        
        all_experiments = self._generate_experiment_configs()
        print(f"Total experiments: {len(all_experiments)}")
        
        start_time = time.time()
        
        if self.config.parallel_execution:
            results = self._run_experiments_parallel(all_experiments)
        else:
            results = self._run_experiments_sequential(all_experiments)
        
        total_time = time.time() - start_time
        print(f"✅ Benchmark completed in {total_time:.2f} seconds")
        
        # Convert to DataFrame
        results_df = self._results_to_dataframe(results)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(results_df, results)
        
        # Generate summary report
        self._generate_summary_report(results_df)
        
        return results_df
    
    def evaluate_single_algorithm(self, algorithm: CausalDiscoveryModel,
                                 data: pd.DataFrame, 
                                 ground_truth: np.ndarray,
                                 algorithm_name: str = "Custom") -> Dict[str, Any]:
        """Evaluate a single algorithm on given data."""
        start_time = time.time()
        
        try:
            # Run algorithm
            result = algorithm.fit_discover(data)
            execution_time = time.time() - start_time
            
            # Evaluate metrics
            metrics = CausalMetrics.evaluate_discovery(
                ground_truth, result.adjacency_matrix, result.confidence_scores
            )
            
            return {
                'algorithm': algorithm_name,
                'execution_time': execution_time,
                'metrics': metrics,
                'result': result,
                'success': True
            }
            
        except Exception as e:
            return {
                'algorithm': algorithm_name,
                'execution_time': time.time() - start_time,
                'metrics': {},
                'result': None,
                'success': False,
                'error': str(e)
            }
    
    def compare_algorithms(self, algorithms: Dict[str, CausalDiscoveryModel],
                          data: pd.DataFrame, 
                          ground_truth: np.ndarray) -> pd.DataFrame:
        """Compare multiple algorithms on the same dataset."""
        results = []
        
        for name, algorithm in algorithms.items():
            print(f"Evaluating {name}...")
            result = self.evaluate_single_algorithm(algorithm, data, ground_truth, name)
            results.append(result)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            if result['success']:
                row = {
                    'Algorithm': result['algorithm'],
                    'Execution Time (s)': result['execution_time'],
                    **result['metrics']
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Generate comparison plots
        self._plot_algorithm_comparison(comparison_df)
        
        return comparison_df
    
    def analyze_scalability(self, algorithm: CausalDiscoveryModel,
                           sample_sizes: List[int] = None,
                           n_variables_list: List[int] = None) -> pd.DataFrame:
        """Analyze algorithm scalability across different data sizes."""
        if sample_sizes is None:
            sample_sizes = [100, 200, 500, 1000, 2000]
        if n_variables_list is None:
            n_variables_list = [5, 10, 20, 50]
        
        scalability_results = []
        
        for n_vars in n_variables_list:
            for n_samples in sample_sizes:
                print(f"Testing scalability: {n_vars} variables, {n_samples} samples")
                
                # Generate synthetic data
                data = self.data_processor.generate_synthetic_data(
                    n_samples=n_samples,
                    n_variables=n_vars,
                    noise_level=0.2
                )
                
                # Create ground truth (simple chain structure)
                ground_truth = np.zeros((n_vars, n_vars))
                for i in range(n_vars - 1):
                    ground_truth[i, i + 1] = 1
                
                # Evaluate algorithm
                start_time = time.time()
                try:
                    result = algorithm.fit_discover(data)
                    execution_time = time.time() - start_time
                    
                    metrics = CausalMetrics.evaluate_discovery(
                        ground_truth, result.adjacency_matrix
                    )
                    
                    scalability_results.append({
                        'n_variables': n_vars,
                        'n_samples': n_samples,
                        'execution_time': execution_time,
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'success': True
                    })
                    
                except Exception as e:
                    scalability_results.append({
                        'n_variables': n_vars,
                        'n_samples': n_samples,
                        'execution_time': float('inf'),
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'success': False,
                        'error': str(e)
                    })
        
        scalability_df = pd.DataFrame(scalability_results)
        
        # Generate scalability plots
        self._plot_scalability_analysis(scalability_df)
        
        return scalability_df
    
    def _generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations."""
        experiments = []
        
        for algorithm in self.config.algorithms:
            for data_type in self.config.data_types:
                for n_samples in self.config.sample_sizes:
                    for n_vars in self.config.n_variables_list:
                        for noise_level in self.config.noise_levels:
                            for rep in range(self.config.n_repetitions):
                                experiments.append({
                                    'algorithm': algorithm,
                                    'data_type': data_type,
                                    'n_samples': n_samples,
                                    'n_variables': n_vars,
                                    'noise_level': noise_level,
                                    'repetition': rep
                                })
        
        return experiments
    
    def _run_experiments_parallel(self, experiments: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp
                for exp in experiments
            }
            
            for i, future in enumerate(future_to_exp):
                try:
                    result = future.result(timeout=self.config.timeout_per_run)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Completed {i + 1}/{len(experiments)} experiments")
                        
                except Exception as e:
                    exp = future_to_exp[future]
                    print(f"Experiment failed: {exp['algorithm']} on {exp['data_type']} data: {e}")
        
        return results
    
    def _run_experiments_sequential(self, experiments: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, exp in enumerate(experiments):
            try:
                result = self._run_single_experiment(exp)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(experiments)} experiments")
                    
            except Exception as e:
                print(f"Experiment failed: {exp['algorithm']} on {exp['data_type']} data: {e}")
        
        return results
    
    def _run_single_experiment(self, exp_config: Dict[str, Any]) -> ExperimentResult:
        """Run a single benchmark experiment."""
        # Generate synthetic data
        data, ground_truth = self._generate_benchmark_data(
            data_type=exp_config['data_type'],
            n_samples=exp_config['n_samples'],
            n_variables=exp_config['n_variables'],
            noise_level=exp_config['noise_level'],
            random_seed=exp_config['repetition']
        )
        
        # Initialize algorithm
        algorithm = self._initialize_algorithm(exp_config['algorithm'])
        
        # Run experiment
        start_time = time.time()
        try:
            result = algorithm.fit_discover(data)
            execution_time = time.time() - start_time
            
            # Evaluate metrics
            metrics = CausalMetrics.evaluate_discovery(
                ground_truth, result.adjacency_matrix, result.confidence_scores
            )
            
            return ExperimentResult(
                algorithm=exp_config['algorithm'],
                data_type=exp_config['data_type'],
                n_samples=exp_config['n_samples'],
                n_variables=exp_config['n_variables'],
                noise_level=exp_config['noise_level'],
                repetition=exp_config['repetition'],
                execution_time=execution_time,
                metrics=metrics,
                causal_result=result,
                ground_truth=ground_truth,
                success=True
            )
            
        except Exception as e:
            return ExperimentResult(
                algorithm=exp_config['algorithm'],
                data_type=exp_config['data_type'],
                n_samples=exp_config['n_samples'],
                n_variables=exp_config['n_variables'],
                noise_level=exp_config['noise_level'],
                repetition=exp_config['repetition'],
                execution_time=time.time() - start_time,
                metrics={},
                causal_result=None,
                ground_truth=ground_truth,
                success=False,
                error_message=str(e)
            )
    
    def _generate_benchmark_data(self, data_type: str, n_samples: int, 
                                n_variables: int, noise_level: float,
                                random_seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate benchmark data with known ground truth."""
        np.random.seed(random_seed)
        
        if data_type == 'linear':
            return self._generate_linear_data(n_samples, n_variables, noise_level)
        elif data_type == 'nonlinear':
            return self._generate_nonlinear_data(n_samples, n_variables, noise_level)
        elif data_type == 'mixed':
            return self._generate_mixed_data(n_samples, n_variables, noise_level)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _generate_linear_data(self, n_samples: int, n_variables: int, 
                             noise_level: float) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate linear causal data."""
        # Create chain structure: X1 -> X2 -> X3 -> ...
        ground_truth = np.zeros((n_variables, n_variables))
        
        data = np.zeros((n_samples, n_variables))
        data[:, 0] = np.random.randn(n_samples)
        
        for i in range(1, n_variables):
            ground_truth[i-1, i] = 1
            # Linear relationship with noise
            coefficient = np.random.uniform(0.5, 1.5)
            data[:, i] = coefficient * data[:, i-1] + noise_level * np.random.randn(n_samples)
        
        columns = [f'X{i+1}' for i in range(n_variables)]
        df = pd.DataFrame(data, columns=columns)
        
        return df, ground_truth
    
    def _generate_nonlinear_data(self, n_samples: int, n_variables: int,
                                noise_level: float) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate nonlinear causal data."""
        ground_truth = np.zeros((n_variables, n_variables))
        
        data = np.zeros((n_samples, n_variables))
        data[:, 0] = np.random.randn(n_samples)
        
        for i in range(1, n_variables):
            ground_truth[i-1, i] = 1
            # Nonlinear relationship
            data[:, i] = np.tanh(data[:, i-1]) + noise_level * np.random.randn(n_samples)
        
        columns = [f'X{i+1}' for i in range(n_variables)]
        df = pd.DataFrame(data, columns=columns)
        
        return df, ground_truth
    
    def _generate_mixed_data(self, n_samples: int, n_variables: int,
                            noise_level: float) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate mixed linear/nonlinear causal data."""
        ground_truth = np.zeros((n_variables, n_variables))
        
        data = np.zeros((n_samples, n_variables))
        data[:, 0] = np.random.randn(n_samples)
        
        for i in range(1, n_variables):
            ground_truth[i-1, i] = 1
            # Randomly choose linear or nonlinear
            if np.random.random() < 0.5:
                # Linear
                coefficient = np.random.uniform(0.5, 1.5)
                data[:, i] = coefficient * data[:, i-1] + noise_level * np.random.randn(n_samples)
            else:
                # Nonlinear
                data[:, i] = np.sin(data[:, i-1]) + noise_level * np.random.randn(n_samples)
        
        columns = [f'X{i+1}' for i in range(n_variables)]
        df = pd.DataFrame(data, columns=columns)
        
        return df, ground_truth
    
    def _initialize_algorithm(self, algorithm_name: str) -> CausalDiscoveryModel:
        """Initialize algorithm by name."""
        pipeline_config = PipelineConfig(algorithms=[algorithm_name])
        pipeline = CausalDiscoveryPipeline(pipeline_config)
        return pipeline.algorithms[algorithm_name]
    
    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Convert experiment results to DataFrame."""
        data = []
        
        for result in results:
            row = {
                'algorithm': result.algorithm,
                'data_type': result.data_type,
                'n_samples': result.n_samples,
                'n_variables': result.n_variables,
                'noise_level': result.noise_level,
                'repetition': result.repetition,
                'execution_time': result.execution_time,
                'success': result.success
            }
            
            # Add metrics if successful
            if result.success and result.metrics:
                row.update(result.metrics)
            else:
                # Add default metrics for failed runs
                default_metrics = {
                    'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                    'structural_hamming_distance': float('inf'),
                    'true_positive_rate': 0.0, 'false_positive_rate': 1.0
                }
                row.update(default_metrics)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_results(self, results_df: pd.DataFrame, raw_results: List[ExperimentResult]):
        """Save benchmark results."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame
        csv_path = self.output_path / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save configuration
        config_path = self.output_path / f"benchmark_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        print(f"Results saved to {csv_path}")
        print(f"Configuration saved to {config_path}")
    
    def _generate_summary_report(self, results_df: pd.DataFrame):
        """Generate summary report with visualizations."""
        print("\n📊 BENCHMARK SUMMARY REPORT")
        print("=" * 50)
        
        # Overall statistics
        total_experiments = len(results_df)
        success_rate = results_df['success'].mean() * 100
        avg_execution_time = results_df[results_df['success']]['execution_time'].mean()
        
        print(f"Total experiments: {total_experiments}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average execution time: {avg_execution_time:.3f}s")
        
        # Performance by algorithm
        print("\n🎯 Performance by Algorithm:")
        algo_performance = results_df[results_df['success']].groupby('algorithm').agg({
            'f1_score': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': 'count'
        }).round(3)
        print(algo_performance)
        
        # Generate plots
        self._plot_benchmark_results(results_df)
        
        print(f"\n📈 Visualizations saved to {self.output_path}")
    
    def _plot_benchmark_results(self, results_df: pd.DataFrame):
        """Generate comprehensive benchmark visualization plots."""
        successful_results = results_df[results_df['success']]
        
        if len(successful_results) == 0:
            print("No successful results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Algorithm performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Causal Discovery Algorithm Benchmark Results', fontsize=16, fontweight='bold')
        
        # F1 Score by Algorithm
        sns.boxplot(data=successful_results, x='algorithm', y='f1_score', ax=axes[0, 0])
        axes[0, 0].set_title('F1 Score Distribution by Algorithm')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Execution Time by Algorithm
        sns.boxplot(data=successful_results, x='algorithm', y='execution_time', ax=axes[0, 1])
        axes[0, 1].set_title('Execution Time Distribution by Algorithm')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Performance vs Data Size
        sns.scatterplot(data=successful_results, x='n_samples', y='f1_score', 
                       hue='algorithm', ax=axes[1, 0])
        axes[1, 0].set_title('F1 Score vs Sample Size')
        
        # Performance vs Noise Level
        sns.lineplot(data=successful_results, x='noise_level', y='f1_score', 
                    hue='algorithm', ax=axes[1, 1])
        axes[1, 1].set_title('F1 Score vs Noise Level')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'benchmark_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scalability analysis
        self._plot_scalability_heatmap(successful_results)
        
        # 3. Data type performance
        self._plot_data_type_performance(successful_results)
    
    def _plot_scalability_heatmap(self, results_df: pd.DataFrame):
        """Plot scalability heatmap."""
        fig, axes = plt.subplots(1, len(results_df['algorithm'].unique()), 
                                figsize=(5 * len(results_df['algorithm'].unique()), 5))
        
        if len(results_df['algorithm'].unique()) == 1:
            axes = [axes]
        
        for i, algorithm in enumerate(results_df['algorithm'].unique()):
            algo_data = results_df[results_df['algorithm'] == algorithm]
            
            # Create pivot table for heatmap
            heatmap_data = algo_data.pivot_table(
                values='f1_score', 
                index='n_variables', 
                columns='n_samples',
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[i])
            axes[i].set_title(f'F1 Score Heatmap - {algorithm}')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'scalability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_type_performance(self, results_df: pd.DataFrame):
        """Plot performance by data type."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1 Score by data type
        sns.boxplot(data=results_df, x='data_type', y='f1_score', hue='algorithm', ax=axes[0])
        axes[0].set_title('F1 Score by Data Type')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Execution time by data type
        sns.boxplot(data=results_df, x='data_type', y='execution_time', hue='algorithm', ax=axes[1])
        axes[1].set_title('Execution Time by Data Type')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'data_type_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_algorithm_comparison(self, comparison_df: pd.DataFrame):
        """Plot algorithm comparison results."""
        metrics_to_plot = ['f1_score', 'precision', 'recall', 'execution_time']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            comparison_df.plot(x='Algorithm', y=metric, kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, scalability_df: pd.DataFrame):
        """Plot scalability analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Execution time vs variables
        successful_results = scalability_df[scalability_df['success']]
        
        if len(successful_results) > 0:
            sns.lineplot(data=successful_results, x='n_variables', y='execution_time', 
                        hue='n_samples', ax=axes[0, 0])
            axes[0, 0].set_title('Execution Time vs Number of Variables')
            
            # F1 score vs variables
            sns.lineplot(data=successful_results, x='n_variables', y='f1_score', 
                        hue='n_samples', ax=axes[0, 1])
            axes[0, 1].set_title('F1 Score vs Number of Variables')
            
            # Execution time vs samples
            sns.lineplot(data=successful_results, x='n_samples', y='execution_time', 
                        hue='n_variables', ax=axes[1, 0])
            axes[1, 0].set_title('Execution Time vs Sample Size')
            
            # F1 score vs samples
            sns.lineplot(data=successful_results, x='n_samples', y='f1_score', 
                        hue='n_variables', ax=axes[1, 1])
            axes[1, 1].set_title('F1 Score vs Sample Size')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()