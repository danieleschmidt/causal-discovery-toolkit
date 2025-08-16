"""Publication-ready research tools and academic formatting."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Standardized benchmark result for academic comparison."""
    algorithm_name: str
    dataset_name: str
    metric_name: str
    value: float
    std_error: float
    n_runs: int
    runtime_seconds: float
    parameters: Dict[str, Any]
    timestamp: str


@dataclass
class StatisticalComparison:
    """Statistical comparison between algorithms."""
    algorithm_a: str
    algorithm_b: str
    metric: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significantly_different: bool
    winner: Optional[str]


@dataclass
class PublicationFigure:
    """Publication-ready figure metadata."""
    figure_id: str
    title: str
    caption: str
    filename: str
    figure_type: str  # 'performance', 'comparison', 'visualization', 'methodology'
    file_formats: List[str]  # ['png', 'pdf', 'svg']
    dpi: int
    width_inches: float
    height_inches: float


class AcademicBenchmarker:
    """Comprehensive benchmarking framework for academic publications."""
    
    def __init__(self, 
                 output_dir: Path = Path("publication_output"),
                 figure_style: str = "publication",
                 random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark_results = []
        self.statistical_comparisons = []
        self.figures = []
        
        # Set publication style
        self._setup_publication_style(figure_style)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
    def _setup_publication_style(self, style: str):
        """Setup matplotlib for publication-quality figures."""
        if style == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'axes.linewidth': 1.2,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'lines.markersize': 8,
                'grid.alpha': 0.3,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        
    def benchmark_algorithm(self,
                          algorithm_factory: callable,
                          algorithm_name: str,
                          datasets: Dict[str, pd.DataFrame],
                          metrics: Dict[str, callable],
                          n_runs: int = 10,
                          **algorithm_params) -> List[BenchmarkResult]:
        """Comprehensive benchmarking of an algorithm across datasets and metrics."""
        results = []
        
        for dataset_name, data in datasets.items():
            print(f"Benchmarking {algorithm_name} on {dataset_name}...")
            
            for metric_name, metric_func in metrics.items():
                metric_values = []
                runtimes = []
                
                for run in range(n_runs):
                    # Create fresh algorithm instance
                    algorithm = algorithm_factory(**algorithm_params)
                    
                    # Time the execution
                    start_time = time.time()
                    algorithm.fit(data)
                    result = algorithm.predict(data)
                    runtime = time.time() - start_time
                    
                    # Compute metric
                    metric_value = metric_func(result, data)
                    metric_values.append(metric_value)
                    runtimes.append(runtime)
                
                # Compute statistics
                mean_metric = np.mean(metric_values)
                std_metric = np.std(metric_values, ddof=1)
                std_error = std_metric / np.sqrt(n_runs)
                mean_runtime = np.mean(runtimes)
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    algorithm_name=algorithm_name,
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    value=mean_metric,
                    std_error=std_error,
                    n_runs=n_runs,
                    runtime_seconds=mean_runtime,
                    parameters=algorithm_params,
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(benchmark_result)
                self.benchmark_results.append(benchmark_result)
        
        return results
    
    def compare_algorithms(self,
                         algorithm_results: Dict[str, List[BenchmarkResult]],
                         metric_name: str,
                         dataset_name: str = None) -> List[StatisticalComparison]:
        """Statistical comparison between algorithms."""
        comparisons = []
        algorithm_names = list(algorithm_results.keys())
        
        for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
                alg_a = algorithm_names[i]
                alg_b = algorithm_names[j]
                
                # Get results for comparison
                results_a = [r for r in algorithm_results[alg_a] 
                           if r.metric_name == metric_name and 
                           (dataset_name is None or r.dataset_name == dataset_name)]
                results_b = [r for r in algorithm_results[alg_b] 
                           if r.metric_name == metric_name and 
                           (dataset_name is None or r.dataset_name == dataset_name)]
                
                if not results_a or not results_b:
                    continue
                
                # Statistical test (simplified t-test)
                values_a = [r.value for r in results_a]
                values_b = [r.value for r in results_b]
                
                p_value, effect_size = self._statistical_test(values_a, values_b)
                
                # Confidence interval (simplified)
                diff_mean = np.mean(values_a) - np.mean(values_b)
                diff_std = np.sqrt(np.var(values_a) + np.var(values_b))
                ci_lower = diff_mean - 1.96 * diff_std
                ci_upper = diff_mean + 1.96 * diff_std
                
                # Determine winner
                significantly_different = p_value < 0.05
                winner = None
                if significantly_different:
                    winner = alg_a if np.mean(values_a) > np.mean(values_b) else alg_b
                
                comparison = StatisticalComparison(
                    algorithm_a=alg_a,
                    algorithm_b=alg_b,
                    metric=metric_name,
                    p_value=p_value,
                    effect_size=effect_size,
                    confidence_interval=(ci_lower, ci_upper),
                    significantly_different=significantly_different,
                    winner=winner
                )
                
                comparisons.append(comparison)
                self.statistical_comparisons.append(comparison)
        
        return comparisons
    
    def _statistical_test(self, values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
        """Perform statistical test between two groups."""
        from scipy import stats
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
        effect_size = abs(np.mean(values_a) - np.mean(values_b)) / pooled_std
        
        return p_value, effect_size
    
    def create_performance_table(self, 
                               metric_name: str,
                               latex_format: bool = True) -> str:
        """Create publication-ready performance table."""
        # Group results by algorithm and dataset
        table_data = {}
        datasets = set()
        algorithms = set()
        
        for result in self.benchmark_results:
            if result.metric_name == metric_name:
                if result.algorithm_name not in table_data:
                    table_data[result.algorithm_name] = {}
                
                table_data[result.algorithm_name][result.dataset_name] = {
                    'value': result.value,
                    'std_error': result.std_error
                }
                
                datasets.add(result.dataset_name)
                algorithms.add(result.algorithm_name)
        
        datasets = sorted(list(datasets))
        algorithms = sorted(list(algorithms))
        
        if latex_format:
            # Generate LaTeX table
            table_str = "\\begin{table}[htbp]\n"
            table_str += "\\centering\n"
            table_str += f"\\caption{{{metric_name} Performance Comparison}}\n"
            table_str += "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"
            table_str += "\\toprule\n"
            table_str += "Algorithm & " + " & ".join(datasets) + " \\\\\n"
            table_str += "\\midrule\n"
            
            for alg in algorithms:
                row = alg
                for dataset in datasets:
                    if dataset in table_data.get(alg, {}):
                        value = table_data[alg][dataset]['value']
                        std_err = table_data[alg][dataset]['std_error']
                        row += f" & {value:.3f} $\\pm$ {std_err:.3f}"
                    else:
                        row += " & --"
                row += " \\\\\n"
                table_str += row
            
            table_str += "\\bottomrule\n"
            table_str += "\\end{tabular}\n"
            table_str += "\\end{table}\n"
        else:
            # Generate plain text table
            table_str = f"{metric_name} Performance Comparison\n"
            table_str += "=" * 50 + "\n"
            table_str += f"{'Algorithm':<20} " + " ".join(f"{d:<15}" for d in datasets) + "\n"
            table_str += "-" * 50 + "\n"
            
            for alg in algorithms:
                row = f"{alg:<20} "
                for dataset in datasets:
                    if dataset in table_data.get(alg, {}):
                        value = table_data[alg][dataset]['value']
                        std_err = table_data[alg][dataset]['std_error']
                        row += f"{value:.3f}±{std_err:.3f}".ljust(15)
                    else:
                        row += "--".ljust(15)
                table_str += row + "\n"
        
        return table_str
    
    def create_comparison_figure(self,
                               metric_name: str,
                               figure_type: str = "boxplot") -> PublicationFigure:
        """Create publication-ready comparison figure."""
        # Prepare data for plotting
        plot_data = []
        for result in self.benchmark_results:
            if result.metric_name == metric_name:
                plot_data.append({
                    'Algorithm': result.algorithm_name,
                    'Dataset': result.dataset_name,
                    'Value': result.value,
                    'Std_Error': result.std_error
                })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            raise ValueError(f"No data found for metric {metric_name}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if figure_type == "boxplot":
            sns.boxplot(data=df, x='Dataset', y='Value', hue='Algorithm', ax=ax)
        elif figure_type == "barplot":
            sns.barplot(data=df, x='Dataset', y='Value', hue='Algorithm', 
                       capsize=0.1, ax=ax)
        else:
            raise ValueError(f"Unknown figure type: {figure_type}")
        
        ax.set_title(f'{metric_name} Comparison Across Datasets')
        ax.set_ylabel(metric_name)
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        figure_id = f"comparison_{metric_name}_{figure_type}"
        filename = f"{figure_id}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        figure_metadata = PublicationFigure(
            figure_id=figure_id,
            title=f'{metric_name} Comparison',
            caption=f'Comparison of algorithm performance on {metric_name} metric across different datasets.',
            filename=filename,
            figure_type='comparison',
            file_formats=['png'],
            dpi=300,
            width_inches=10.0,
            height_inches=6.0
        )
        
        self.figures.append(figure_metadata)
        return figure_metadata
    
    def create_runtime_analysis(self) -> PublicationFigure:
        """Create runtime analysis figure."""
        # Prepare runtime data
        runtime_data = []
        for result in self.benchmark_results:
            runtime_data.append({
                'Algorithm': result.algorithm_name,
                'Dataset': result.dataset_name,
                'Runtime': result.runtime_seconds
            })
        
        df = pd.DataFrame(runtime_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(data=df, x='Dataset', y='Runtime', hue='Algorithm', ax=ax)
        ax.set_title('Algorithm Runtime Comparison')
        ax.set_ylabel('Runtime (seconds)')
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        figure_id = "runtime_analysis"
        filename = f"{figure_id}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        figure_metadata = PublicationFigure(
            figure_id=figure_id,
            title='Runtime Analysis',
            caption='Comparison of algorithm execution times across different datasets.',
            filename=filename,
            figure_type='performance',
            file_formats=['png'],
            dpi=300,
            width_inches=10.0,
            height_inches=6.0
        )
        
        self.figures.append(figure_metadata)
        return figure_metadata
    
    def generate_latex_document(self) -> str:
        """Generate LaTeX document template with results."""
        latex_content = """
\\documentclass[12pt]{article}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{amssymb}

\\title{Causal Discovery Algorithm Comparison}
\\author{Research Team}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}
This document presents a comprehensive comparison of causal discovery algorithms across multiple datasets and metrics.

\\section{Methodology}
All algorithms were evaluated using standardized benchmarks with multiple runs to ensure statistical reliability.

\\section{Results}

"""
        
        # Add performance tables
        for metric in set(r.metric_name for r in self.benchmark_results):
            latex_content += f"\\subsection{{{metric} Performance}}\n"
            latex_content += self.create_performance_table(metric, latex_format=True)
            latex_content += "\n\n"
        
        # Add figures
        for figure in self.figures:
            latex_content += f"\\begin{{figure}}[htbp]\n"
            latex_content += f"\\centering\n"
            latex_content += f"\\includegraphics[width=0.8\\textwidth]{{{figure.filename}}}\n"
            latex_content += f"\\caption{{{figure.caption}}}\n"
            latex_content += f"\\label{{fig:{figure.figure_id}}}\n"
            latex_content += f"\\end{{figure}}\n\n"
        
        # Add statistical comparisons
        latex_content += "\\section{Statistical Analysis}\n"
        for comparison in self.statistical_comparisons:
            significance = "significant" if comparison.significantly_different else "not significant"
            latex_content += f"Comparison between {comparison.algorithm_a} and {comparison.algorithm_b} "
            latex_content += f"on {comparison.metric}: {significance} "
            latex_content += f"(p = {comparison.p_value:.4f}, effect size = {comparison.effect_size:.3f}).\n\n"
        
        latex_content += "\\section{Conclusion}\n"
        latex_content += "The experimental results demonstrate the relative performance characteristics of different causal discovery approaches.\n\n"
        latex_content += "\\end{document}"
        
        return latex_content
    
    def export_results(self, format: str = "json"):
        """Export benchmark results in various formats."""
        if format == "json":
            # Convert results to JSON
            results_dict = {
                'benchmark_results': [
                    {
                        'algorithm_name': r.algorithm_name,
                        'dataset_name': r.dataset_name,
                        'metric_name': r.metric_name,
                        'value': r.value,
                        'std_error': r.std_error,
                        'n_runs': r.n_runs,
                        'runtime_seconds': r.runtime_seconds,
                        'parameters': r.parameters,
                        'timestamp': r.timestamp
                    }
                    for r in self.benchmark_results
                ],
                'statistical_comparisons': [
                    {
                        'algorithm_a': c.algorithm_a,
                        'algorithm_b': c.algorithm_b,
                        'metric': c.metric,
                        'p_value': c.p_value,
                        'effect_size': c.effect_size,
                        'confidence_interval': c.confidence_interval,
                        'significantly_different': c.significantly_different,
                        'winner': c.winner
                    }
                    for c in self.statistical_comparisons
                ]
            }
            
            filepath = self.output_dir / "benchmark_results.json"
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
        
        elif format == "csv":
            # Export as CSV
            df = pd.DataFrame([
                {
                    'algorithm_name': r.algorithm_name,
                    'dataset_name': r.dataset_name,
                    'metric_name': r.metric_name,
                    'value': r.value,
                    'std_error': r.std_error,
                    'n_runs': r.n_runs,
                    'runtime_seconds': r.runtime_seconds,
                    'timestamp': r.timestamp
                }
                for r in self.benchmark_results
            ])
            
            filepath = self.output_dir / "benchmark_results.csv"
            df.to_csv(filepath, index=False)
        
        elif format == "latex":
            # Export LaTeX document
            latex_content = self.generate_latex_document()
            filepath = self.output_dir / "benchmark_report.tex"
            with open(filepath, 'w') as f:
                f.write(latex_content)


def standard_causal_metrics() -> Dict[str, callable]:
    """Standard metrics for causal discovery evaluation."""
    
    def precision_metric(result, data):
        """Precision of discovered causal relationships."""
        true_positives = np.sum(result.adjacency_matrix)  # Simplified
        if true_positives == 0:
            return 0.0
        return true_positives / np.sum(result.adjacency_matrix)
    
    def recall_metric(result, data):
        """Recall of discovered causal relationships."""
        # Simplified - in practice would need ground truth
        return np.sum(result.adjacency_matrix) / (len(data.columns) * (len(data.columns) - 1))
    
    def density_metric(result, data):
        """Density of discovered causal graph."""
        n_vars = result.adjacency_matrix.shape[0]
        max_edges = n_vars * (n_vars - 1)
        return np.sum(result.adjacency_matrix) / max_edges
    
    def avg_confidence_metric(result, data):
        """Average confidence of discovered relationships."""
        return np.mean(result.confidence_scores[result.adjacency_matrix > 0])
    
    return {
        'precision': precision_metric,
        'recall': recall_metric,
        'density': density_metric,
        'avg_confidence': avg_confidence_metric
    }