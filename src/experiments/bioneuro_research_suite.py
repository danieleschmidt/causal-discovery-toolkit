"""
Comprehensive research suite for bioneuro-olfactory fusion studies.
Includes automated experiments, benchmarking, and result validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
import json
from datetime import datetime
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import warnings

try:
    from ..algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel, MultiModalOlfactoryCausalModel, BioneuroFusionResult
    from ..utils.bioneuro_data_processing import BioneuroDataProcessor, OlfactoryFeatureExtractor, OlfactoryDataProcessingConfig
    from ..utils.monitoring import monitor_performance, PerformanceMonitor
    from ..utils.validation import DataValidator
    from ..utils.metrics import CausalMetrics
    from .benchmark import CausalBenchmark
except ImportError:
    # For direct execution
    try:
        from algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel, MultiModalOlfactoryCausalModel, BioneuroFusionResult
        from utils.bioneuro_data_processing import BioneuroDataProcessor, OlfactoryFeatureExtractor, OlfactoryDataProcessingConfig
        from utils.monitoring import monitor_performance, PerformanceMonitor
        from utils.validation import DataValidator
        from utils.metrics import CausalMetrics
        from experiments.benchmark import CausalBenchmark
    except ImportError:
        # Minimal imports for standalone operation
        import sys
        sys.path.append('..')
        from algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel, MultiModalOlfactoryCausalModel, BioneuroFusionResult
        from utils.bioneuro_data_processing import BioneuroDataProcessor, OlfactoryFeatureExtractor, OlfactoryDataProcessingConfig
        from utils.monitoring import monitor_performance, PerformanceMonitor
        from utils.validation import DataValidator
        from utils.metrics import CausalMetrics
        # Mock CausalBenchmark if not available
        CausalBenchmark = None

logger = logging.getLogger(__name__)


@dataclass
class ResearchExperimentConfig:
    """Configuration for bioneuro research experiments."""
    experiment_name: str
    n_samples_range: List[int] = None
    n_receptors_range: List[int] = None
    n_neurons_range: List[int] = None
    noise_levels: List[float] = None
    receptor_thresholds: List[float] = None
    neural_thresholds: List[float] = None
    temporal_windows: List[int] = None
    n_runs_per_condition: int = 5
    enable_multimodal: bool = True
    enable_benchmarking: bool = True
    save_results: bool = True
    output_dir: str = "research_results"
    random_seed: int = 42

    def __post_init__(self):
        """Set default values for None fields."""
        if self.n_samples_range is None:
            self.n_samples_range = [100, 500, 1000]
        if self.n_receptors_range is None:
            self.n_receptors_range = [3, 6, 10]
        if self.n_neurons_range is None:
            self.n_neurons_range = [2, 5, 8]
        if self.noise_levels is None:
            self.noise_levels = [0.05, 0.1, 0.2]
        if self.receptor_thresholds is None:
            self.receptor_thresholds = [0.1, 0.15, 0.2]
        if self.neural_thresholds is None:
            self.neural_thresholds = [5.0, 10.0, 15.0]
        if self.temporal_windows is None:
            self.temporal_windows = [50, 100, 200]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    causal_result: BioneuroFusionResult
    performance_metrics: Dict[str, float]
    data_quality_metrics: Dict[str, float]
    statistical_validation: Dict[str, Any]
    execution_time: float
    memory_usage: float


class BioneuroResearchSuite:
    """Comprehensive research suite for bioneuro-olfactory studies."""
    
    def __init__(self, config: ResearchExperimentConfig):
        """
        Initialize the research suite.
        
        Args:
            config: Configuration for experiments
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        self.performance_monitor = PerformanceMonitor()
        self.validator = DataValidator()
        self.metrics = CausalMetrics()
        self.benchmark = CausalBenchmark() if CausalBenchmark else None
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized BioneuroResearchSuite for experiment: {config.experiment_name}")
    
    @monitor_performance()
    def run_comprehensive_study(self) -> Dict[str, Any]:
        """
        Run comprehensive research study with all configured experiments.
        
        Returns:
            Dictionary containing study results and analytics
        """
        logger.info("Starting comprehensive bioneuro research study")
        start_time = time.time()
        
        study_results = {
            "study_metadata": {
                "experiment_name": self.config.experiment_name,
                "start_time": datetime.now().isoformat(),
                "config": asdict(self.config)
            },
            "experiment_results": [],
            "aggregate_analytics": {},
            "statistical_summaries": {},
            "performance_analysis": {}
        }
        
        # Generate experiment conditions
        experiment_conditions = self._generate_experiment_conditions()
        total_experiments = len(experiment_conditions)
        
        logger.info(f"Generated {total_experiments} experiment conditions")
        
        # Run experiments in parallel for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_condition = {
                executor.submit(self._run_single_experiment, condition, i): condition
                for i, condition in enumerate(experiment_conditions)
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_condition):
                condition = future_to_condition[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
                    study_results["experiment_results"].append(asdict(result))
                    
                    if completed % 10 == 0:
                        logger.info(f"Completed {completed}/{total_experiments} experiments")
                        
                except Exception as e:
                    logger.error(f"Experiment failed for condition {condition}: {str(e)}")
        
        # Compute aggregate analytics
        study_results["aggregate_analytics"] = self._compute_aggregate_analytics()
        study_results["statistical_summaries"] = self._compute_statistical_summaries()
        study_results["performance_analysis"] = self._analyze_performance_trends()
        
        # Generate visualizations
        if len(self.results) > 0:
            self._generate_research_visualizations()
        
        # Save results
        if self.config.save_results:
            self._save_study_results(study_results)
        
        execution_time = time.time() - start_time
        study_results["study_metadata"]["execution_time"] = execution_time
        study_results["study_metadata"]["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Completed comprehensive study in {execution_time:.2f} seconds")
        return study_results
    
    def _generate_experiment_conditions(self) -> List[Dict[str, Any]]:
        """Generate all experimental conditions to test."""
        conditions = []
        
        for n_samples in self.config.n_samples_range:
            for n_receptors in self.config.n_receptors_range:
                for n_neurons in self.config.n_neurons_range:
                    for noise_level in self.config.noise_levels:
                        for receptor_thresh in self.config.receptor_thresholds:
                            for neural_thresh in self.config.neural_thresholds:
                                for temporal_window in self.config.temporal_windows:
                                    for run_id in range(self.config.n_runs_per_condition):
                                        conditions.append({
                                            "n_samples": n_samples,
                                            "n_receptors": n_receptors,
                                            "n_neurons": n_neurons,
                                            "noise_level": noise_level,
                                            "receptor_threshold": receptor_thresh,
                                            "neural_threshold": neural_thresh,
                                            "temporal_window": temporal_window,
                                            "run_id": run_id,
                                            "random_seed": self.config.random_seed + run_id
                                        })
        
        return conditions
    
    def _run_single_experiment(self, condition: Dict[str, Any], experiment_idx: int) -> ExperimentResult:
        """Run a single experiment with given conditions."""
        experiment_id = f"{self.config.experiment_name}_{experiment_idx:04d}"
        start_time = time.time()
        
        # Generate synthetic data
        data = self._generate_experimental_data(**condition)
        
        # Initialize models
        olfactory_model = OlfactoryNeuralCausalModel(
            receptor_sensitivity_threshold=condition["receptor_threshold"],
            neural_firing_threshold=condition["neural_threshold"],
            temporal_window_ms=condition["temporal_window"],
            cross_modal_integration=True,
            bootstrap_samples=100,
            confidence_level=0.95
        )
        
        # Run causal discovery
        causal_result = olfactory_model.fit_discover(data)
        
        # Compute performance metrics
        performance_metrics = self._compute_performance_metrics(causal_result, data)
        
        # Data quality assessment
        data_quality_metrics = self._assess_data_quality(data)
        
        # Statistical validation
        statistical_validation = self._perform_statistical_validation(causal_result, data)
        
        execution_time = time.time() - start_time
        memory_usage = self.performance_monitor.get_memory_usage()
        
        return ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config=condition,
            causal_result=causal_result,
            performance_metrics=performance_metrics,
            data_quality_metrics=data_quality_metrics,
            statistical_validation=statistical_validation,
            execution_time=execution_time,
            memory_usage=memory_usage
        )
    
    def _generate_experimental_data(self, **kwargs) -> pd.DataFrame:
        """Generate experimental data with specified parameters."""
        np.random.seed(kwargs.get("random_seed", 42))
        
        n_samples = kwargs["n_samples"]
        n_receptors = kwargs["n_receptors"]
        n_neurons = kwargs["n_neurons"]
        noise_level = kwargs["noise_level"]
        
        # Time vector
        dt = 0.001  # 1ms resolution
        time = np.arange(n_samples) * dt
        
        # Generate realistic olfactory data
        data = pd.DataFrame()
        data['time'] = time
        
        # Odor stimuli
        n_odors = min(3, n_receptors)
        for i in range(n_odors):
            if i == 0:
                # Step stimulus
                stimulus = np.zeros(n_samples)
                start_idx = int(0.2 * n_samples)
                end_idx = int(0.6 * n_samples)
                stimulus[start_idx:end_idx] = 1.0
            elif i == 1:
                # Gaussian pulse
                center = 0.4 * n_samples * dt
                stimulus = np.exp(-((time - center)**2) / (2 * (0.05)**2))
            else:
                # Oscillatory
                stimulus = 0.5 * (1 + np.sin(2 * np.pi * 10 * time)) * ((time > 0.3) & (time < 0.8))
            
            data[f'odor_concentration_{i}'] = stimulus
        
        # Receptor responses
        for i in range(n_receptors):
            response = np.zeros(n_samples)
            # Each receptor responds to subset of odors
            for j in range(n_odors):
                if i % (j + 1) == 0:  # Selective response pattern
                    odor_response = np.convolve(
                        data[f'odor_concentration_{j}'], 
                        np.exp(-time[:100] / 0.02),
                        mode='same'
                    )
                    response += 0.5 * odor_response
            
            # Add adaptation
            adaptation = np.exp(-time / (0.1 + 0.1 * i))
            response *= adaptation
            
            # Add noise
            response += noise_level * np.random.randn(n_samples)
            data[f'receptor_response_{i}'] = response
        
        # Neural firing rates
        for i in range(n_neurons):
            firing_rate = np.zeros(n_samples)
            # Neural responses driven by receptor inputs
            for j in range(min(n_receptors, 3)):  # Each neuron receives from subset of receptors
                if (i + j) % 2 == 0:
                    receptor_input = data[f'receptor_response_{j}'].values
                    # Nonlinear transformation
                    neural_response = 30 * (1 / (1 + np.exp(-2 * receptor_input + 0.3)))
                    firing_rate += neural_response
            
            # Add temporal dynamics
            firing_rate = np.convolve(firing_rate, np.exp(-time[:50] / 0.01), mode='same')
            
            # Add noise
            firing_rate += noise_level * np.random.randn(n_samples)
            firing_rate = np.maximum(firing_rate, 0)  # Non-negative firing rates
            
            data[f'neural_firing_rate_{i}'] = firing_rate
        
        # Behavioral response
        neural_sum = sum(data[f'neural_firing_rate_{i}'].values for i in range(n_neurons))
        behavioral = np.convolve(neural_sum, np.ones(20)/20, mode='same')  # Moving average
        behavioral = behavioral / np.max(behavioral) if np.max(behavioral) > 0 else behavioral
        behavioral += noise_level * np.random.randn(n_samples)
        data['behavioral_response'] = behavioral
        
        return data
    
    def _compute_performance_metrics(self, result: BioneuroFusionResult, data: pd.DataFrame) -> Dict[str, float]:
        """Compute performance metrics for the causal discovery result."""
        metrics = {}
        
        # Network metrics
        adj_matrix = result.adjacency_matrix
        if adj_matrix.size > 0:
            metrics["network_density"] = np.sum(adj_matrix) / adj_matrix.size
            metrics["n_causal_edges"] = int(np.sum(adj_matrix))
            metrics["sparsity"] = 1 - metrics["network_density"]
            
            # Connectivity metrics
            if adj_matrix.shape[0] > 1:
                metrics["clustering_coefficient"] = self._compute_clustering_coefficient(adj_matrix)
                metrics["path_length"] = self._compute_average_path_length(adj_matrix)
        
        # Neural pathway metrics
        metrics.update({f"pathway_{k}": v for k, v in result.neural_pathways.items()})
        
        # Sensory integration metrics
        metrics.update({f"integration_{k}": v for k, v in result.sensory_integration_map.items()})
        
        # Confidence metrics
        if result.confidence_scores.size > 0:
            metrics["mean_confidence"] = np.mean(result.confidence_scores)
            metrics["confidence_std"] = np.std(result.confidence_scores)
        
        return metrics
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """Assess quality of input data."""
        quality_metrics = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Signal-to-noise ratio estimates
        for col in numeric_cols:
            values = data[col].values
            if len(values) > 10:
                # Estimate SNR using signal variance vs noise variance
                signal_power = np.var(values)
                # Estimate noise as high-frequency component
                diff_values = np.diff(values)
                noise_power = np.var(diff_values) / 2  # Approximate noise variance
                
                snr = signal_power / (noise_power + 1e-10)
                quality_metrics[f"{col}_snr"] = snr
        
        # Overall data quality metrics
        quality_metrics["data_completeness"] = 1 - data.isnull().sum().sum() / data.size
        quality_metrics["temporal_consistency"] = self._assess_temporal_consistency(data)
        
        return quality_metrics
    
    def _perform_statistical_validation(self, result: BioneuroFusionResult, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical validation of results."""
        validation = {}
        
        # Test statistical significance
        validation.update(result.statistical_significance)
        
        # Bootstrap validation
        n_bootstrap = 50  # Reduced for performance
        bootstrap_densities = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_indices = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_data = data.iloc[sample_indices].reset_index(drop=True)
            
            try:
                # Quick causal discovery on bootstrap sample
                model = OlfactoryNeuralCausalModel(
                    receptor_sensitivity_threshold=0.15,
                    neural_firing_threshold=10.0,
                    temporal_window_ms=100
                )
                bootstrap_result = model.fit_discover(bootstrap_data)
                bootstrap_densities.append(
                    np.sum(bootstrap_result.adjacency_matrix) / bootstrap_result.adjacency_matrix.size
                )
            except:
                continue
        
        if bootstrap_densities:
            validation["bootstrap_density_mean"] = np.mean(bootstrap_densities)
            validation["bootstrap_density_std"] = np.std(bootstrap_densities)
            validation["bootstrap_confidence_interval"] = (
                np.percentile(bootstrap_densities, 2.5),
                np.percentile(bootstrap_densities, 97.5)
            )
        
        return validation
    
    def _compute_aggregate_analytics(self) -> Dict[str, Any]:
        """Compute aggregate analytics across all experiments."""
        if not self.results:
            return {}
        
        analytics = {}
        
        # Performance trends
        execution_times = [r.execution_time for r in self.results]
        memory_usages = [r.memory_usage for r in self.results]
        
        analytics["performance"] = {
            "mean_execution_time": np.mean(execution_times),
            "std_execution_time": np.std(execution_times),
            "mean_memory_usage": np.mean(memory_usages),
            "std_memory_usage": np.std(memory_usages)
        }
        
        # Causal discovery trends
        n_edges = [r.causal_result.metadata.get("n_causal_edges", 0) for r in self.results]
        densities = [r.performance_metrics.get("network_density", 0) for r in self.results]
        
        analytics["causal_discovery"] = {
            "mean_edges": np.mean(n_edges),
            "std_edges": np.std(n_edges),
            "mean_density": np.mean(densities),
            "std_density": np.std(densities)
        }
        
        # Parameter sensitivity analysis
        analytics["parameter_sensitivity"] = self._analyze_parameter_sensitivity()
        
        return analytics
    
    def _compute_statistical_summaries(self) -> Dict[str, Any]:
        """Compute statistical summaries of results."""
        summaries = {}
        
        # Collect key metrics across experiments
        metrics_data = {}
        for result in self.results:
            for metric, value in result.performance_metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
        
        # Compute statistics for each metric
        for metric, values in metrics_data.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                summaries[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75)
                }
        
        return summaries
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across different conditions."""
        trends = {}
        
        # Group results by key parameters
        by_sample_size = {}
        by_noise_level = {}
        by_complexity = {}
        
        for result in self.results:
            config = result.config
            
            # By sample size
            n_samples = config["n_samples"]
            if n_samples not in by_sample_size:
                by_sample_size[n_samples] = []
            by_sample_size[n_samples].append(result.execution_time)
            
            # By noise level
            noise = config["noise_level"]
            if noise not in by_noise_level:
                by_noise_level[noise] = []
            by_noise_level[noise].append(result.performance_metrics.get("mean_confidence", 0))
            
            # By complexity (n_receptors * n_neurons)
            complexity = config["n_receptors"] * config["n_neurons"]
            if complexity not in by_complexity:
                by_complexity[complexity] = []
            by_complexity[complexity].append(result.execution_time)
        
        # Compute trends
        trends["execution_time_vs_sample_size"] = {
            str(k): np.mean(v) for k, v in by_sample_size.items()
        }
        trends["confidence_vs_noise_level"] = {
            str(k): np.mean(v) for k, v in by_noise_level.items()
        }
        trends["execution_time_vs_complexity"] = {
            str(k): np.mean(v) for k, v in by_complexity.items()
        }
        
        return trends
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to different parameters."""
        sensitivity = {}
        
        # Group by parameter values and compute outcome variance
        parameters = ["receptor_threshold", "neural_threshold", "temporal_window", "noise_level"]
        
        for param in parameters:
            param_groups = {}
            for result in self.results:
                param_value = result.config[param]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(
                    result.performance_metrics.get("network_density", 0)
                )
            
            # Compute coefficient of variation for each parameter value
            param_cv = {}
            for value, outcomes in param_groups.items():
                if len(outcomes) > 1:
                    mean_outcome = np.mean(outcomes)
                    std_outcome = np.std(outcomes)
                    cv = std_outcome / (mean_outcome + 1e-10)
                    param_cv[str(value)] = cv
            
            sensitivity[param] = param_cv
        
        return sensitivity
    
    def _generate_research_visualizations(self):
        """Generate research visualizations and save them."""
        try:
            # Create visualization directory
            viz_dir = self.output_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Plot 1: Performance vs sample size
            self._plot_performance_trends(viz_dir)
            
            # Plot 2: Parameter sensitivity heatmap
            self._plot_parameter_sensitivity(viz_dir)
            
            # Plot 3: Causal discovery success rates
            self._plot_causal_success_rates(viz_dir)
            
            logger.info(f"Generated research visualizations in {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
    
    def _plot_performance_trends(self, output_dir: Path):
        """Plot performance trends."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Trends - {self.config.experiment_name}', fontsize=16)
        
        # Execution time vs sample size
        sample_sizes = []
        exec_times = []
        for result in self.results:
            sample_sizes.append(result.config["n_samples"])
            exec_times.append(result.execution_time)
        
        axes[0, 0].scatter(sample_sizes, exec_times, alpha=0.6)
        axes[0, 0].set_xlabel('Sample Size')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time vs Sample Size')
        
        # Memory usage vs complexity
        complexities = []
        memory_usages = []
        for result in self.results:
            complexity = result.config["n_receptors"] * result.config["n_neurons"]
            complexities.append(complexity)
            memory_usages.append(result.memory_usage)
        
        axes[0, 1].scatter(complexities, memory_usages, alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Problem Complexity (receptors × neurons)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Complexity')
        
        # Confidence vs noise level
        noise_levels = []
        confidences = []
        for result in self.results:
            noise_levels.append(result.config["noise_level"])
            confidences.append(result.performance_metrics.get("mean_confidence", 0))
        
        axes[1, 0].scatter(noise_levels, confidences, alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Noise Level')
        axes[1, 0].set_ylabel('Mean Confidence')
        axes[1, 0].set_title('Confidence vs Noise Level')
        
        # Network density distribution
        densities = [r.performance_metrics.get("network_density", 0) for r in self.results]
        axes[1, 1].hist(densities, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Network Density')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Network Density Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, output_dir: Path):
        """Plot parameter sensitivity analysis."""
        # Create sensitivity matrix
        parameters = ["receptor_threshold", "neural_threshold", "temporal_window", "noise_level"]
        metrics = ["network_density", "mean_confidence", "execution_time"]
        
        sensitivity_matrix = np.zeros((len(parameters), len(metrics)))
        
        for i, param in enumerate(parameters):
            for j, metric in enumerate(metrics):
                # Compute correlation between parameter and metric
                param_values = [r.config[param] for r in self.results]
                metric_values = []
                
                for r in self.results:
                    if metric == "execution_time":
                        metric_values.append(r.execution_time)
                    else:
                        metric_values.append(r.performance_metrics.get(metric, 0))
                
                if len(set(param_values)) > 1 and len(set(metric_values)) > 1:
                    correlation = np.corrcoef(param_values, metric_values)[0, 1]
                    sensitivity_matrix[i, j] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(parameters)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(parameters)
        
        # Add values to heatmap
        for i in range(len(parameters)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('Parameter Sensitivity Analysis')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_causal_success_rates(self, output_dir: Path):
        """Plot causal discovery success rates."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate by noise level
        noise_groups = {}
        for result in self.results:
            noise = result.config["noise_level"]
            if noise not in noise_groups:
                noise_groups[noise] = []
            
            # Define "success" as significant causal relationships found
            success = result.causal_result.metadata.get("n_causal_edges", 0) > 0
            noise_groups[noise].append(success)
        
        noise_levels = sorted(noise_groups.keys())
        success_rates = [np.mean(noise_groups[noise]) for noise in noise_levels]
        
        axes[0].bar(range(len(noise_levels)), success_rates, color='skyblue')
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Causal Discovery Success Rate by Noise Level')
        axes[0].set_xticks(range(len(noise_levels)))
        axes[0].set_xticklabels([f'{n:.2f}' for n in noise_levels])
        
        # Edge count distribution
        edge_counts = [r.causal_result.metadata.get("n_causal_edges", 0) for r in self.results]
        axes[1].hist(edge_counts, bins=15, alpha=0.7, color='lightcoral')
        axes[1].set_xlabel('Number of Causal Edges')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Causal Edges Discovered')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'causal_success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_study_results(self, study_results: Dict[str, Any]):
        """Save study results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = self.output_path / f"study_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(study_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save CSV summary
        summary_data = []
        for result in self.results:
            row = {
                "experiment_id": result.experiment_id,
                "timestamp": result.timestamp,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                **result.config,
                **result.performance_metrics
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_path / f"experiment_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved study results to {results_file} and {summary_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, ExperimentResult):
            return asdict(obj)
        elif isinstance(obj, BioneuroFusionResult):
            return {
                "adjacency_matrix": obj.adjacency_matrix.tolist(),
                "confidence_scores": obj.confidence_scores.tolist(),
                "method_used": obj.method_used,
                "metadata": obj.metadata,
                "neural_pathways": obj.neural_pathways,
                "olfactory_correlations": obj.olfactory_correlations.tolist(),
                "sensory_integration_map": obj.sensory_integration_map,
                "confidence_intervals": obj.confidence_intervals,
                "statistical_significance": obj.statistical_significance
            }
        else:
            return obj
    
    def _compute_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """Compute clustering coefficient of the network."""
        n = adj_matrix.shape[0]
        if n < 3:
            return 0.0
        
        clustering = 0.0
        for i in range(n):
            neighbors = np.where(adj_matrix[i, :] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[l]] > 0:
                        triangles += 1
            
            # Local clustering coefficient
            possible_triangles = k * (k - 1) / 2
            if possible_triangles > 0:
                clustering += triangles / possible_triangles
        
        return clustering / n
    
    def _compute_average_path_length(self, adj_matrix: np.ndarray) -> float:
        """Compute average path length of the network."""
        n = adj_matrix.shape[0]
        if n < 2:
            return 0.0
        
        # Use simplified approximation for performance
        # In real implementation, would use proper shortest path algorithm
        total_paths = 0
        total_length = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Direct connection
                if adj_matrix[i, j] > 0:
                    total_length += 1
                    total_paths += 1
                else:
                    # Check for 2-step path (simplified)
                    found_path = False
                    for k in range(n):
                        if k != i and k != j and adj_matrix[i, k] > 0 and adj_matrix[k, j] > 0:
                            total_length += 2
                            total_paths += 1
                            found_path = True
                            break
                    
                    if not found_path:
                        # Assume disconnected or longer path
                        total_length += n  # Penalty for disconnection
                        total_paths += 1
        
        return total_length / total_paths if total_paths > 0 else float('inf')
    
    def _assess_temporal_consistency(self, data: pd.DataFrame) -> float:
        """Assess temporal consistency of the data."""
        if 'time' not in data.columns:
            return 1.0
        
        time_values = data['time'].values
        if len(time_values) < 2:
            return 1.0
        
        # Check for regular sampling
        time_diffs = np.diff(time_values)
        consistency = 1.0 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-10))
        return max(0.0, min(1.0, consistency))