#!/usr/bin/env python3
"""
Production-ready benchmark script for bioneuro-olfactory causal discovery.
Comprehensive testing, validation, and performance evaluation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import logging
from datetime import datetime
from pathlib import Path
import json

from experiments.bioneuro_research_suite import BioneuroResearchSuite, ResearchExperimentConfig
from utils.logging_config import get_logger
from utils.monitoring import PerformanceMonitor
from utils.validation import DataValidator

logger = get_logger(__name__)


def create_benchmark_configs() -> dict:
    """Create predefined benchmark configurations."""
    configs = {
        "quick_test": ResearchExperimentConfig(
            experiment_name="quick_test_bioneuro",
            n_samples_range=[100, 300],
            n_receptors_range=[3, 5],
            n_neurons_range=[2, 4],
            noise_levels=[0.1, 0.2],
            receptor_thresholds=[0.1, 0.15],
            neural_thresholds=[5.0, 10.0],
            temporal_windows=[50, 100],
            n_runs_per_condition=2,
            enable_multimodal=True,
            enable_benchmarking=True,
            save_results=True,
            output_dir="benchmark_results_quick",
            random_seed=42
        ),
        
        "standard_benchmark": ResearchExperimentConfig(
            experiment_name="standard_bioneuro_benchmark",
            n_samples_range=[200, 500, 1000],
            n_receptors_range=[3, 6, 10],
            n_neurons_range=[2, 5, 8],
            noise_levels=[0.05, 0.1, 0.2],
            receptor_thresholds=[0.1, 0.15, 0.2],
            neural_thresholds=[5.0, 10.0, 15.0],
            temporal_windows=[50, 100, 200],
            n_runs_per_condition=3,
            enable_multimodal=True,
            enable_benchmarking=True,
            save_results=True,
            output_dir="benchmark_results_standard",
            random_seed=42
        ),
        
        "comprehensive": ResearchExperimentConfig(
            experiment_name="comprehensive_bioneuro_study",
            n_samples_range=[100, 300, 500, 1000, 2000],
            n_receptors_range=[3, 6, 10, 15],
            n_neurons_range=[2, 5, 8, 12],
            noise_levels=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            receptor_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25],
            neural_thresholds=[2.0, 5.0, 10.0, 15.0, 20.0],
            temporal_windows=[25, 50, 100, 150, 200],
            n_runs_per_condition=5,
            enable_multimodal=True,
            enable_benchmarking=True,
            save_results=True,
            output_dir="benchmark_results_comprehensive",
            random_seed=42
        ),
        
        "performance_test": ResearchExperimentConfig(
            experiment_name="performance_scaling_test",
            n_samples_range=[1000, 2000, 5000, 10000],
            n_receptors_range=[10, 20, 30],
            n_neurons_range=[8, 16, 24],
            noise_levels=[0.1],
            receptor_thresholds=[0.15],
            neural_thresholds=[10.0],
            temporal_windows=[100],
            n_runs_per_condition=3,
            enable_multimodal=True,
            enable_benchmarking=True,
            save_results=True,
            output_dir="benchmark_results_performance",
            random_seed=42
        ),
        
        "robustness_test": ResearchExperimentConfig(
            experiment_name="robustness_stress_test",
            n_samples_range=[500],
            n_receptors_range=[5],
            n_neurons_range=[5],
            noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
            receptor_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            neural_thresholds=[1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
            temporal_windows=[25, 50, 100, 200, 400],
            n_runs_per_condition=5,
            enable_multimodal=True,
            enable_benchmarking=True,
            save_results=True,
            output_dir="benchmark_results_robustness",
            random_seed=42
        )
    }
    
    return configs


def run_benchmark(config_name: str, custom_config: dict = None) -> dict:
    """
    Run benchmark with specified configuration.
    
    Args:
        config_name: Name of predefined config or "custom"
        custom_config: Custom configuration dict if config_name is "custom"
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Starting benchmark: {config_name}")
    
    # Get configuration
    if config_name == "custom" and custom_config:
        config = ResearchExperimentConfig(**custom_config)
    else:
        configs = create_benchmark_configs()
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        config = configs[config_name]
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize performance monitoring
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_monitoring()
    
    try:
        # Initialize research suite
        research_suite = BioneuroResearchSuite(config)
        
        # Run comprehensive study
        study_results = research_suite.run_comprehensive_study()
        
        # Stop monitoring
        perf_monitor.stop_monitoring()
        
        # Add performance monitoring results
        study_results["system_performance"] = {
            "peak_memory_usage": perf_monitor.get_peak_memory_usage(),
            "average_cpu_usage": perf_monitor.get_average_cpu_usage(),
            "system_info": perf_monitor.get_system_info()
        }
        
        # Generate summary report
        summary_report = generate_summary_report(study_results, config)
        
        # Save summary report
        report_file = output_path / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Benchmark completed successfully. Results in: {output_path}")
        logger.info(f"Summary report: {report_file}")
        
        return study_results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        perf_monitor.stop_monitoring()
        raise


def generate_summary_report(study_results: dict, config: ResearchExperimentConfig) -> str:
    """Generate human-readable summary report."""
    report = []
    report.append("=" * 80)
    report.append("BIONEURO-OLFACTORY CAUSAL DISCOVERY BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Study metadata
    metadata = study_results["study_metadata"]
    report.append(f"Experiment Name: {metadata['experiment_name']}")
    report.append(f"Start Time: {metadata['start_time']}")
    report.append(f"End Time: {metadata['end_time']}")
    report.append(f"Total Execution Time: {metadata['execution_time']:.2f} seconds")
    report.append("")
    
    # Configuration summary
    report.append("CONFIGURATION SUMMARY:")
    report.append("-" * 40)
    config_dict = metadata['config']
    report.append(f"Sample Sizes: {config_dict['n_samples_range']}")
    report.append(f"Receptor Counts: {config_dict['n_receptors_range']}")
    report.append(f"Neuron Counts: {config_dict['n_neurons_range']}")
    report.append(f"Noise Levels: {config_dict['noise_levels']}")
    report.append(f"Runs per Condition: {config_dict['n_runs_per_condition']}")
    report.append(f"Total Experiments: {len(study_results['experiment_results'])}")
    report.append("")
    
    # Performance summary
    if "aggregate_analytics" in study_results:
        analytics = study_results["aggregate_analytics"]
        
        if "performance" in analytics:
            perf = analytics["performance"]
            report.append("PERFORMANCE SUMMARY:")
            report.append("-" * 40)
            report.append(f"Mean Execution Time: {perf['mean_execution_time']:.3f} ± {perf['std_execution_time']:.3f} seconds")
            report.append(f"Mean Memory Usage: {perf['mean_memory_usage']:.2f} ± {perf['std_memory_usage']:.2f} MB")
            report.append("")
        
        if "causal_discovery" in analytics:
            causal = analytics["causal_discovery"]
            report.append("CAUSAL DISCOVERY SUMMARY:")
            report.append("-" * 40)
            report.append(f"Mean Causal Edges: {causal['mean_edges']:.2f} ± {causal['std_edges']:.2f}")
            report.append(f"Mean Network Density: {causal['mean_density']:.3f} ± {causal['std_density']:.3f}")
            report.append("")
    
    # Statistical summaries
    if "statistical_summaries" in study_results:
        summaries = study_results["statistical_summaries"]
        
        report.append("KEY METRICS STATISTICS:")
        report.append("-" * 40)
        
        key_metrics = ["network_density", "mean_confidence", "pathway_receptor_to_neural", 
                      "integration_integration_strength"]
        
        for metric in key_metrics:
            if metric in summaries:
                stats = summaries[metric]
                report.append(f"{metric}:")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Std:  {stats['std']:.4f}")
                report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report.append("")
    
    # Performance trends
    if "performance_analysis" in study_results:
        trends = study_results["performance_analysis"]
        
        report.append("PERFORMANCE TRENDS:")
        report.append("-" * 40)
        
        if "execution_time_vs_sample_size" in trends:
            report.append("Execution Time vs Sample Size:")
            for size, time in trends["execution_time_vs_sample_size"].items():
                report.append(f"  {size} samples: {time:.3f}s")
            report.append("")
        
        if "confidence_vs_noise_level" in trends:
            report.append("Confidence vs Noise Level:")
            for noise, conf in trends["confidence_vs_noise_level"].items():
                report.append(f"  {noise} noise: {conf:.3f}")
            report.append("")
    
    # System performance
    if "system_performance" in study_results:
        sys_perf = study_results["system_performance"]
        
        report.append("SYSTEM PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Peak Memory Usage: {sys_perf['peak_memory_usage']:.2f} MB")
        report.append(f"Average CPU Usage: {sys_perf['average_cpu_usage']:.1f}%")
        
        if "system_info" in sys_perf:
            sys_info = sys_perf["system_info"]
            report.append(f"CPU Cores: {sys_info.get('cpu_cores', 'N/A')}")
            report.append(f"Total Memory: {sys_info.get('total_memory_gb', 'N/A')} GB")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)
    
    # Analyze results and provide recommendations
    if "statistical_summaries" in study_results:
        summaries = study_results["statistical_summaries"]
        
        # Performance recommendations
        if "execution_time" in study_results["performance_analysis"].get("execution_time_vs_sample_size", {}):
            max_time = max(float(t) for t in study_results["performance_analysis"]["execution_time_vs_sample_size"].values())
            if max_time > 10.0:
                report.append("• Consider optimizing for large datasets (execution time > 10s)")
        
        # Accuracy recommendations
        if "mean_confidence" in summaries:
            mean_conf = summaries["mean_confidence"]["mean"]
            if mean_conf < 0.7:
                report.append("• Low confidence scores suggest parameter tuning needed")
            elif mean_conf > 0.9:
                report.append("• High confidence scores indicate robust discovery")
        
        # Robustness recommendations
        if "network_density" in summaries:
            density_std = summaries["network_density"]["std"]
            if density_std > 0.2:
                report.append("• High density variance suggests sensitivity to parameters")
        
        report.append("• Review visualizations in the output directory for detailed analysis")
        report.append("• Consider running robustness tests for production deployment")
    
    report.append("")
    report.append("=" * 80)
    report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    return "\\n".join(report)


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Run bioneuro-olfactory causal discovery benchmark")
    
    parser.add_argument(
        "--config", 
        choices=["quick_test", "standard_benchmark", "comprehensive", "performance_test", "robustness_test"],
        default="quick_test",
        help="Benchmark configuration to run"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running benchmark"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Get configuration
        configs = create_benchmark_configs()
        config = configs[args.config]
        
        # Override output directory if specified
        if args.output_dir:
            config.output_dir = args.output_dir
        
        if args.dry_run:
            print("\\nBenchmark Configuration:")
            print("=" * 50)
            config_dict = config.__dict__
            for key, value in config_dict.items():
                print(f"{key}: {value}")
            
            # Estimate number of experiments
            total_experiments = 1
            for param_range in ['n_samples_range', 'n_receptors_range', 'n_neurons_range', 
                              'noise_levels', 'receptor_thresholds', 'neural_thresholds', 'temporal_windows']:
                total_experiments *= len(getattr(config, param_range))
            total_experiments *= config.n_runs_per_condition
            
            print(f"\\nEstimated total experiments: {total_experiments}")
            print("\\nUse --config to run the benchmark")
            return
        
        # Run benchmark
        print(f"\\nRunning benchmark: {args.config}")
        print(f"Output directory: {config.output_dir}")
        print("This may take several minutes to hours depending on configuration...")
        
        results = run_benchmark(args.config)
        
        print("\\n✅ Benchmark completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}")
        print(f"📈 Visualizations generated")
        print(f"📋 Summary report available")
        
        # Print quick summary
        if results and "aggregate_analytics" in results:
            analytics = results["aggregate_analytics"]
            if "performance" in analytics:
                perf = analytics["performance"]
                print(f"\\n📈 Quick Summary:")
                print(f"   Average execution time: {perf['mean_execution_time']:.2f}s")
                print(f"   Average memory usage: {perf['mean_memory_usage']:.1f}MB")
            
            if "causal_discovery" in analytics:
                causal = analytics["causal_discovery"]
                print(f"   Average causal edges: {causal['mean_edges']:.1f}")
                print(f"   Average network density: {causal['mean_density']:.3f}")
        
    except KeyboardInterrupt:
        print("\\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        print(f"\\n❌ Benchmark failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()