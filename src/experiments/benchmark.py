"""Benchmarking experiments for causal discovery methods."""

import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
try:
    from ..algorithms.base import CausalDiscoveryModel
    from ..utils.data_processing import DataProcessor
    from ..utils.metrics import CausalMetrics
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from base import CausalDiscoveryModel
    from data_processing import DataProcessor
    from metrics import CausalMetrics


class CausalBenchmark:
    """Benchmark suite for causal discovery algorithms."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.data_processor = DataProcessor()
        
    def run_synthetic_benchmark(self,
                              models: List[CausalDiscoveryModel],
                              n_samples_list: List[int] = [100, 500, 1000],
                              n_variables_list: List[int] = [3, 5, 10],
                              noise_levels: List[float] = [0.1, 0.3, 0.5],
                              n_runs: int = 5,
                              random_state: int = 42) -> pd.DataFrame:
        """Run comprehensive benchmark on synthetic data.
        
        Args:
            models: List of causal discovery models to benchmark
            n_samples_list: List of sample sizes to test
            n_variables_list: List of variable counts to test
            noise_levels: List of noise levels to test
            n_runs: Number of runs per configuration
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for model in models:
            model_name = model.__class__.__name__
            
            for n_samples in n_samples_list:
                for n_vars in n_variables_list:
                    for noise_level in noise_levels:
                        for run_idx in range(n_runs):
                            # Generate synthetic data
                            np.random.seed(random_state + run_idx)
                            data = self.data_processor.generate_synthetic_data(
                                n_samples=n_samples,
                                n_variables=n_vars,
                                noise_level=noise_level,
                                random_state=random_state + run_idx
                            )
                            
                            # Create true adjacency matrix (simple chain)
                            true_adj = np.zeros((n_vars, n_vars))
                            for i in range(n_vars - 1):
                                true_adj[i, i + 1] = 1
                            
                            # Run model and measure time
                            start_time = time.time()
                            try:
                                result = model.fit_discover(data)
                                runtime = time.time() - start_time
                                success = True
                                error_msg = None
                            except Exception as e:
                                runtime = time.time() - start_time
                                success = False
                                error_msg = str(e)
                                result = None
                            
                            # Evaluate if successful
                            if success and result is not None:
                                metrics = CausalMetrics.evaluate_discovery(
                                    true_adj, 
                                    result.adjacency_matrix,
                                    result.confidence_scores
                                )
                            else:
                                metrics = {}
                            
                            # Store results
                            result_dict = {
                                'model': model_name,
                                'n_samples': n_samples,
                                'n_variables': n_vars,
                                'noise_level': noise_level,
                                'run_idx': run_idx,
                                'runtime_seconds': runtime,
                                'success': success,
                                'error_msg': error_msg,
                                **metrics
                            }
                            results.append(result_dict)
                            
        return pd.DataFrame(results)
    
    def run_single_experiment(self,
                            model: CausalDiscoveryModel,
                            data: pd.DataFrame,
                            true_adjacency: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run a single causal discovery experiment.
        
        Args:
            model: Causal discovery model
            data: Input data
            true_adjacency: True adjacency matrix if known
            
        Returns:
            Experiment results dictionary
        """
        start_time = time.time()
        
        try:
            result = model.fit_discover(data)
            runtime = time.time() - start_time
            success = True
            error_msg = None
            
            experiment_result = {
                'model': model.__class__.__name__,
                'runtime_seconds': runtime,
                'success': success,
                'error_msg': error_msg,
                'n_samples': len(data),
                'n_variables': len(data.columns),
                'n_discovered_edges': np.sum(result.adjacency_matrix),
                'discovery_metadata': result.metadata
            }
            
            # Add evaluation metrics if true adjacency is provided
            if true_adjacency is not None:
                metrics = CausalMetrics.evaluate_discovery(
                    true_adjacency,
                    result.adjacency_matrix, 
                    result.confidence_scores
                )
                experiment_result.update(metrics)
            
            return experiment_result
            
        except Exception as e:
            runtime = time.time() - start_time
            return {
                'model': model.__class__.__name__,
                'runtime_seconds': runtime,
                'success': False,
                'error_msg': str(e),
                'n_samples': len(data),
                'n_variables': len(data.columns)
            }
    
    def summarize_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize benchmark results.
        
        Args:
            results_df: DataFrame with benchmark results
            
        Returns:
            Summary statistics dictionary
        """
        if results_df.empty:
            return {}
            
        summary = {}
        
        # Overall statistics
        summary['total_experiments'] = len(results_df)
        summary['success_rate'] = results_df['success'].mean()
        summary['mean_runtime'] = results_df['runtime_seconds'].mean()
        summary['std_runtime'] = results_df['runtime_seconds'].std()
        
        # Per-model statistics
        model_stats = results_df.groupby('model').agg({
            'success': 'mean',
            'runtime_seconds': ['mean', 'std'],
            'f1_score': 'mean' if 'f1_score' in results_df.columns else lambda x: None,
            'precision': 'mean' if 'precision' in results_df.columns else lambda x: None,
            'recall': 'mean' if 'recall' in results_df.columns else lambda x: None
        }).round(4)
        
        summary['model_statistics'] = model_stats.to_dict()
        
        # Best performing model by F1 score
        if 'f1_score' in results_df.columns:
            successful_results = results_df[results_df['success'] == True]
            if not successful_results.empty:
                best_model_row = successful_results.loc[successful_results['f1_score'].idxmax()]
                summary['best_f1_model'] = best_model_row['model']
                summary['best_f1_score'] = best_model_row['f1_score']
        
        return summary