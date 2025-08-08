"""Tests for experiment and benchmarking functionality."""

import pytest
import pandas as pd
import numpy as np
from src.experiments.benchmark import CausalBenchmark
from src.algorithms.base import SimpleLinearCausalModel
from src.utils.data_processing import DataProcessor


class TestCausalBenchmark:
    """Tests for CausalBenchmark class."""
    
    def test_init(self):
        """Test benchmark initialization."""
        benchmark = CausalBenchmark()
        assert benchmark.results == []
        assert isinstance(benchmark.data_processor, DataProcessor)
    
    def test_run_single_experiment_success(self):
        """Test running a single successful experiment."""
        benchmark = CausalBenchmark()
        
        # Generate test data
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(
            n_samples=100, n_variables=3, random_state=42
        )
        
        # Create true adjacency matrix
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        # Run experiment
        model = SimpleLinearCausalModel(threshold=0.3)
        result = benchmark.run_single_experiment(model, data, true_adj)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'runtime_seconds' in result
        assert 'success' in result
        assert 'n_samples' in result
        assert 'n_variables' in result
        
        assert result['model'] == 'SimpleLinearCausalModel'
        assert result['success'] == True
        assert result['n_samples'] == 100
        assert result['n_variables'] == 3
        assert result['runtime_seconds'] > 0
        
        # Should have evaluation metrics since true adjacency was provided
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
    
    def test_run_single_experiment_without_true_adjacency(self):
        """Test running experiment without true adjacency matrix."""
        benchmark = CausalBenchmark()
        
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=50, n_variables=3)
        
        model = SimpleLinearCausalModel()
        result = benchmark.run_single_experiment(model, data)
        
        assert result['success'] == True
        # Should not have evaluation metrics without true adjacency
        assert 'precision' not in result
        assert 'recall' not in result
        assert 'f1_score' not in result
        # But should have basic discovery info
        assert 'n_discovered_edges' in result
        assert 'discovery_metadata' in result
    
    def test_run_single_experiment_failure(self):
        """Test handling of failed experiments."""
        benchmark = CausalBenchmark()
        
        # Create a model that will fail (invalid data type)
        class FailingModel(SimpleLinearCausalModel):
            def fit(self, data):
                raise ValueError("Intentional test failure")
        
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=50, n_variables=3)
        
        model = FailingModel()
        result = benchmark.run_single_experiment(model, data)
        
        assert result['success'] == False
        assert 'error_msg' in result
        assert "Intentional test failure" in result['error_msg']
        assert result['runtime_seconds'] > 0
    
    def test_run_synthetic_benchmark_small(self):
        """Test running a small synthetic benchmark."""
        benchmark = CausalBenchmark()
        models = [SimpleLinearCausalModel(threshold=0.3)]
        
        results_df = benchmark.run_synthetic_benchmark(
            models=models,
            n_samples_list=[50, 100],
            n_variables_list=[3, 4],
            noise_levels=[0.1, 0.2],
            n_runs=2,
            random_state=42
        )
        
        # Should have 1 model × 2 sample sizes × 2 variable counts × 2 noise levels × 2 runs = 16 experiments
        assert len(results_df) == 16
        
        # Check column structure
        expected_columns = {
            'model', 'n_samples', 'n_variables', 'noise_level', 'run_idx',
            'runtime_seconds', 'success', 'error_msg'
        }
        assert expected_columns.issubset(set(results_df.columns))
        
        # All experiments should be with SimpleLinearCausalModel
        assert all(results_df['model'] == 'SimpleLinearCausalModel')
        
        # Should have the specified parameter combinations
        assert set(results_df['n_samples']) == {50, 100}
        assert set(results_df['n_variables']) == {3, 4}
        assert set(results_df['noise_level']) == {0.1, 0.2}
        assert set(results_df['run_idx']) == {0, 1}
    
    def test_summarize_results_empty(self):
        """Test summarizing empty results."""
        benchmark = CausalBenchmark()
        empty_df = pd.DataFrame()
        
        summary = benchmark.summarize_results(empty_df)
        assert summary == {}
    
    def test_summarize_results_with_data(self):
        """Test summarizing benchmark results with data."""
        benchmark = CausalBenchmark()
        
        # Create mock results data
        results_data = {
            'model': ['ModelA', 'ModelA', 'ModelB', 'ModelB'],
            'success': [True, True, True, False],
            'runtime_seconds': [0.1, 0.15, 0.2, 0.05],
            'f1_score': [0.8, 0.75, 0.9, np.nan],
            'precision': [0.85, 0.8, 0.95, np.nan],
            'recall': [0.75, 0.7, 0.86, np.nan]
        }
        results_df = pd.DataFrame(results_data)
        
        summary = benchmark.summarize_results(results_df)
        
        # Check basic statistics
        assert summary['total_experiments'] == 4
        assert summary['success_rate'] == 0.75  # 3 out of 4 successful
        assert 'mean_runtime' in summary
        assert 'std_runtime' in summary
        
        # Check model statistics
        assert 'model_statistics' in summary
        model_stats = summary['model_statistics']
        # The structure is nested due to multi-level column aggregation
        # Check that we can access the data
        assert len(model_stats) > 0
        
        # Check best model identification
        assert 'best_f1_model' in summary
        assert summary['best_f1_model'] == 'ModelB'  # Has highest F1 score (0.9)
        assert summary['best_f1_score'] == 0.9
    
    def test_benchmark_reproducibility(self):
        """Test that benchmarks are reproducible with same random state."""
        benchmark1 = CausalBenchmark()
        benchmark2 = CausalBenchmark()
        
        models = [SimpleLinearCausalModel(threshold=0.3)]
        params = {
            'models': models,
            'n_samples_list': [50],
            'n_variables_list': [3],
            'noise_levels': [0.1],
            'n_runs': 2,
            'random_state': 42
        }
        
        results1 = benchmark1.run_synthetic_benchmark(**params)
        results2 = benchmark2.run_synthetic_benchmark(**params)
        
        # Results should be identical except for runtime (which can vary slightly)
        cols_to_compare = [col for col in results1.columns if col != 'runtime_seconds']
        pd.testing.assert_frame_equal(results1[cols_to_compare], results2[cols_to_compare])
    
    def test_benchmark_with_multiple_models(self):
        """Test benchmarking with multiple models."""
        benchmark = CausalBenchmark()
        
        models = [
            SimpleLinearCausalModel(threshold=0.2),
            SimpleLinearCausalModel(threshold=0.5)
        ]
        
        results_df = benchmark.run_synthetic_benchmark(
            models=models,
            n_samples_list=[100],
            n_variables_list=[3],
            noise_levels=[0.1],
            n_runs=1,
            random_state=42
        )
        
        # Should have results for both models
        assert len(results_df) == 2
        model_names = set(results_df['model'])
        assert model_names == {'SimpleLinearCausalModel'}
        
        # But should have different hyperparameters (thresholds)
        # We can't directly check this from the results, but both should run successfully
        assert all(results_df['success'] == True)