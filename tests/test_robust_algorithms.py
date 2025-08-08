"""Tests for robust causal discovery algorithms."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.algorithms.robust import RobustSimpleLinearCausalModel
from src.utils.data_processing import DataProcessor
from src.utils.monitoring import CircuitBreakerOpenException


class TestRobustSimpleLinearCausalModel:
    """Tests for RobustSimpleLinearCausalModel."""
    
    def test_init_with_validation(self):
        """Test initialization with parameter validation."""
        # Valid initialization
        model = RobustSimpleLinearCausalModel(threshold=0.5)
        assert model.threshold == 0.5
        assert model.validate_inputs == True
        
        # Invalid threshold should raise error
        with pytest.raises(ValueError, match="Invalid threshold"):
            RobustSimpleLinearCausalModel(threshold=1.5)
    
    def test_fit_with_valid_data(self):
        """Test fitting with valid data."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
        
        model = RobustSimpleLinearCausalModel()
        result = model.fit(data)
        
        assert model.is_fitted
        assert model._fitted_successfully
        assert result is model
        assert 'validation' in model.fit_metadata
    
    def test_fit_with_invalid_data(self):
        """Test fitting with invalid data."""
        model = RobustSimpleLinearCausalModel()
        
        # Empty DataFrame
        with pytest.raises(RuntimeError, match="Model fitting failed"):
            model.fit(pd.DataFrame())
        
        # Non-DataFrame
        with pytest.raises(RuntimeError, match="Model fitting failed"):
            model.fit("not a dataframe")
    
    def test_fit_with_insufficient_samples(self):
        """Test fitting with insufficient samples."""
        model = RobustSimpleLinearCausalModel(min_samples=100)
        small_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pytest.raises(RuntimeError, match="Insufficient samples"):
            model.fit(small_data)
    
    def test_fit_with_too_many_features(self):
        """Test fitting with too many features."""
        model = RobustSimpleLinearCausalModel(max_features=2)
        
        # Create data with 5 features
        data = pd.DataFrame(np.random.randn(50, 5), 
                          columns=[f'X{i}' for i in range(5)])
        
        with pytest.raises(RuntimeError, match="Too many features"):
            model.fit(data)
    
    def test_discover_without_fit(self):
        """Test discovery without fitting."""
        model = RobustSimpleLinearCausalModel()
        
        with pytest.raises(RuntimeError, match="must be successfully fitted"):
            model.discover()
    
    def test_discover_after_successful_fit(self):
        """Test discovery after successful fitting."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
        
        model = RobustSimpleLinearCausalModel(threshold=0.3)
        model.fit(data)
        result = model.discover()
        
        assert result.method_used == "RobustSimpleLinearCausal"
        assert result.adjacency_matrix.shape == (3, 3)
        assert 'sparsity' in result.metadata
        assert 'max_confidence' in result.metadata
    
    def test_missing_data_handling(self):
        """Test different missing data handling strategies."""
        # Create data with missing values
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [2, np.nan, 4, 5, 6],
            'C': [3, 4, 5, 6, 7]
        })
        
        # Test drop strategy
        model_drop = RobustSimpleLinearCausalModel(handle_missing='drop')
        model_drop.fit(data)
        assert model_drop._data.shape[0] < data.shape[0]
        
        # Test impute mean strategy
        model_mean = RobustSimpleLinearCausalModel(handle_missing='impute_mean')
        model_mean.fit(data)
        assert not model_mean._data.isnull().any().any()
        
        # Test impute median strategy
        model_median = RobustSimpleLinearCausalModel(handle_missing='impute_median')
        model_median.fit(data)
        assert not model_median._data.isnull().any().any()
    
    def test_correlation_methods(self):
        """Test different correlation methods."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
        
        methods = ['pearson', 'spearman', 'kendall']
        
        for method in methods:
            model = RobustSimpleLinearCausalModel(correlation_method=method)
            model.fit(data)
            result = model.discover()
            assert result.metadata['correlation_method'] == method
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker behavior."""
        model = RobustSimpleLinearCausalModel()
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=50, n_variables=3, random_state=42)
        model.fit(data)
        
        # Mock the internal discovery method to always fail
        with patch.object(model, '_perform_discovery', side_effect=Exception("Mock failure")):
            # First few failures should raise RuntimeError
            for i in range(3):
                with pytest.raises(RuntimeError, match="Causal discovery failed"):
                    model.discover()
            
            # After threshold failures, circuit breaker should open
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                model.discover()
        
        # Reset circuit breaker
        model.reset_circuit_breaker()
        assert model.circuit_breaker.state == "CLOSED"
    
    def test_preprocessing_edge_cases(self):
        """Test data preprocessing with edge cases."""
        model = RobustSimpleLinearCausalModel()
        
        # All constant columns
        constant_data = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [2, 2, 2, 2, 2]
        })
        
        with pytest.raises(RuntimeError, match="preprocessing failed"):
            model.fit(constant_data)
        
        # Mixed data types
        mixed_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Should work - non-numeric columns get filtered out
        model.fit(mixed_data)
        assert model._data.shape[1] == 1  # Only numeric column A
    
    def test_model_info(self):
        """Test getting comprehensive model information."""
        model = RobustSimpleLinearCausalModel(threshold=0.4, correlation_method='spearman')
        
        info = model.get_model_info()
        assert info['model_type'] == 'RobustSimpleLinearCausalModel'
        assert info['parameters']['threshold'] == 0.4
        assert info['parameters']['correlation_method'] == 'spearman'
        assert not info['state']['fitted_successfully']
        assert info['circuit_breaker_state'] == "CLOSED"
    
    def test_health_validation(self):
        """Test model health validation."""
        model = RobustSimpleLinearCausalModel()
        
        # Initially not healthy (not fitted)
        health = model.validate_health()
        assert not health['healthy']
        assert "not fitted successfully" in " ".join(health['issues']).lower()
        
        # Fit model
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=50, n_variables=3)
        model.fit(data)
        
        # Should be healthy after fitting
        health = model.validate_health()
        assert health['healthy']
        assert health['model_ready']
    
    def test_validation_disabled(self):
        """Test behavior with input validation disabled."""
        model = RobustSimpleLinearCausalModel(validate_inputs=False)
        
        # Should work even with problematic data when validation is disabled
        problematic_data = pd.DataFrame({'A': [1, 2, 3]})  # Single column, few samples
        
        try:
            model.fit(problematic_data)
            result = model.discover()
            # Should work but may produce warnings
            assert result is not None
        except Exception as e:
            # May still fail due to insufficient data, but not validation
            assert "validation" not in str(e).lower()
    
    def test_performance_monitoring_integration(self):
        """Test that performance monitoring is working."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3)
        
        model = RobustSimpleLinearCausalModel()
        
        # The @monitor_performance decorators should be working
        # We can't easily test the actual monitoring without complex mocking,
        # but we can ensure the methods still work with the decorators
        model.fit(data)
        result = model.discover()
        
        assert result is not None
        assert model._fitted_successfully
    
    def test_error_propagation(self):
        """Test that errors are properly propagated with context."""
        model = RobustSimpleLinearCausalModel()
        
        # Test with truly invalid data that should cause preprocessing to fail
        with patch.object(model, '_preprocess_data', side_effect=ValueError("Preprocessing error")):
            with pytest.raises(RuntimeError, match="Model fitting failed") as exc_info:
                model.fit(pd.DataFrame({'A': [1, 2, 3]}))
            
            # Should have original error as cause
            assert "Preprocessing error" in str(exc_info.value.__cause__)
    
    def test_adjacency_matrix_validation_in_discovery(self):
        """Test that adjacency matrix validation is performed during discovery."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
        
        model = RobustSimpleLinearCausalModel()
        model.fit(data)
        result = model.discover()
        
        # The adjacency matrix should be valid
        assert result.adjacency_matrix.shape == (3, 3)
        assert np.all(np.isin(result.adjacency_matrix, [0, 1]))
        assert np.all(np.diag(result.adjacency_matrix) == 0)  # No self-loops