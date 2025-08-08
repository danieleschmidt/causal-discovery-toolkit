"""Tests for causal discovery algorithms."""

import pytest
import pandas as pd
import numpy as np
from src.algorithms.base import SimpleLinearCausalModel, CausalResult
from src.utils.data_processing import DataProcessor


class TestSimpleLinearCausalModel:
    """Tests for SimpleLinearCausalModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = SimpleLinearCausalModel(threshold=0.5)
        assert model.threshold == 0.5
        assert not model.is_fitted
    
    def test_fit_with_valid_data(self):
        """Test fitting with valid data."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3)
        
        model = SimpleLinearCausalModel()
        result = model.fit(data)
        
        assert model.is_fitted
        assert result is model  # Should return self
        assert model._data is not None
    
    def test_fit_with_invalid_data(self):
        """Test fitting with invalid data."""
        model = SimpleLinearCausalModel()
        
        # Test with non-DataFrame
        with pytest.raises(TypeError):
            model.fit("not a dataframe")
        
        # Test with empty DataFrame
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame())
    
    def test_discover_without_fit(self):
        """Test discovery without fitting first."""
        model = SimpleLinearCausalModel()
        
        with pytest.raises(RuntimeError):
            model.discover()
    
    def test_discover_after_fit(self):
        """Test discovery after fitting."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(
            n_samples=100, 
            n_variables=3,
            random_state=42
        )
        
        model = SimpleLinearCausalModel(threshold=0.3)
        model.fit(data)
        result = model.discover()
        
        assert isinstance(result, CausalResult)
        assert result.adjacency_matrix.shape == (3, 3)
        assert result.confidence_scores.shape == (3, 3)
        assert result.method_used == "SimpleLinearCausal"
        assert "threshold" in result.metadata
        assert result.metadata["n_variables"] == 3
    
    def test_fit_discover_convenience(self):
        """Test fit_discover convenience method."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=100, n_variables=3)
        
        model = SimpleLinearCausalModel()
        result = model.fit_discover(data)
        
        assert isinstance(result, CausalResult)
        assert model.is_fitted
    
    def test_adjacency_matrix_properties(self):
        """Test properties of discovered adjacency matrix."""
        # Create data with known structure
        np.random.seed(42)
        n_samples = 200
        X1 = np.random.randn(n_samples)
        X2 = 0.8 * X1 + 0.1 * np.random.randn(n_samples)  # Strong correlation
        X3 = 0.1 * X1 + 0.1 * np.random.randn(n_samples)  # Weak correlation
        
        data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
        
        model = SimpleLinearCausalModel(threshold=0.5)
        result = model.fit_discover(data)
        
        # Should have no self-connections
        assert np.all(np.diag(result.adjacency_matrix) == 0)
        
        # Should detect strong correlation between X1 and X2
        assert result.adjacency_matrix[0, 1] == 1 or result.adjacency_matrix[1, 0] == 1
        
        # Adjacency matrix should be binary
        assert np.all(np.isin(result.adjacency_matrix, [0, 1]))
    
    def test_different_thresholds(self):
        """Test behavior with different thresholds."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(
            n_samples=200, 
            n_variables=4,
            random_state=42
        )
        
        # Lower threshold should find more edges
        model_low = SimpleLinearCausalModel(threshold=0.1)
        result_low = model_low.fit_discover(data)
        
        # Higher threshold should find fewer edges  
        model_high = SimpleLinearCausalModel(threshold=0.8)
        result_high = model_high.fit_discover(data)
        
        edges_low = np.sum(result_low.adjacency_matrix)
        edges_high = np.sum(result_high.adjacency_matrix)
        
        assert edges_low >= edges_high
    
    def test_metadata_completeness(self):
        """Test that metadata contains expected information."""
        data_processor = DataProcessor()
        data = data_processor.generate_synthetic_data(n_samples=50, n_variables=3)
        
        model = SimpleLinearCausalModel(threshold=0.4)
        result = model.fit_discover(data)
        
        expected_keys = {'threshold', 'n_variables', 'n_edges', 'variable_names'}
        assert all(key in result.metadata for key in expected_keys)
        
        assert result.metadata['threshold'] == 0.4
        assert result.metadata['n_variables'] == 3
        assert result.metadata['variable_names'] == list(data.columns)
        assert result.metadata['n_edges'] == np.sum(result.adjacency_matrix)