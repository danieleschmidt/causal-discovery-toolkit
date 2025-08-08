"""Tests for utility functions."""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_processing import DataProcessor
from src.utils.metrics import CausalMetrics


class TestDataProcessor:
    """Tests for DataProcessor class."""
    
    def test_clean_data_drop_na(self):
        """Test cleaning data by dropping NaN values."""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, np.nan, 3, 4],
            'C': [1, 2, 3, 4]
        })
        
        processor = DataProcessor()
        cleaned = processor.clean_data(data, drop_na=True)
        
        assert len(cleaned) == 2  # Only rows 0 and 3 have no NaNs
        assert not cleaned.isnull().any().any()
    
    def test_clean_data_fill_mean(self):
        """Test cleaning data by filling with mean."""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [2, np.nan, 4, 6]
        })
        
        processor = DataProcessor()
        cleaned = processor.clean_data(data, drop_na=False, fill_method='mean')
        
        assert not cleaned.isnull().any().any()
        # Check that NaN in A was filled with mean of [1, 2, 4] = 2.33...
        assert abs(cleaned.loc[2, 'A'] - (1 + 2 + 4) / 3) < 1e-10
    
    def test_standardize_fit(self):
        """Test data standardization with fitting."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        processor = DataProcessor()
        standardized = processor.standardize(data, fit=True)
        
        # Should have zero mean and unit variance (approximately)
        assert abs(standardized['A'].mean()) < 1e-10
        assert abs(standardized['B'].mean()) < 1e-10
        # Use ddof=0 to match sklearn's StandardScaler behavior
        assert abs(standardized['A'].std(ddof=0) - 1.0) < 1e-10
        assert abs(standardized['B'].std(ddof=0) - 1.0) < 1e-10
    
    def test_standardize_transform_only(self):
        """Test standardization transform without fitting."""
        train_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        test_data = pd.DataFrame({
            'A': [2, 3, 4],
            'B': [20, 30, 40]
        })
        
        processor = DataProcessor()
        _ = processor.standardize(train_data, fit=True)
        standardized_test = processor.standardize(test_data, fit=False)
        
        # Test data should be transformed using training data's statistics
        assert len(standardized_test) == 3
        assert not standardized_test.isnull().any().any()
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        processor = DataProcessor()
        data = processor.generate_synthetic_data(
            n_samples=100,
            n_variables=4,
            noise_level=0.1,
            random_state=42
        )
        
        assert data.shape == (100, 4)
        assert list(data.columns) == ['X1', 'X2', 'X3', 'X4']
        assert not data.isnull().any().any()
        
        # Check that later variables are correlated with earlier ones
        corr_matrix = data.corr()
        assert corr_matrix.loc['X1', 'X2'] > 0.5  # Should be strongly correlated
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        processor = DataProcessor()
        data = processor.generate_synthetic_data(n_samples=50, n_variables=3)
        
        is_valid, issues = processor.validate_data(data)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_data_invalid(self):
        """Test data validation with various invalid cases."""
        processor = DataProcessor()
        
        # Test with non-DataFrame
        is_valid, issues = processor.validate_data("not a dataframe")
        assert not is_valid
        assert any("DataFrame" in issue for issue in issues)
        
        # Test with empty DataFrame
        is_valid, issues = processor.validate_data(pd.DataFrame())
        assert not is_valid
        assert any("empty" in issue for issue in issues)
        
        # Test with too few samples
        small_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        is_valid, issues = processor.validate_data(small_data)
        assert not is_valid
        assert any("Too few samples" in issue for issue in issues)
        
        # Test with non-numeric data
        text_data = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        is_valid, issues = processor.validate_data(text_data)
        assert not is_valid
        assert any("Non-numeric" in issue for issue in issues)


class TestCausalMetrics:
    """Tests for CausalMetrics class."""
    
    def test_structural_hamming_distance(self):
        """Test structural Hamming distance calculation."""
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred_adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        
        shd = CausalMetrics.structural_hamming_distance(true_adj, pred_adj)
        
        # Differences: (0,2) and (1,2) positions
        assert shd == 2
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 score calculation."""
        # Perfect prediction
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        metrics = CausalMetrics.precision_recall_f1(true_adj, pred_adj)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        
        # Test with some errors
        pred_adj_errors = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        metrics_errors = CausalMetrics.precision_recall_f1(true_adj, pred_adj_errors)
        
        # Only found 1 out of 2 true edges, but added 1 false positive
        assert metrics_errors['recall'] == 0.5  # Found 1/2 true edges
        assert metrics_errors['precision'] == 0.5  # 1 true positive out of 2 predicted edges
        assert 0 < metrics_errors['f1_score'] < 1.0
    
    def test_true_positive_rate(self):
        """Test true positive rate calculation."""
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 2 true edges
        pred_adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # Found 1 true edge, 1 false positive
        
        tpr = CausalMetrics.true_positive_rate(true_adj, pred_adj)
        
        assert tpr == 0.5  # Found 1 out of 2 true edges
    
    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 2 true edges, 7 true non-edges
        pred_adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # 1 false positive
        
        fpr = CausalMetrics.false_positive_rate(true_adj, pred_adj)
        
        assert abs(fpr - 1/7) < 1e-10  # 1 false positive out of 7 true non-edges
    
    def test_evaluate_discovery_comprehensive(self):
        """Test comprehensive evaluation function."""
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred_adj = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        confidence_scores = np.array([[0, 0.8, 0.6], [0, 0, 0.2], [0, 0, 0]])
        
        results = CausalMetrics.evaluate_discovery(
            true_adj, pred_adj, confidence_scores
        )
        
        # Check that all expected keys are present
        expected_keys = {
            'structural_hamming_distance', 'precision', 'recall', 'f1_score',
            'true_positive_rate', 'false_positive_rate', 'n_true_edges',
            'n_predicted_edges', 'mean_confidence', 'std_confidence'
        }
        assert all(key in results for key in expected_keys)
        
        # Check some specific values
        assert results['n_true_edges'] == 2
        assert results['n_predicted_edges'] == 2
        assert results['structural_hamming_distance'] == 2
        assert results['mean_confidence'] > 0
    
    def test_edge_case_no_edges(self):
        """Test metrics when no edges exist."""
        true_adj = np.zeros((3, 3))
        pred_adj = np.zeros((3, 3))
        
        metrics = CausalMetrics.precision_recall_f1(true_adj, pred_adj)
        
        # When no edges exist and none predicted, should be perfect
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_mismatched_shapes(self):
        """Test error handling for mismatched matrix shapes."""
        true_adj = np.array([[0, 1], [0, 0]])
        pred_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        with pytest.raises(ValueError):
            CausalMetrics.structural_hamming_distance(true_adj, pred_adj)
        
        with pytest.raises(ValueError):
            CausalMetrics.precision_recall_f1(true_adj, pred_adj)