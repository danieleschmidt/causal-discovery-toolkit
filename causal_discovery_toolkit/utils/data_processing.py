"""Data processing utilities for causal discovery."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Utility class for preprocessing data for causal discovery."""
    
    def __init__(self):
        self._scaler = StandardScaler()
        self._is_fitted = False
    
    def clean_data(self, data: pd.DataFrame, 
                   drop_na: bool = True,
                   fill_method: Optional[str] = None) -> pd.DataFrame:
        """Clean data by handling missing values.
        
        Args:
            data: Input DataFrame
            drop_na: Whether to drop rows with NaN values
            fill_method: Method to fill NaN values ('mean', 'median', 'forward', 'backward')
            
        Returns:
            Cleaned DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        cleaned = data.copy()
        
        if drop_na:
            cleaned = cleaned.dropna()
        elif fill_method:
            if fill_method == 'mean':
                cleaned = cleaned.fillna(cleaned.mean())
            elif fill_method == 'median':
                cleaned = cleaned.fillna(cleaned.median())
            elif fill_method == 'forward':
                cleaned = cleaned.fillna(method='ffill')
            elif fill_method == 'backward':
                cleaned = cleaned.fillna(method='bfill')
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
                
        return cleaned
    
    def standardize(self, data: pd.DataFrame, 
                   fit: bool = True) -> pd.DataFrame:
        """Standardize data to zero mean and unit variance.
        
        Args:
            data: Input DataFrame
            fit: Whether to fit the scaler on this data
            
        Returns:
            Standardized DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if fit:
            scaled_values = self._scaler.fit_transform(data)
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise RuntimeError("Scaler must be fitted first")
            scaled_values = self._scaler.transform(data)
            
        return pd.DataFrame(
            scaled_values, 
            columns=data.columns,
            index=data.index
        )
    
    def generate_synthetic_data(self, 
                              n_samples: int = 1000,
                              n_variables: int = 5,
                              noise_level: float = 0.1,
                              random_state: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data with known causal structure.
        
        Args:
            n_samples: Number of samples to generate
            n_variables: Number of variables
            noise_level: Level of noise to add
            random_state: Random seed for reproducibility
            
        Returns:
            Synthetic DataFrame with causal structure
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate base random variables
        data = {}
        
        # Create variables with simple causal chain: X1 -> X2 -> X3 etc.
        for i in range(n_variables):
            var_name = f'X{i+1}'
            
            if i == 0:
                # Root variable
                data[var_name] = np.random.randn(n_samples)
            else:
                # Causally dependent on previous variable
                prev_var = f'X{i}'
                data[var_name] = (0.7 * data[prev_var] + 
                                 noise_level * np.random.randn(n_samples))
        
        return pd.DataFrame(data)
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data for causal discovery.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not isinstance(data, pd.DataFrame):
            issues.append("Input must be a pandas DataFrame")
            return False, issues
            
        if data.empty:
            issues.append("Data is empty")
            
        if data.shape[0] < 10:
            issues.append("Too few samples (minimum 10 required)")
            
        if data.shape[1] < 2:
            issues.append("Too few variables (minimum 2 required)")
            
        # Check for non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            issues.append(f"Non-numeric columns found: {non_numeric}")
            
        # Check for constant columns
        constant_cols = data.columns[data.nunique() <= 1].tolist()
        if constant_cols:
            issues.append(f"Constant columns found: {constant_cols}")
            
        # Check for high correlation with identical variables
        if data.shape[1] > 1:
            try:
                corr_matrix = data.corr()
                # Find perfect correlations (excluding diagonal)
                perfect_corr = np.where((np.abs(corr_matrix) > 0.999) & 
                                      (corr_matrix != 1.0))
                if len(perfect_corr[0]) > 0:
                    issues.append("Nearly identical variables found")
            except (ValueError, TypeError):
                # Skip correlation check for non-numeric data
                pass
        
        return len(issues) == 0, issues