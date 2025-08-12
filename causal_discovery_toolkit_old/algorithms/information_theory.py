"""Information Theory-based Causal Discovery Algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import entropy

from .base import CausalDiscoveryModel, CausalResult


class MutualInformationDiscovery(CausalDiscoveryModel):
    """Mutual Information-based causal discovery."""
    
    def __init__(self, 
                 threshold: float = 0.1,
                 discretization_method: str = 'equal_width',
                 n_bins: int = 10,
                 use_conditional_mi: bool = True,
                 **kwargs):
        super().__init__(
            threshold=threshold,
            discretization_method=discretization_method,
            n_bins=n_bins,
            use_conditional_mi=use_conditional_mi,
            **kwargs
        )
        self.threshold = threshold
        self.discretization_method = discretization_method
        self.n_bins = n_bins
        self.use_conditional_mi = use_conditional_mi
        self._data = None
        
    def fit(self, data: pd.DataFrame) -> 'MutualInformationDiscovery':
        """Fit the mutual information model to data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        # Discretize continuous data
        self._data = self._discretize_data(data)
        self.is_fitted = True
        return self
        
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using mutual information."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
            
        if data is not None:
            data_to_use = self._discretize_data(data)
        else:
            data_to_use = self._data
            
        n_vars = len(data_to_use.columns)
        adjacency = np.zeros((n_vars, n_vars))
        confidence_scores = np.zeros((n_vars, n_vars))
        
        # Compute mutual information matrix
        mi_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    mi_value = self._mutual_information(
                        data_to_use.iloc[:, i],
                        data_to_use.iloc[:, j]
                    )
                    mi_matrix[i, j] = mi_value
                    
        # Apply threshold to create adjacency matrix
        adjacency = (mi_matrix > self.threshold).astype(int)
        confidence_scores = mi_matrix
        
        # Optional: Use conditional mutual information to refine edges
        if self.use_conditional_mi and n_vars > 2:
            adjacency, confidence_scores = self._refine_with_conditional_mi(
                data_to_use, adjacency, confidence_scores
            )
        
        return CausalResult(
            adjacency_matrix=adjacency,
            confidence_scores=confidence_scores,
            method_used="MutualInformation",
            metadata={
                "threshold": self.threshold,
                "discretization_method": self.discretization_method,
                "n_bins": self.n_bins,
                "use_conditional_mi": self.use_conditional_mi,
                "n_variables": n_vars,
                "n_edges": np.sum(adjacency),
                "variable_names": list(data_to_use.columns)
            }
        )
    
    def _discretize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous data for mutual information computation."""
        discretized = data.copy()
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'int32']:
                # Already discrete
                continue
                
            if self.discretization_method == 'equal_width':
                discretized[column] = pd.cut(
                    data[column], 
                    bins=self.n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
            elif self.discretization_method == 'equal_frequency':
                discretized[column] = pd.qcut(
                    data[column], 
                    q=self.n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
            else:
                raise ValueError(f"Unknown discretization method: {self.discretization_method}")
        
        return discretized.fillna(0)  # Handle any NaN values from discretization
    
    def _mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Compute mutual information between two discrete variables."""
        # Create contingency table
        contingency = pd.crosstab(x, y, normalize=False).values
        
        # Add small constant to avoid log(0)
        contingency = contingency + 1e-10
        
        # Normalize to get probabilities
        p_xy = contingency / contingency.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def _conditional_mutual_information(self, x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        """Compute conditional mutual information I(X;Y|Z)."""
        # Create three-way contingency table
        data_combined = pd.DataFrame({'x': x, 'y': y, 'z': z})
        contingency_xyz = data_combined.groupby(['x', 'y', 'z']).size().unstack(fill_value=0).unstack(fill_value=0)
        
        # Flatten and add small constant
        contingency_xyz = contingency_xyz.values.flatten() + 1e-10
        n_total = contingency_xyz.sum()
        
        # Get marginal and conditional probabilities
        p_xyz = contingency_xyz / n_total
        
        # Compute conditional MI using the definition
        cmi = 0.0
        unique_z = z.unique()
        
        for z_val in unique_z:
            z_mask = (z == z_val)
            if z_mask.sum() == 0:
                continue
                
            x_given_z = x[z_mask]
            y_given_z = y[z_mask]
            
            if len(x_given_z) > 0 and len(y_given_z) > 0:
                p_z = z_mask.sum() / len(z)
                mi_xy_given_z = self._mutual_information(x_given_z, y_given_z)
                cmi += p_z * mi_xy_given_z
        
        return cmi
    
    def _refine_with_conditional_mi(self, data: pd.DataFrame, adjacency: np.ndarray, 
                                   confidence_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine adjacency matrix using conditional mutual information."""
        n_vars = len(data.columns)
        refined_adjacency = adjacency.copy()
        refined_confidence = confidence_scores.copy()
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and adjacency[i, j] == 1:
                    # Test if edge i->j should be removed due to confounding
                    max_cmi = 0.0
                    
                    for k in range(n_vars):
                        if k != i and k != j:
                            cmi = self._conditional_mutual_information(
                                data.iloc[:, i],
                                data.iloc[:, j],
                                data.iloc[:, k]
                            )
                            max_cmi = max(max_cmi, cmi)
                    
                    # If conditional MI is significantly lower than MI, remove edge
                    original_mi = confidence_scores[i, j]
                    if max_cmi < original_mi * 0.5:  # Threshold for removal
                        refined_adjacency[i, j] = 0
                        refined_confidence[i, j] = max_cmi
        
        return refined_adjacency, refined_confidence


class TransferEntropyDiscovery(CausalDiscoveryModel):
    """Transfer Entropy-based causal discovery for temporal data."""
    
    def __init__(self, 
                 threshold: float = 0.05,
                 lag: int = 1,
                 discretization_method: str = 'equal_width',
                 n_bins: int = 10,
                 **kwargs):
        super().__init__(
            threshold=threshold,
            lag=lag,
            discretization_method=discretization_method,
            n_bins=n_bins,
            **kwargs
        )
        self.threshold = threshold
        self.lag = lag
        self.discretization_method = discretization_method
        self.n_bins = n_bins
        self._data = None
        
    def fit(self, data: pd.DataFrame) -> 'TransferEntropyDiscovery':
        """Fit the transfer entropy model to data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        if len(data) <= self.lag:
            raise ValueError(f"Data length must be greater than lag ({self.lag})")
            
        self._data = data.copy()
        self.is_fitted = True
        return self
        
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using transfer entropy."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
            
        if data is not None:
            data_to_use = data.copy()
        else:
            data_to_use = self._data
            
        n_vars = len(data_to_use.columns)
        adjacency = np.zeros((n_vars, n_vars))
        confidence_scores = np.zeros((n_vars, n_vars))
        
        # Discretize data
        discretized_data = self._discretize_data(data_to_use)
        
        # Compute transfer entropy matrix
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    te = self._transfer_entropy(
                        discretized_data.iloc[:, i],  # Source
                        discretized_data.iloc[:, j],  # Target
                        self.lag
                    )
                    confidence_scores[i, j] = te
                    
                    if te > self.threshold:
                        adjacency[i, j] = 1
        
        return CausalResult(
            adjacency_matrix=adjacency.astype(int),
            confidence_scores=confidence_scores,
            method_used="TransferEntropy",
            metadata={
                "threshold": self.threshold,
                "lag": self.lag,
                "discretization_method": self.discretization_method,
                "n_bins": self.n_bins,
                "n_variables": n_vars,
                "n_edges": np.sum(adjacency),
                "variable_names": list(data_to_use.columns)
            }
        )
    
    def _discretize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous data."""
        discretized = data.copy()
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'int32']:
                continue
                
            if self.discretization_method == 'equal_width':
                discretized[column] = pd.cut(
                    data[column], 
                    bins=self.n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
            elif self.discretization_method == 'equal_frequency':
                discretized[column] = pd.qcut(
                    data[column], 
                    q=self.n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
        
        return discretized.fillna(0)
    
    def _transfer_entropy(self, source: pd.Series, target: pd.Series, lag: int) -> float:
        """Compute transfer entropy from source to target with given lag."""
        n = len(source)
        if n <= lag:
            return 0.0
        
        # Create lagged series
        target_present = target.iloc[lag:]
        target_past = target.iloc[:-lag]
        source_past = source.iloc[:-lag]
        
        # Reset indices to align series
        target_present = target_present.reset_index(drop=True)
        target_past = target_past.reset_index(drop=True)
        source_past = source_past.reset_index(drop=True)
        
        # Compute conditional entropies
        # H(X_t | X_t-1)
        h_target_given_target_past = self._conditional_entropy(target_present, target_past)
        
        # H(X_t | X_t-1, Y_t-1)
        combined_past = pd.DataFrame({
            'target_past': target_past,
            'source_past': source_past
        })
        h_target_given_both_past = self._conditional_entropy(target_present, combined_past)
        
        # Transfer entropy = H(X_t | X_t-1) - H(X_t | X_t-1, Y_t-1)
        te = h_target_given_target_past - h_target_given_both_past
        
        return max(0.0, te)  # TE should be non-negative
    
    def _conditional_entropy(self, x: pd.Series, y) -> float:
        """Compute conditional entropy H(X|Y)."""
        if isinstance(y, pd.DataFrame):
            # Multiple conditioning variables
            y_combined = y.apply(lambda row: ''.join(row.astype(str)), axis=1)
        else:
            y_combined = y
        
        # Create joint distribution
        joint_data = pd.DataFrame({'x': x, 'y': y_combined})
        joint_counts = joint_data.groupby(['x', 'y']).size()
        y_counts = joint_data.groupby('y').size()
        
        conditional_entropy = 0.0
        
        for y_val in y_combined.unique():
            p_y = y_counts[y_val] / len(y_combined)
            
            # Get conditional distribution of X given Y=y_val
            x_given_y = x[y_combined == y_val]
            
            if len(x_given_y) > 0:
                # Compute entropy of X|Y=y_val
                x_probs = x_given_y.value_counts(normalize=True)
                h_x_given_y = -np.sum(x_probs * np.log2(x_probs + 1e-10))
                
                conditional_entropy += p_y * h_x_given_y
        
        return conditional_entropy