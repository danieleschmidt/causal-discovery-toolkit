"""Bayesian Network-based Causal Discovery Algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from .base import CausalDiscoveryModel, CausalResult


class BayesianNetworkDiscovery(CausalDiscoveryModel):
    """Bayesian Network approach for causal discovery using score-based methods."""
    
    def __init__(self, 
                 score_method: str = 'bic',
                 max_parents: int = 3,
                 use_bootstrap: bool = True,
                 bootstrap_samples: int = 100,
                 n_jobs: int = -1,
                 **kwargs):
        super().__init__(
            score_method=score_method,
            max_parents=max_parents,
            use_bootstrap=use_bootstrap,
            bootstrap_samples=bootstrap_samples,
            n_jobs=n_jobs,
            **kwargs
        )
        self.score_method = score_method
        self.max_parents = max_parents
        self.use_bootstrap = use_bootstrap
        self.bootstrap_samples = bootstrap_samples
        self.n_jobs = n_jobs
        self._data = None
        self._variable_names = None
        
    def fit(self, data: pd.DataFrame) -> 'BayesianNetworkDiscovery':
        """Fit the Bayesian Network model to data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        if data.isnull().any().any():
            warnings.warn("Data contains missing values. Consider preprocessing.")
            
        self._data = data.copy()
        self._variable_names = list(data.columns)
        self.is_fitted = True
        return self
        
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using Bayesian Networks."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
            
        if data is None:
            data = self._data
            
        n_vars = len(data.columns)
        
        if self.use_bootstrap:
            # Bootstrap ensemble for robust discovery
            bootstrap_results = []
            
            for _ in range(self.bootstrap_samples):
                # Bootstrap sample
                boot_data = data.sample(n=len(data), replace=True)
                adj_matrix = self._discover_single(boot_data)
                bootstrap_results.append(adj_matrix)
            
            # Aggregate bootstrap results
            final_adjacency = np.mean(bootstrap_results, axis=0)
            confidence_scores = np.std(bootstrap_results, axis=0)
            
            # Threshold to create binary adjacency matrix
            threshold = 0.5
            binary_adjacency = (final_adjacency > threshold).astype(int)
            
        else:
            binary_adjacency = self._discover_single(data)
            confidence_scores = binary_adjacency.astype(float)
            
        return CausalResult(
            adjacency_matrix=binary_adjacency,
            confidence_scores=confidence_scores,
            method_used="BayesianNetwork",
            metadata={
                "score_method": self.score_method,
                "max_parents": self.max_parents,
                "n_variables": n_vars,
                "n_edges": np.sum(binary_adjacency),
                "variable_names": list(data.columns),
                "bootstrap_used": self.use_bootstrap,
                "bootstrap_samples": self.bootstrap_samples if self.use_bootstrap else 0
            }
        )
    
    def _discover_single(self, data: pd.DataFrame) -> np.ndarray:
        """Run single causal discovery without bootstrap."""
        n_vars = len(data.columns)
        best_adjacency = np.zeros((n_vars, n_vars))
        
        # Score-based structure learning
        for child in range(n_vars):
            best_parents = self._find_best_parents(data, child)
            for parent in best_parents:
                best_adjacency[parent, child] = 1
                
        return best_adjacency
    
    def _find_best_parents(self, data: pd.DataFrame, child_idx: int) -> List[int]:
        """Find best parent set for a given child variable."""
        n_vars = len(data.columns)
        candidate_parents = [i for i in range(n_vars) if i != child_idx]
        
        best_score = float('-inf')
        best_parents = []
        
        # Try all possible parent combinations up to max_parents
        for r in range(min(self.max_parents + 1, len(candidate_parents) + 1)):
            for parent_set in itertools.combinations(candidate_parents, r):
                score = self._score_parent_set(data, child_idx, list(parent_set))
                if score > best_score:
                    best_score = score
                    best_parents = list(parent_set)
                    
        return best_parents
    
    def _score_parent_set(self, data: pd.DataFrame, child_idx: int, parent_indices: List[int]) -> float:
        """Score a parent set for a child variable."""
        if self.score_method == 'bic':
            return self._bic_score(data, child_idx, parent_indices)
        elif self.score_method == 'aic':
            return self._aic_score(data, child_idx, parent_indices)
        else:
            raise ValueError(f"Unknown score method: {self.score_method}")
    
    def _bic_score(self, data: pd.DataFrame, child_idx: int, parent_indices: List[int]) -> float:
        """Calculate BIC score for given parent-child relationship."""
        n_samples = len(data)
        
        if not parent_indices:
            # No parents - just variance of child
            child_data = data.iloc[:, child_idx]
            residual_ss = np.var(child_data) * n_samples
            n_params = 2  # mean and variance
        else:
            # Linear regression with parents
            X = data.iloc[:, parent_indices].values
            y = data.iloc[:, child_idx].values
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            
            try:
                # Solve normal equations
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ beta
                residual_ss = np.sum((y - y_pred) ** 2)
                n_params = len(beta) + 1  # regression coefficients + error variance
            except np.linalg.LinAlgError:
                # Singular matrix - penalize heavily
                return float('-inf')
        
        # BIC = -2 * log_likelihood + k * log(n)
        log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(residual_ss / n_samples) + 1)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return -bic  # Return negative BIC for maximization
    
    def _aic_score(self, data: pd.DataFrame, child_idx: int, parent_indices: List[int]) -> float:
        """Calculate AIC score for given parent-child relationship."""
        n_samples = len(data)
        
        if not parent_indices:
            child_data = data.iloc[:, child_idx]
            residual_ss = np.var(child_data) * n_samples
            n_params = 2
        else:
            X = data.iloc[:, parent_indices].values
            y = data.iloc[:, child_idx].values
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            
            try:
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ beta
                residual_ss = np.sum((y - y_pred) ** 2)
                n_params = len(beta) + 1
            except np.linalg.LinAlgError:
                return float('-inf')
        
        # AIC = -2 * log_likelihood + 2 * k
        log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(residual_ss / n_samples) + 1)
        aic = -2 * log_likelihood + 2 * n_params
        
        return -aic  # Return negative AIC for maximization


class ConstraintBasedDiscovery(CausalDiscoveryModel):
    """Constraint-based causal discovery using conditional independence tests."""
    
    def __init__(self, 
                 independence_test: str = 'correlation',
                 alpha: float = 0.05,
                 max_conditioning_set_size: int = 3,
                 **kwargs):
        super().__init__(
            independence_test=independence_test,
            alpha=alpha,
            max_conditioning_set_size=max_conditioning_set_size,
            **kwargs
        )
        self.independence_test = independence_test
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self._data = None
        
    def fit(self, data: pd.DataFrame) -> 'ConstraintBasedDiscovery':
        """Fit the constraint-based model to data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        self._data = data.copy()
        self.is_fitted = True
        return self
        
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships using constraint-based approach."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
            
        if data is None:
            data = self._data
            
        n_vars = len(data.columns)
        
        # Start with complete graph
        adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        confidence_scores = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Remove edges based on conditional independence tests
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] == 1:  # Edge still exists
                    # Test independence with increasing conditioning set sizes
                    independent = False
                    
                    for cond_size in range(self.max_conditioning_set_size + 1):
                        if independent:
                            break
                            
                        # Get all possible conditioning sets of current size
                        other_vars = [k for k in range(n_vars) if k != i and k != j]
                        
                        if cond_size == 0:
                            # Test marginal independence
                            p_value = self._independence_test(data, i, j, [])
                            if p_value > self.alpha:
                                independent = True
                                confidence_scores[i, j] = confidence_scores[j, i] = 1 - p_value
                        else:
                            if len(other_vars) >= cond_size:
                                for conditioning_set in itertools.combinations(other_vars, cond_size):
                                    p_value = self._independence_test(data, i, j, list(conditioning_set))
                                    if p_value > self.alpha:
                                        independent = True
                                        confidence_scores[i, j] = confidence_scores[j, i] = 1 - p_value
                                        break
                    
                    if independent:
                        adjacency[i, j] = adjacency[j, i] = 0
        
        return CausalResult(
            adjacency_matrix=adjacency.astype(int),
            confidence_scores=confidence_scores,
            method_used="ConstraintBased",
            metadata={
                "independence_test": self.independence_test,
                "alpha": self.alpha,
                "max_conditioning_set_size": self.max_conditioning_set_size,
                "n_variables": n_vars,
                "n_edges": np.sum(adjacency),
                "variable_names": list(data.columns)
            }
        )
    
    def _independence_test(self, data: pd.DataFrame, var1: int, var2: int, 
                          conditioning_set: List[int]) -> float:
        """Test conditional independence between two variables."""
        if self.independence_test == 'correlation':
            return self._partial_correlation_test(data, var1, var2, conditioning_set)
        else:
            raise ValueError(f"Unknown independence test: {self.independence_test}")
    
    def _partial_correlation_test(self, data: pd.DataFrame, var1: int, var2: int,
                                 conditioning_set: List[int]) -> float:
        """Partial correlation test for conditional independence."""
        n_samples = len(data)
        
        if not conditioning_set:
            # Marginal correlation
            corr = np.corrcoef(data.iloc[:, var1], data.iloc[:, var2])[0, 1]
            
            # Fisher z-transformation for significance testing
            if abs(corr) >= 0.9999:
                return 0.0  # Perfect correlation
            
            z = 0.5 * np.log((1 + corr) / (1 - corr))
            z_stat = z * np.sqrt(n_samples - 3)
            p_value = 2 * (1 - self._standard_normal_cdf(abs(z_stat)))
            
        else:
            # Partial correlation
            all_vars = [var1, var2] + conditioning_set
            corr_matrix = data.iloc[:, all_vars].corr().values
            
            try:
                # Compute partial correlation using matrix inversion
                inv_corr = np.linalg.inv(corr_matrix)
                partial_corr = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1])
                
                if abs(partial_corr) >= 0.9999:
                    return 0.0
                
                z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
                z_stat = z * np.sqrt(n_samples - len(conditioning_set) - 3)
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_stat)))
                
            except np.linalg.LinAlgError:
                # Singular matrix - assume independence
                return 1.0
        
        return max(p_value, 1e-10)  # Avoid numerical issues
    
    def _standard_normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF using error function."""
        return 0.5 * (1 + self._erf(x / np.sqrt(2)))
    
    def _erf(self, x: float) -> float:
        """Approximation of error function."""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y