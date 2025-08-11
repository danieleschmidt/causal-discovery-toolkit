"""Robust ensemble causal discovery methods."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.robust_validation import RobustValidationSuite
    from ..utils.error_recovery import resilient_causal_discovery, ProgressiveExecution
except ImportError:
    from base import CausalDiscoveryModel, CausalResult
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        from robust_validation import RobustValidationSuite
        from error_recovery import resilient_causal_discovery, ProgressiveExecution
    except ImportError:
        # Minimal fallback implementations
        class RobustValidationSuite:
            def validate_preprocessing(self, data): return []
            def validate_parameters(self, method, params): return type('obj', (object,), {'is_valid': True, 'message': 'OK'})
        def resilient_causal_discovery(recovery_enabled=True): return lambda x: x
        class ProgressiveExecution:
            def __init__(self): pass
            def add_strategy(self, name, params, desc=""): pass

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult(CausalResult):
    """Enhanced result with ensemble information."""
    individual_results: List[CausalResult] = None
    ensemble_method: str = "majority_vote"
    agreement_scores: np.ndarray = None
    method_weights: Dict[str, float] = None


class RobustEnsembleDiscovery(CausalDiscoveryModel):
    """Robust ensemble causal discovery combining multiple methods."""
    
    def __init__(self, 
                 base_models: List[Tuple[CausalDiscoveryModel, str, float]] = None,
                 ensemble_method: str = "weighted_vote",
                 consensus_threshold: float = 0.5,
                 enable_validation: bool = True,
                 parallel_execution: bool = True,
                 **kwargs):
        """
        Initialize robust ensemble.
        
        Args:
            base_models: List of (model, name, weight) tuples
            ensemble_method: Method for combining results
            consensus_threshold: Threshold for edge consensus
            enable_validation: Enable robust validation
            parallel_execution: Execute models in parallel
        """
        super().__init__(**kwargs)
        self.base_models = base_models or []
        self.ensemble_method = ensemble_method
        self.consensus_threshold = consensus_threshold
        self.enable_validation = enable_validation
        self.parallel_execution = parallel_execution
        self._data = None
        
        if self.enable_validation:
            self.validator = RobustValidationSuite()
    
    def add_base_model(self, model: CausalDiscoveryModel, name: str, weight: float = 1.0):
        """Add a base model to the ensemble."""
        self.base_models.append((model, name, weight))
        logger.info(f"Added base model: {name} (weight: {weight})")
    
    @resilient_causal_discovery(recovery_enabled=True)
    def fit(self, data: pd.DataFrame) -> 'RobustEnsembleDiscovery':
        """Fit ensemble models with robust validation."""
        
        # Validation if enabled
        if self.enable_validation:
            validation_results = self.validator.validate_preprocessing(data)
            
            for result in validation_results:
                if not result.is_valid:
                    raise ValueError(f"Validation failed: {result.message}")
                elif result.severity == "warning":
                    warnings.warn(f"Validation warning: {result.message}")
        
        self._data = data.copy()
        
        # Fit all base models
        for model, name, weight in self.base_models:
            try:
                logger.debug(f"Fitting {name}...")
                model.fit(data)
                logger.debug(f"✅ {name} fitted successfully")
            except Exception as e:
                logger.error(f"❌ Failed to fit {name}: {str(e)}")
                # Continue with other models
        
        self.is_fitted = True
        return self
    
    @resilient_causal_discovery(recovery_enabled=True)
    def discover(self, data: Optional[pd.DataFrame] = None) -> EnsembleResult:
        """Discover causal relationships using ensemble approach."""
        
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before discovery")
        
        data_to_use = data if data is not None else self._data
        
        # Execute base models
        individual_results = []
        
        if self.parallel_execution and len(self.base_models) > 1:
            individual_results = self._parallel_discovery(data_to_use)
        else:
            individual_results = self._sequential_discovery(data_to_use)
        
        # Filter successful results
        successful_results = [(result, name, weight) for result, name, weight in individual_results 
                            if result is not None]
        
        if not successful_results:
            raise RuntimeError("No base models produced valid results")
        
        logger.info(f"Successfully executed {len(successful_results)}/{len(self.base_models)} base models")
        
        # Combine results
        ensemble_result = self._combine_results(successful_results, data_to_use)
        
        return ensemble_result
    
    def _sequential_discovery(self, data: pd.DataFrame) -> List[Tuple[Optional[CausalResult], str, float]]:
        """Execute models sequentially."""
        results = []
        
        for model, name, weight in self.base_models:
            try:
                logger.debug(f"Running {name}...")
                start_time = time.time()
                
                result = model.discover(data)
                runtime = time.time() - start_time
                
                logger.debug(f"✅ {name} completed in {runtime:.3f}s")
                results.append((result, name, weight))
                
            except Exception as e:
                logger.warning(f"❌ {name} failed: {str(e)[:50]}...")
                results.append((None, name, weight))
        
        return results
    
    def _parallel_discovery(self, data: pd.DataFrame) -> List[Tuple[Optional[CausalResult], str, float]]:
        """Execute models in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(len(self.base_models), 4)) as executor:
            # Submit all tasks
            future_to_model = {}
            for model, name, weight in self.base_models:
                future = executor.submit(self._safe_discover, model, data)
                future_to_model[future] = (name, weight)
            
            # Collect results
            for future in as_completed(future_to_model):
                name, weight = future_to_model[future]
                try:
                    result = future.result()
                    logger.debug(f"✅ {name} completed")
                    results.append((result, name, weight))
                except Exception as e:
                    logger.warning(f"❌ {name} failed: {str(e)[:50]}...")
                    results.append((None, name, weight))
        
        return results
    
    def _safe_discover(self, model: CausalDiscoveryModel, data: pd.DataFrame) -> CausalResult:
        """Safely execute discovery for a single model."""
        try:
            return model.discover(data)
        except Exception as e:
            logger.debug(f"Model discovery failed: {e}")
            raise
    
    def _combine_results(self, successful_results: List[Tuple[CausalResult, str, float]], 
                        data: pd.DataFrame) -> EnsembleResult:
        """Combine individual results into ensemble result."""
        
        if len(successful_results) == 1:
            # Single result - return as-is but wrapped
            result, name, weight = successful_results[0]
            return EnsembleResult(
                adjacency_matrix=result.adjacency_matrix,
                confidence_scores=result.confidence_scores,
                method_used=f"Ensemble({name})",
                metadata=result.metadata,
                individual_results=[result],
                ensemble_method="single",
                method_weights={name: weight}
            )
        
        # Multiple results - combine based on method
        if self.ensemble_method == "majority_vote":
            return self._majority_vote_combination(successful_results, data)
        elif self.ensemble_method == "weighted_vote":
            return self._weighted_vote_combination(successful_results, data)
        elif self.ensemble_method == "intersection":
            return self._intersection_combination(successful_results, data)
        elif self.ensemble_method == "union":
            return self._union_combination(successful_results, data)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _majority_vote_combination(self, results: List[Tuple[CausalResult, str, float]], 
                                  data: pd.DataFrame) -> EnsembleResult:
        """Combine results using majority vote."""
        
        # Stack adjacency matrices
        adjacency_matrices = [result.adjacency_matrix for result, _, _ in results]
        stacked = np.stack(adjacency_matrices)
        
        # Majority vote
        vote_counts = np.sum(stacked, axis=0)
        ensemble_adjacency = (vote_counts > len(results) / 2).astype(int)
        
        # Confidence is proportion of votes
        ensemble_confidence = vote_counts / len(results)
        
        # Agreement scores
        agreement_scores = np.minimum(vote_counts, len(results) - vote_counts) / len(results)
        
        return EnsembleResult(
            adjacency_matrix=ensemble_adjacency,
            confidence_scores=ensemble_confidence,
            method_used=f"Ensemble(majority_vote)",
            metadata={
                "n_base_models": len(results),
                "consensus_threshold": 0.5,
                "n_variables": ensemble_adjacency.shape[0],
                "n_edges": int(np.sum(ensemble_adjacency)),
                "variable_names": list(data.columns)
            },
            individual_results=[result for result, _, _ in results],
            ensemble_method="majority_vote",
            agreement_scores=agreement_scores,
            method_weights={name: 1.0 for _, name, _ in results}
        )
    
    def _weighted_vote_combination(self, results: List[Tuple[CausalResult, str, float]], 
                                  data: pd.DataFrame) -> EnsembleResult:
        """Combine results using weighted vote."""
        
        # Weighted combination
        total_weight = sum(weight for _, _, weight in results)
        normalized_weights = [weight / total_weight for _, _, weight in results]
        
        # Weighted adjacency
        weighted_sum = np.zeros_like(results[0][0].adjacency_matrix, dtype=float)
        for (result, _, _), weight in zip(results, normalized_weights):
            weighted_sum += weight * result.adjacency_matrix
        
        # Threshold for final adjacency
        ensemble_adjacency = (weighted_sum > self.consensus_threshold).astype(int)
        ensemble_confidence = weighted_sum
        
        return EnsembleResult(
            adjacency_matrix=ensemble_adjacency,
            confidence_scores=ensemble_confidence,
            method_used=f"Ensemble(weighted_vote)",
            metadata={
                "n_base_models": len(results),
                "consensus_threshold": self.consensus_threshold,
                "n_variables": ensemble_adjacency.shape[0],
                "n_edges": int(np.sum(ensemble_adjacency)),
                "variable_names": list(data.columns)
            },
            individual_results=[result for result, _, _ in results],
            ensemble_method="weighted_vote",
            agreement_scores=np.abs(weighted_sum - 0.5) * 2,  # Distance from uncertain
            method_weights={name: weight for _, name, weight in results}
        )
    
    def _intersection_combination(self, results: List[Tuple[CausalResult, str, float]], 
                                 data: pd.DataFrame) -> EnsembleResult:
        """Combine results using intersection (conservative)."""
        
        adjacency_matrices = [result.adjacency_matrix for result, _, _ in results]
        
        # Intersection - all must agree
        ensemble_adjacency = np.ones_like(adjacency_matrices[0])
        for adj in adjacency_matrices:
            ensemble_adjacency = np.logical_and(ensemble_adjacency, adj).astype(int)
        
        # Confidence is minimum confidence across methods
        confidence_scores = [result.confidence_scores for result, _, _ in results]
        ensemble_confidence = np.minimum.reduce(confidence_scores)
        
        return EnsembleResult(
            adjacency_matrix=ensemble_adjacency,
            confidence_scores=ensemble_confidence,
            method_used=f"Ensemble(intersection)",
            metadata={
                "n_base_models": len(results),
                "combination_method": "intersection",
                "n_variables": ensemble_adjacency.shape[0],
                "n_edges": int(np.sum(ensemble_adjacency)),
                "variable_names": list(data.columns)
            },
            individual_results=[result for result, _, _ in results],
            ensemble_method="intersection",
            method_weights={name: weight for _, name, weight in results}
        )
    
    def _union_combination(self, results: List[Tuple[CausalResult, str, float]], 
                          data: pd.DataFrame) -> EnsembleResult:
        """Combine results using union (liberal)."""
        
        adjacency_matrices = [result.adjacency_matrix for result, _, _ in results]
        
        # Union - any method finding an edge includes it
        ensemble_adjacency = np.zeros_like(adjacency_matrices[0])
        for adj in adjacency_matrices:
            ensemble_adjacency = np.logical_or(ensemble_adjacency, adj).astype(int)
        
        # Confidence is maximum confidence across methods
        confidence_scores = [result.confidence_scores for result, _, _ in results]
        ensemble_confidence = np.maximum.reduce(confidence_scores)
        
        return EnsembleResult(
            adjacency_matrix=ensemble_adjacency,
            confidence_scores=ensemble_confidence,
            method_used=f"Ensemble(union)",
            metadata={
                "n_base_models": len(results),
                "combination_method": "union",
                "n_variables": ensemble_adjacency.shape[0],
                "n_edges": int(np.sum(ensemble_adjacency)),
                "variable_names": list(data.columns)
            },
            individual_results=[result for result, _, _ in results],
            ensemble_method="union",
            method_weights={name: weight for _, name, weight in results}
        )


class AdaptiveEnsembleDiscovery(RobustEnsembleDiscovery):
    """Adaptive ensemble that selects best methods based on data characteristics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_characteristics = {}
        
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data to determine optimal methods."""
        characteristics = {}
        
        n_samples, n_vars = data.shape
        characteristics['n_samples'] = n_samples
        characteristics['n_variables'] = n_vars
        characteristics['sample_to_var_ratio'] = n_samples / n_vars
        
        # Check for linearity (correlation vs mutual information)
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            characteristics['max_correlation'] = corr_matrix.max().max()
            characteristics['mean_correlation'] = corr_matrix.mean().mean()
        
        # Data distribution characteristics
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                characteristics[f'{col}_skewness'] = abs(col_data.skew())
                characteristics[f'{col}_kurtosis'] = col_data.kurtosis()
        
        # Temporal characteristics (if time-like data)
        if any('time' in col.lower() or 'date' in col.lower() for col in data.columns):
            characteristics['has_temporal_component'] = True
        else:
            characteristics['has_temporal_component'] = False
        
        return characteristics
    
    def _select_optimal_methods(self, characteristics: Dict[str, Any]) -> List[str]:
        """Select optimal methods based on data characteristics."""
        selected_methods = []
        
        # Rule-based method selection
        if characteristics['sample_to_var_ratio'] > 100:
            # Large sample size relative to variables
            selected_methods.extend(['BayesianNetwork', 'ConstraintBased'])
        
        if characteristics.get('max_correlation', 0) > 0.7:
            # High linear relationships
            selected_methods.extend(['SimpleLinear', 'ConstraintBased'])
        
        if characteristics.get('has_temporal_component', False):
            # Time series data
            selected_methods.append('TransferEntropy')
        
        # Non-linear relationships (high skewness/kurtosis)
        high_skewness_cols = [k for k, v in characteristics.items() 
                             if 'skewness' in k and v > 2.0]
        if high_skewness_cols:
            selected_methods.append('MutualInformation')
        
        # Always include at least one robust method
        if not selected_methods:
            selected_methods = ['SimpleLinear']
        
        return list(set(selected_methods))  # Remove duplicates
    
    def fit(self, data: pd.DataFrame) -> 'AdaptiveEnsembleDiscovery':
        """Fit with adaptive method selection."""
        
        # Analyze data characteristics
        self.data_characteristics = self._analyze_data_characteristics(data)
        logger.info(f"Data characteristics: {self.data_characteristics}")
        
        # Select optimal methods
        optimal_methods = self._select_optimal_methods(self.data_characteristics)
        logger.info(f"Selected optimal methods: {optimal_methods}")
        
        # Configure base models based on selection
        self._configure_adaptive_models(optimal_methods)
        
        # Call parent fit
        return super().fit(data)
    
    def _configure_adaptive_models(self, method_names: List[str]):
        """Configure base models based on selected methods."""
        self.base_models = []
        
        # Import here to avoid circular imports
        try:
            from .base import SimpleLinearCausalModel
            from .bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
            from .information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
            
            model_map = {
                'SimpleLinear': (SimpleLinearCausalModel, {'threshold': 0.3}),
                'BayesianNetwork': (BayesianNetworkDiscovery, {'max_parents': 2, 'use_bootstrap': False}),
                'ConstraintBased': (ConstraintBasedDiscovery, {'alpha': 0.05}),
                'MutualInformation': (MutualInformationDiscovery, {'threshold': 0.1, 'n_bins': 8}),
                'TransferEntropy': (TransferEntropyDiscovery, {'threshold': 0.05, 'lag': 1})
            }
            
            for method_name in method_names:
                if method_name in model_map:
                    model_class, params = model_map[method_name]
                    model = model_class(**params)
                    weight = self._get_method_weight(method_name)
                    self.add_base_model(model, method_name, weight)
                    
        except ImportError as e:
            logger.warning(f"Could not import some models: {e}")
            # Fallback to simple model
            from .base import SimpleLinearCausalModel
            model = SimpleLinearCausalModel(threshold=0.3)
            self.add_base_model(model, "SimpleLinear", 1.0)
    
    def _get_method_weight(self, method_name: str) -> float:
        """Get weight for method based on data characteristics."""
        base_weight = 1.0
        
        # Adjust weights based on data characteristics
        if method_name == 'SimpleLinear':
            if self.data_characteristics.get('max_correlation', 0) > 0.5:
                base_weight *= 1.5
        
        elif method_name == 'BayesianNetwork':
            if self.data_characteristics.get('sample_to_var_ratio', 0) > 50:
                base_weight *= 1.3
        
        elif method_name == 'MutualInformation':
            high_skew_count = sum(1 for k, v in self.data_characteristics.items() 
                                if 'skewness' in k and v > 2.0)
            if high_skew_count > 0:
                base_weight *= 1.2
        
        return base_weight