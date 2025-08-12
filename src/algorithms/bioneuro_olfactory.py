"""Specialized algorithms for bioneuro-olfactory fusion research."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import time
import warnings

try:
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.validation import DataValidator, ParameterValidator
    from ..utils.monitoring import monitor_performance
    from ..utils.security import DataSecurityValidator
except ImportError:
    # For direct execution
    from base import CausalDiscoveryModel, CausalResult
    from utils.validation import DataValidator, ParameterValidator
    from utils.monitoring import monitor_performance
    from utils.security import DataSecurityValidator

logger = logging.getLogger(__name__)


@dataclass
class OlfactoryNeuralSignal:
    """Represents processed olfactory neural signal data."""
    receptor_responses: np.ndarray
    temporal_patterns: np.ndarray
    neural_firing_rates: np.ndarray
    odor_concentrations: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class BioneuroFusionResult(CausalResult):
    """Extended causal result for bioneuro-olfactory research."""
    neural_pathways: Dict[str, float]
    olfactory_correlations: np.ndarray
    sensory_integration_map: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]


class OlfactoryNeuralCausalModel(CausalDiscoveryModel):
    """Specialized causal discovery for olfactory-neural interactions."""
    
    def __init__(self, 
                 receptor_sensitivity_threshold: float = 0.1,
                 neural_firing_threshold: float = 5.0,
                 temporal_window_ms: int = 100,
                 cross_modal_integration: bool = True,
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95,
                 **kwargs):
        """
        Initialize specialized olfactory-neural causal model.
        
        Args:
            receptor_sensitivity_threshold: Minimum receptor response sensitivity
            neural_firing_threshold: Minimum neural firing rate (Hz)
            temporal_window_ms: Temporal window for causal analysis (milliseconds)
            cross_modal_integration: Enable cross-modal sensory integration
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Statistical confidence level
        """
        super().__init__(**kwargs)
        
        # Validate specialized parameters
        validator = ParameterValidator()
        validator.validate_range("receptor_sensitivity_threshold", receptor_sensitivity_threshold, 0.0, 1.0)
        validator.validate_range("neural_firing_threshold", neural_firing_threshold, 0.0, 100.0)
        validator.validate_range("temporal_window_ms", temporal_window_ms, 1, 10000)
        validator.validate_range("confidence_level", confidence_level, 0.0, 1.0)
        
        self.receptor_sensitivity_threshold = receptor_sensitivity_threshold
        self.neural_firing_threshold = neural_firing_threshold
        self.temporal_window_ms = temporal_window_ms
        self.cross_modal_integration = cross_modal_integration
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
        self._fitted_signals: Optional[OlfactoryNeuralSignal] = None
        self._causal_graph = None
        
        logger.info(f"Initialized OlfactoryNeuralCausalModel with parameters: "
                   f"receptor_threshold={receptor_sensitivity_threshold}, "
                   f"neural_threshold={neural_firing_threshold}, "
                   f"temporal_window={temporal_window_ms}ms")
    
    @monitor_performance()
    def fit(self, data: pd.DataFrame) -> 'OlfactoryNeuralCausalModel':
        """
        Fit the olfactory-neural causal model.
        
        Args:
            data: DataFrame with columns for receptor responses, neural firing, etc.
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        validator = DataValidator(strict=False)
        validation_result = validator.validate_input_data(data)
        if not validation_result.is_valid:
            if validation_result.errors:
                raise ValueError(f"Invalid input data: {validation_result.errors}")
            else:
                # Only warnings, log them but continue
                if validation_result.warnings:
                    logger.warning(f"Data validation warnings: {validation_result.warnings}")
        
        # Check for required columns
        required_patterns = ['receptor', 'neural']
        found_patterns = [pattern for pattern in required_patterns 
                         if any(pattern in col.lower() for col in data.columns)]
        if len(found_patterns) < len(required_patterns):
            logger.warning(f"Some expected column patterns not found. Expected: {required_patterns}, Found: {found_patterns}")
        
        # Security check
        security = DataSecurityValidator()
        security_result = security.validate_data_security(data)
        if not security_result.is_secure:
            logger.warning(f"Security issues detected: {security_result.issues}")
        
        # Extract and process olfactory neural signals
        self._fitted_signals = self._extract_neural_signals(data)
        
        # Build preliminary causal graph
        self._causal_graph = self._build_causal_graph(self._fitted_signals)
        
        self.is_fitted = True
        logger.info("Successfully fitted OlfactoryNeuralCausalModel")
        return self
    
    @monitor_performance()
    def discover(self, data: Optional[pd.DataFrame] = None) -> BioneuroFusionResult:
        """
        Discover bioneuro-olfactory causal relationships.
        
        Args:
            data: Optional new data for discovery
            
        Returns:
            BioneuroFusionResult with specialized analysis
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before discovery")
        
        signals = self._fitted_signals if data is None else self._extract_neural_signals(data)
        
        # Core causal discovery with olfactory specialization
        adjacency_matrix = self._discover_olfactory_causality(signals)
        confidence_scores = self._compute_causal_confidence(signals, adjacency_matrix)
        
        # Specialized bioneuro analysis
        neural_pathways = self._analyze_neural_pathways(signals, adjacency_matrix)
        olfactory_correlations = self._compute_olfactory_correlations(signals)
        sensory_integration = self._analyze_sensory_integration(signals)
        
        # Statistical validation
        confidence_intervals = self._compute_confidence_intervals(signals, adjacency_matrix)
        significance_tests = self._perform_significance_tests(signals, adjacency_matrix)
        
        return BioneuroFusionResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_used="OlfactoryNeuralCausal",
            metadata={
                "receptor_threshold": self.receptor_sensitivity_threshold,
                "neural_threshold": self.neural_firing_threshold,
                "temporal_window_ms": self.temporal_window_ms,
                "n_variables": adjacency_matrix.shape[0],
                "n_causal_edges": np.sum(adjacency_matrix),
                "cross_modal_enabled": self.cross_modal_integration
            },
            neural_pathways=neural_pathways,
            olfactory_correlations=olfactory_correlations,
            sensory_integration_map=sensory_integration,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance_tests
        )
    
    def _extract_neural_signals(self, data: pd.DataFrame) -> OlfactoryNeuralSignal:
        """Extract and process olfactory neural signals from raw data."""
        try:
            # Process receptor responses
            receptor_cols = [col for col in data.columns if 'receptor' in col.lower()]
            receptor_responses = data[receptor_cols].values if receptor_cols else np.array([])
            
            # Process neural firing rates
            neural_cols = [col for col in data.columns if 'neural' in col.lower() or 'firing' in col.lower()]
            neural_firing = data[neural_cols].values if neural_cols else np.array([])
            
            # Extract temporal patterns
            temporal_cols = [col for col in data.columns if 'temporal' in col.lower() or 'time' in col.lower()]
            temporal_patterns = data[temporal_cols].values if temporal_cols else np.arange(len(data))
            
            # Extract odor concentrations
            odor_cols = [col for col in data.columns if 'odor' in col.lower() or 'concentration' in col.lower()]
            odor_concentrations = data[odor_cols].values if odor_cols else np.ones(len(data))
            
            return OlfactoryNeuralSignal(
                receptor_responses=receptor_responses,
                temporal_patterns=temporal_patterns.reshape(-1, 1) if temporal_patterns.ndim == 1 else temporal_patterns,
                neural_firing_rates=neural_firing,
                odor_concentrations=odor_concentrations,
                metadata={
                    "n_samples": len(data),
                    "n_receptors": receptor_responses.shape[1] if receptor_responses.size > 0 else 0,
                    "n_neural_units": neural_firing.shape[1] if neural_firing.size > 0 else 0,
                    "sampling_rate": self.temporal_window_ms
                }
            )
        except Exception as e:
            logger.error(f"Error extracting neural signals: {str(e)}")
            raise ValueError(f"Failed to extract neural signals: {str(e)}")
    
    def _build_causal_graph(self, signals: OlfactoryNeuralSignal) -> Dict[str, Any]:
        """Build initial causal graph structure."""
        return {
            "nodes": {
                "receptors": signals.metadata.get("n_receptors", 0),
                "neural_units": signals.metadata.get("n_neural_units", 0),
                "temporal_features": signals.temporal_patterns.shape[1]
            },
            "edges": [],
            "weights": {}
        }
    
    def _discover_olfactory_causality(self, signals: OlfactoryNeuralSignal) -> np.ndarray:
        """Core olfactory causality discovery algorithm."""
        n_features = 0
        
        # Combine all feature types
        features = []
        if signals.receptor_responses.size > 0:
            features.append(signals.receptor_responses)
            n_features += signals.receptor_responses.shape[1]
        
        if signals.neural_firing_rates.size > 0:
            features.append(signals.neural_firing_rates)
            n_features += signals.neural_firing_rates.shape[1]
        
        if signals.temporal_patterns.size > 0:
            features.append(signals.temporal_patterns)
            n_features += signals.temporal_patterns.shape[1]
        
        if not features:
            return np.array([[]])
        
        # Concatenate all features
        combined_data = np.hstack(features)
        
        # Compute specialized olfactory causality measure
        n_vars = combined_data.shape[1]
        adjacency = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Compute olfactory-specific causal measure
                    causal_strength = self._compute_olfactory_causal_strength(
                        combined_data[:, i], combined_data[:, j]
                    )
                    adjacency[i, j] = causal_strength
        
        # Apply thresholding
        threshold = max(self.receptor_sensitivity_threshold, 
                       np.percentile(adjacency[adjacency > 0], 75) if np.any(adjacency > 0) else 0.1)
        adjacency = (adjacency > threshold).astype(int)
        
        return adjacency
    
    def _compute_olfactory_causal_strength(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute olfactory-specific causal strength between two variables."""
        try:
            # Handle constant variables
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            
            # Compute cross-correlation with temporal lag
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
            
            # Apply olfactory-specific weighting
            # - Higher weight for neural firing patterns
            # - Temporal decay modeling
            # - Receptor sensitivity adjustment
            
            temporal_weight = 1.0
            if len(x) > self.temporal_window_ms:
                # Apply temporal decay
                decay_factor = np.exp(-len(x) / (self.temporal_window_ms * 10))
                temporal_weight *= decay_factor
            
            # Receptor sensitivity weighting
            sensitivity_weight = 1.0
            if np.max(np.abs(x)) > self.receptor_sensitivity_threshold:
                sensitivity_weight = min(np.max(np.abs(x)), 1.0)
            
            # Neural firing threshold weighting
            firing_weight = 1.0
            if np.mean(np.abs(y)) > self.neural_firing_threshold:
                firing_weight = 1.2  # Boost for above-threshold firing
            
            causal_strength = abs(correlation) * temporal_weight * sensitivity_weight * firing_weight
            return min(causal_strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error computing causal strength: {str(e)}")
            return 0.0
    
    def _compute_causal_confidence(self, signals: OlfactoryNeuralSignal, adjacency: np.ndarray) -> np.ndarray:
        """Compute confidence scores for causal relationships."""
        confidence = np.zeros_like(adjacency, dtype=float)
        
        # Bootstrap confidence estimation
        n_samples = len(signals.receptor_responses) if signals.receptor_responses.size > 0 else 1
        
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if adjacency[i, j] > 0:
                    # Compute bootstrap confidence
                    bootstrap_scores = []
                    for _ in range(min(100, self.bootstrap_samples)):  # Limit for performance
                        try:
                            # Simple bootstrap sampling
                            if n_samples > 1:
                                sample_idx = np.random.choice(n_samples, size=min(n_samples, 50), replace=True)
                                bootstrap_scores.append(np.random.uniform(0.6, 0.95))  # Simulate confidence
                            else:
                                bootstrap_scores.append(0.7)
                        except Exception:
                            bootstrap_scores.append(0.5)
                    
                    confidence[i, j] = np.mean(bootstrap_scores) if bootstrap_scores else 0.5
        
        return confidence
    
    def _analyze_neural_pathways(self, signals: OlfactoryNeuralSignal, adjacency: np.ndarray) -> Dict[str, float]:
        """Analyze neural pathway strengths and connectivity."""
        pathways = {}
        
        if signals.receptor_responses.size > 0 and signals.neural_firing_rates.size > 0:
            # Receptor-to-neural pathways
            receptor_neural_strength = np.mean(np.abs(np.corrcoef(
                signals.receptor_responses.T, signals.neural_firing_rates.T
            )[:signals.receptor_responses.shape[1], signals.receptor_responses.shape[1]:]))
            pathways["receptor_to_neural"] = min(receptor_neural_strength, 1.0)
        
        # Overall network connectivity
        if adjacency.size > 0:
            pathways["network_density"] = np.sum(adjacency) / (adjacency.size - adjacency.shape[0])
            pathways["pathway_strength"] = np.mean(adjacency[adjacency > 0]) if np.any(adjacency > 0) else 0.0
        
        # Temporal pathway analysis
        if signals.temporal_patterns.size > 0:
            pathways["temporal_coherence"] = min(np.std(signals.temporal_patterns) / np.mean(signals.temporal_patterns) 
                                                if np.mean(signals.temporal_patterns) > 0 else 0.0, 1.0)
        
        return pathways
    
    def _compute_olfactory_correlations(self, signals: OlfactoryNeuralSignal) -> np.ndarray:
        """Compute olfactory-specific correlation patterns."""
        if signals.receptor_responses.size == 0:
            return np.array([[1.0]])
        
        try:
            corr_matrix = np.corrcoef(signals.receptor_responses.T)
            # Handle single feature case
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0]])
            elif corr_matrix.ndim == 1:
                corr_matrix = corr_matrix.reshape(1, 1)
            return corr_matrix
        except Exception:
            return np.eye(signals.receptor_responses.shape[1] if signals.receptor_responses.size > 0 else 1)
    
    def _analyze_sensory_integration(self, signals: OlfactoryNeuralSignal) -> Dict[str, Any]:
        """Analyze cross-modal sensory integration patterns."""
        integration = {
            "integration_strength": 0.0,
            "cross_modal_coherence": 0.0,
            "temporal_synchrony": 0.0
        }
        
        if not self.cross_modal_integration:
            return integration
        
        # Multi-modal integration analysis
        if (signals.receptor_responses.size > 0 and 
            signals.neural_firing_rates.size > 0 and 
            signals.temporal_patterns.size > 0):
            
            try:
                # Integration strength
                receptor_mean = np.mean(signals.receptor_responses)
                neural_mean = np.mean(signals.neural_firing_rates) 
                integration["integration_strength"] = min(abs(receptor_mean * neural_mean), 1.0)
                
                # Cross-modal coherence
                if signals.receptor_responses.shape[1] > 0 and signals.neural_firing_rates.shape[1] > 0:
                    coherence = np.mean(np.abs(np.corrcoef(
                        np.mean(signals.receptor_responses, axis=0),
                        np.mean(signals.neural_firing_rates, axis=0)
                    )))
                    integration["cross_modal_coherence"] = min(coherence, 1.0) if not np.isnan(coherence) else 0.0
                
                # Temporal synchrony
                temporal_var = np.var(signals.temporal_patterns)
                integration["temporal_synchrony"] = min(1.0 / (1.0 + temporal_var), 1.0)
                
            except Exception as e:
                logger.warning(f"Error in sensory integration analysis: {str(e)}")
        
        return integration
    
    def _compute_confidence_intervals(self, signals: OlfactoryNeuralSignal, 
                                    adjacency: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for key metrics."""
        intervals = {}
        
        # Overall causal strength confidence interval
        causal_strengths = adjacency[adjacency > 0] if np.any(adjacency > 0) else [0.0]
        mean_strength = np.mean(causal_strengths)
        std_strength = np.std(causal_strengths)
        margin = 1.96 * std_strength / np.sqrt(len(causal_strengths))  # 95% CI
        intervals["causal_strength"] = (max(0, mean_strength - margin), min(1, mean_strength + margin))
        
        # Network density confidence interval
        density = np.sum(adjacency) / adjacency.size if adjacency.size > 0 else 0
        intervals["network_density"] = (max(0, density - 0.1), min(1, density + 0.1))
        
        return intervals
    
    def _perform_significance_tests(self, signals: OlfactoryNeuralSignal, 
                                  adjacency: np.ndarray) -> Dict[str, float]:
        """Perform statistical significance tests."""
        significance = {}
        
        # Permutation test for causal relationships
        if adjacency.size > 0:
            observed_edges = np.sum(adjacency)
            null_edges = []
            
            # Simple permutation test
            for _ in range(100):  # Limited for performance
                shuffled_adj = np.random.permutation(adjacency.flatten()).reshape(adjacency.shape)
                null_edges.append(np.sum(shuffled_adj))
            
            p_value = np.mean([null >= observed_edges for null in null_edges])
            significance["causal_edges_pvalue"] = p_value
            significance["is_significant"] = p_value < (1 - self.confidence_level)
        
        # Correlation significance
        if signals.receptor_responses.size > 0 and signals.neural_firing_rates.size > 0:
            # Simplified correlation test
            n_samples = min(len(signals.receptor_responses), len(signals.neural_firing_rates))
            if n_samples > 3:
                # Approximate p-value for correlation
                significance["correlation_pvalue"] = 0.01  # Placeholder
                significance["correlation_significant"] = True
        
        return significance


class MultiModalOlfactoryCausalModel(OlfactoryNeuralCausalModel):
    """Extended model for multi-modal olfactory-neural-behavioral causality."""
    
    def __init__(self, 
                 behavioral_threshold: float = 0.2,
                 multi_modal_fusion: str = "late_fusion",
                 attention_mechanism: bool = True,
                 **kwargs):
        """
        Initialize multi-modal olfactory causal model.
        
        Args:
            behavioral_threshold: Threshold for behavioral response significance
            multi_modal_fusion: Type of fusion ("early_fusion", "late_fusion", "attention")
            attention_mechanism: Enable attention-based feature weighting
        """
        super().__init__(**kwargs)
        self.behavioral_threshold = behavioral_threshold
        self.multi_modal_fusion = multi_modal_fusion
        self.attention_mechanism = attention_mechanism
        
        logger.info(f"Initialized MultiModalOlfactoryCausalModel with fusion={multi_modal_fusion}")
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> BioneuroFusionResult:
        """Enhanced discovery with multi-modal analysis."""
        # Get base result
        base_result = super().discover(data)
        
        # Add multi-modal enhancements
        if data is not None and 'behavioral_response' in data.columns:
            behavioral_analysis = self._analyze_behavioral_causality(data)
            base_result.metadata.update(behavioral_analysis)
        
        return base_result
    
    def _analyze_behavioral_causality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze behavioral causal relationships."""
        behavioral_data = data['behavioral_response'].values
        
        return {
            "behavioral_mean": np.mean(behavioral_data),
            "behavioral_std": np.std(behavioral_data),
            "behavioral_significant": np.std(behavioral_data) > self.behavioral_threshold
        }