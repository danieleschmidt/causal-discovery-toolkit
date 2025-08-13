"""Comprehensive Causal Discovery Pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

try:
    from .algorithms.base import CausalDiscoveryModel, CausalResult
    from .algorithms.information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
    from .algorithms.bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
    from .algorithms.robust import RobustSimpleLinearCausalModel
    from .algorithms.optimized import OptimizedCausalModel
    from .utils.data_processing import DataProcessor
    from .utils.metrics import CausalMetrics
    from .utils.validation import DataValidator
except ImportError:
    from algorithms.base import CausalDiscoveryModel, CausalResult
    from algorithms.information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
    from algorithms.bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
    from algorithms.robust import RobustSimpleLinearCausalModel
    from algorithms.optimized import OptimizedCausalModel
    from utils.data_processing import DataProcessor
    from utils.metrics import CausalMetrics
    from utils.validation import DataValidator


@dataclass
class PipelineConfig:
    """Configuration for causal discovery pipeline."""
    algorithms: List[str] = None
    preprocessing_steps: List[str] = None
    validation_enabled: bool = True
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: float = 300.0
    cross_validation_folds: int = 5
    bootstrap_samples: int = 100
    confidence_threshold: float = 0.95
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = [
                'simple_linear',
                'mutual_information', 
                'bayesian_network',
                'constraint_based'
            ]
        if self.preprocessing_steps is None:
            self.preprocessing_steps = [
                'clean',
                'standardize',
                'validate'
            ]


@dataclass
class PipelineResult:
    """Results from causal discovery pipeline."""
    best_result: CausalResult
    all_results: Dict[str, CausalResult]
    ensemble_result: CausalResult
    performance_metrics: Dict[str, Dict[str, float]]
    execution_times: Dict[str, float]
    validation_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]


class CausalDiscoveryPipeline:
    """Comprehensive causal discovery pipeline with multiple algorithms and validation."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.data_processor = DataProcessor()
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        
        # Initialize algorithms
        self.algorithms = self._initialize_algorithms()
        
    def _initialize_algorithms(self) -> Dict[str, CausalDiscoveryModel]:
        """Initialize causal discovery algorithms."""
        algorithms = {}
        
        if 'simple_linear' in self.config.algorithms:
            algorithms['simple_linear'] = RobustSimpleLinearCausalModel(threshold=0.3)
            
        if 'mutual_information' in self.config.algorithms:
            algorithms['mutual_information'] = MutualInformationDiscovery(
                threshold=0.1, 
                n_bins=10,
                use_conditional_mi=True
            )
            
        if 'transfer_entropy' in self.config.algorithms:
            algorithms['transfer_entropy'] = TransferEntropyDiscovery(
                threshold=0.05,
                lag=1
            )
            
        if 'bayesian_network' in self.config.algorithms:
            algorithms['bayesian_network'] = BayesianNetworkDiscovery(
                scoring_method='bic',
                prior_knowledge=None
            )
            
        if 'constraint_based' in self.config.algorithms:
            algorithms['constraint_based'] = ConstraintBasedDiscovery(
                independence_test='chi_square',
                alpha=0.05
            )
            
        if 'optimized' in self.config.algorithms:
            algorithms['optimized'] = OptimizedCausalModel(
                threshold=0.3,
                use_gpu=False,
                batch_size=1000
            )
            
        return algorithms
    
    def run(self, data: pd.DataFrame, 
            ground_truth: Optional[np.ndarray] = None) -> PipelineResult:
        """Run the complete causal discovery pipeline.
        
        Args:
            data: Input data for causal discovery
            ground_truth: Optional ground truth adjacency matrix for evaluation
            
        Returns:
            PipelineResult with comprehensive results
        """
        start_time = time.time()
        
        # Step 1: Data preprocessing
        self.logger.info("Starting data preprocessing...")
        processed_data = self._preprocess_data(data)
        
        # Step 2: Run all algorithms
        self.logger.info(f"Running {len(self.algorithms)} causal discovery algorithms...")
        all_results, execution_times = self._run_algorithms(processed_data)
        
        # Step 3: Evaluate results if ground truth is available
        performance_metrics = {}
        if ground_truth is not None:
            self.logger.info("Evaluating algorithm performance...")
            performance_metrics = self._evaluate_results(all_results, ground_truth)
        
        # Step 4: Create ensemble result
        self.logger.info("Creating ensemble result...")
        ensemble_result = self._create_ensemble(all_results, performance_metrics)
        
        # Step 5: Cross-validation
        validation_scores = {}
        if self.config.validation_enabled:
            self.logger.info("Performing cross-validation...")
            validation_scores = self._cross_validate(processed_data)
        
        # Step 6: Bootstrap confidence intervals
        confidence_intervals = {}
        if self.config.bootstrap_samples > 0:
            self.logger.info("Computing bootstrap confidence intervals...")
            confidence_intervals = self._bootstrap_confidence(processed_data)
        
        # Step 7: Select best result
        best_result = self._select_best_result(all_results, performance_metrics, validation_scores)
        
        total_time = time.time() - start_time
        
        return PipelineResult(
            best_result=best_result,
            all_results=all_results,
            ensemble_result=ensemble_result,
            performance_metrics=performance_metrics,
            execution_times=execution_times,
            validation_scores=validation_scores,
            confidence_intervals=confidence_intervals,
            metadata={
                'total_execution_time': total_time,
                'data_shape': data.shape,
                'algorithms_used': list(self.algorithms.keys()),
                'config': self.config
            }
        )
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data according to configuration."""
        processed_data = data.copy()
        
        for step in self.config.preprocessing_steps:
            if step == 'clean':
                processed_data = self.data_processor.clean_data(processed_data)
            elif step == 'standardize':
                processed_data = self.data_processor.standardize(processed_data)
            elif step == 'validate':
                validation_results = self.validator.validate_dataset(processed_data)
                if not validation_results['is_valid']:
                    warnings.warn(f"Data validation issues: {validation_results['issues']}")
                    
        return processed_data
    
    def _run_algorithms(self, data: pd.DataFrame) -> Tuple[Dict[str, CausalResult], Dict[str, float]]:
        """Run all configured algorithms on the data."""
        all_results = {}
        execution_times = {}
        
        if self.config.parallel_execution:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_name = {
                    executor.submit(self._run_single_algorithm, name, model, data): name
                    for name, model in self.algorithms.items()
                }
                
                for future in as_completed(future_to_name, timeout=self.config.timeout_seconds):
                    name = future_to_name[future]
                    try:
                        result, exec_time = future.result()
                        all_results[name] = result
                        execution_times[name] = exec_time
                        self.logger.info(f"Completed {name} in {exec_time:.3f}s")
                    except Exception as e:
                        self.logger.error(f"Algorithm {name} failed: {e}")
                        execution_times[name] = float('inf')
        else:
            # Sequential execution
            for name, model in self.algorithms.items():
                try:
                    result, exec_time = self._run_single_algorithm(name, model, data)
                    all_results[name] = result
                    execution_times[name] = exec_time
                    self.logger.info(f"Completed {name} in {exec_time:.3f}s")
                except Exception as e:
                    self.logger.error(f"Algorithm {name} failed: {e}")
                    execution_times[name] = float('inf')
                    
        return all_results, execution_times
    
    def _run_single_algorithm(self, name: str, model: CausalDiscoveryModel, 
                             data: pd.DataFrame) -> Tuple[CausalResult, float]:
        """Run a single causal discovery algorithm."""
        start_time = time.time()
        
        try:
            result = model.fit_discover(data)
            exec_time = time.time() - start_time
            return result, exec_time
        except Exception as e:
            self.logger.error(f"Error in {name}: {e}")
            raise
    
    def _evaluate_results(self, results: Dict[str, CausalResult], 
                         ground_truth: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate algorithm results against ground truth."""
        performance_metrics = {}
        
        for name, result in results.items():
            try:
                metrics = CausalMetrics.evaluate_discovery(
                    ground_truth,
                    result.adjacency_matrix,
                    result.confidence_scores
                )
                performance_metrics[name] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
                performance_metrics[name] = {'f1_score': 0.0}
                
        return performance_metrics
    
    def _create_ensemble(self, results: Dict[str, CausalResult], 
                        performance_metrics: Dict[str, Dict[str, float]]) -> CausalResult:
        """Create ensemble result by combining individual algorithm results."""
        if not results:
            raise ValueError("No valid results to ensemble")
        
        # Get dimensions from first result
        first_result = next(iter(results.values()))
        n_vars = first_result.adjacency_matrix.shape[0]
        
        # Initialize ensemble matrices
        ensemble_adjacency = np.zeros((n_vars, n_vars))
        ensemble_confidence = np.zeros((n_vars, n_vars))
        
        # Weight algorithms by performance if available
        if performance_metrics:
            weights = {name: metrics.get('f1_score', 0.0) 
                      for name, metrics in performance_metrics.items()}
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w/total_weight for name, w in weights.items()}
            else:
                weights = {name: 1.0/len(results) for name in results.keys()}
        else:
            weights = {name: 1.0/len(results) for name in results.keys()}
        
        # Combine results
        for name, result in results.items():
            weight = weights.get(name, 0.0)
            ensemble_adjacency += weight * result.adjacency_matrix
            ensemble_confidence += weight * result.confidence_scores
        
        # Threshold ensemble adjacency matrix
        threshold = self.config.confidence_threshold
        final_adjacency = (ensemble_adjacency >= threshold).astype(int)
        
        return CausalResult(
            adjacency_matrix=final_adjacency,
            confidence_scores=ensemble_confidence,
            method_used="Ensemble",
            metadata={
                'component_algorithms': list(results.keys()),
                'weights': weights,
                'threshold': threshold,
                'n_variables': n_vars,
                'n_edges': np.sum(final_adjacency),
                'variable_names': first_result.metadata.get('variable_names', [])
            }
        )
    
    def _cross_validate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Perform cross-validation on algorithms."""
        n_folds = self.config.cross_validation_folds
        n_samples = len(data)
        fold_size = n_samples // n_folds
        
        cv_scores = {name: [] for name in self.algorithms.keys()}
        
        for fold in range(n_folds):
            # Create train/test split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n_samples) if i not in test_indices]
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Run algorithms on training data
            for name, model in self.algorithms.items():
                try:
                    # Fit on training data
                    model.fit(train_data)
                    
                    # Predict on test data
                    result = model.discover(test_data)
                    
                    # Compute stability score (example metric)
                    stability_score = self._compute_stability_score(result)
                    cv_scores[name].append(stability_score)
                    
                except Exception as e:
                    self.logger.warning(f"CV fold {fold} failed for {name}: {e}")
                    cv_scores[name].append(0.0)
        
        # Average scores
        return {name: np.mean(scores) for name, scores in cv_scores.items()}
    
    def _bootstrap_confidence(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for edge probabilities."""
        n_samples = len(data)
        n_bootstrap = self.config.bootstrap_samples
        
        bootstrap_results = {name: [] for name in self.algorithms.keys()}
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Run algorithms
            for name, model in self.algorithms.items():
                try:
                    result = model.fit_discover(bootstrap_data)
                    bootstrap_results[name].append(result.adjacency_matrix)
                except Exception:
                    continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_threshold
        
        for name, results_list in bootstrap_results.items():
            if results_list:
                # Stack all bootstrap results
                stacked = np.stack(results_list, axis=0)
                
                # Compute edge probabilities
                edge_probs = np.mean(stacked, axis=0)
                
                # Compute confidence intervals
                lower = np.percentile(edge_probs, 100 * alpha/2)
                upper = np.percentile(edge_probs, 100 * (1 - alpha/2))
                
                confidence_intervals[name] = (lower, upper)
        
        return confidence_intervals
    
    def _select_best_result(self, results: Dict[str, CausalResult],
                          performance_metrics: Dict[str, Dict[str, float]],
                          validation_scores: Dict[str, float]) -> CausalResult:
        """Select the best algorithm result based on multiple criteria."""
        if not results:
            raise ValueError("No valid results to select from")
        
        # If we have ground truth performance metrics, use F1 score
        if performance_metrics:
            best_name = max(performance_metrics.keys(), 
                          key=lambda x: performance_metrics[x].get('f1_score', 0.0))
            return results[best_name]
        
        # If we have validation scores, use those
        if validation_scores:
            best_name = max(validation_scores.keys(), key=lambda x: validation_scores[x])
            return results[best_name]
        
        # Default to first available result
        return next(iter(results.values()))
    
    def _compute_stability_score(self, result: CausalResult) -> float:
        """Compute a stability score for a causal discovery result."""
        # Simple stability metric based on edge density and confidence
        adj_matrix = result.adjacency_matrix
        conf_matrix = result.confidence_scores
        
        n_edges = np.sum(adj_matrix)
        if n_edges == 0:
            return 0.0
        
        # Average confidence of detected edges
        edge_mask = adj_matrix == 1
        avg_confidence = np.mean(conf_matrix[edge_mask]) if np.any(edge_mask) else 0.0
        
        # Penalize too many or too few edges
        n_possible = adj_matrix.size - np.trace(np.ones_like(adj_matrix))
        edge_density = n_edges / n_possible
        density_penalty = min(edge_density, 1 - edge_density)
        
        return avg_confidence * density_penalty