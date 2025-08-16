"""
Federated Causal Discovery with Privacy Preservation
===================================================

Novel privacy-preserving federated causal discovery across distributed datasets.
Implements differential privacy, secure aggregation, and adaptive client selection.

Research Contributions:
- Sample quality heterogeneity-aware federated learning
- Homomorphic encryption for secure causal structure learning  
- Adaptive client weighting based on data distribution
- Privacy-preserving causal discovery with formal guarantees

Target Venue: ICML 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import secrets
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

try:
    from .base import CausalDiscoveryModel, CausalResult
    from .information_theory import MutualInformationDiscovery
    from ..utils.validation import DataValidator
    from ..utils.metrics import CausalMetrics
    from ..utils.security import SecureComputing
except ImportError:
    from base import CausalDiscoveryModel, CausalResult
    from information_theory import MutualInformationDiscovery
    from utils.validation import DataValidator
    from utils.metrics import CausalMetrics
    from utils.security import SecureComputing


@dataclass
class FederatedConfig:
    """Configuration for Federated Causal Discovery."""
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5
    secure_aggregation: bool = True
    adaptive_client_selection: bool = True
    min_clients: int = 3
    max_clients: int = 10
    quality_threshold: float = 0.7
    communication_rounds: int = 10
    local_epochs: int = 5
    consensus_threshold: float = 0.8
    noise_multiplier: float = 1.0
    gradient_clipping: float = 1.0
    sample_rate: float = 0.1


class DifferentialPrivacyMechanism:
    """Differential privacy mechanisms for causal discovery."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0  # Global sensitivity for causal structures
        
    def laplace_mechanism(self, value: float) -> float:
        """Apply Laplace mechanism for ε-differential privacy."""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def gaussian_mechanism(self, value: float) -> float:
        """Apply Gaussian mechanism for (ε,δ)-differential privacy."""
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def private_aggregation(self, values: List[float]) -> float:
        """Aggregate values with differential privacy."""
        # Add noise to sum
        noisy_sum = sum(values) + self.gaussian_mechanism(0)
        return noisy_sum / len(values)
    
    def privatize_causal_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply differential privacy to causal adjacency matrix."""
        private_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                private_matrix[i, j] = self.laplace_mechanism(matrix[i, j])
        return np.clip(private_matrix, 0, 1)  # Ensure valid probabilities


class SecureAggregation:
    """Secure aggregation for federated causal discovery."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.private_keys = {}
        self.public_keys = {}
        self._generate_keys()
        
    def _generate_keys(self):
        """Generate public/private key pairs for each client."""
        for client_id in range(self.num_clients):
            private_key = secrets.randbelow(2**256)
            public_key = pow(2, private_key, 2**256)  # Simplified for demo
            self.private_keys[client_id] = private_key
            self.public_keys[client_id] = public_key
    
    def encrypt_value(self, value: float, client_id: int) -> int:
        """Encrypt a value using client's private key (simplified homomorphic encryption)."""
        # Simplified additive homomorphic encryption
        scaled_value = int(value * 1000)  # Scale for integer operations
        encrypted = (scaled_value + self.private_keys[client_id]) % (2**32)
        return encrypted
    
    def secure_sum(self, encrypted_values: List[int]) -> float:
        """Compute secure sum of encrypted values."""
        # Sum encrypted values
        encrypted_sum = sum(encrypted_values) % (2**32)
        
        # Decrypt by subtracting all private keys
        total_key_sum = sum(self.private_keys.values()) % (2**32)
        decrypted_sum = (encrypted_sum - total_key_sum) % (2**32)
        
        # Handle negative values and scale back
        if decrypted_sum > 2**31:
            decrypted_sum -= 2**32
        
        return decrypted_sum / 1000.0
    
    def aggregate_matrices(self, encrypted_matrices: List[np.ndarray]) -> np.ndarray:
        """Securely aggregate causal matrices."""
        if not encrypted_matrices:
            return np.array([])
        
        result_shape = encrypted_matrices[0].shape
        aggregated = np.zeros(result_shape)
        
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                encrypted_values = [matrix[i, j] for matrix in encrypted_matrices]
                aggregated[i, j] = self.secure_sum(encrypted_values)
        
        return aggregated / len(encrypted_matrices)


class DataQualityAssessment:
    """Assess data quality for adaptive client selection."""
    
    @staticmethod
    def compute_sample_size_score(data: np.ndarray) -> float:
        """Score based on sample size (more samples = higher score)."""
        n_samples = data.shape[0]
        # Logarithmic scaling for diminishing returns
        return min(1.0, np.log(n_samples + 1) / np.log(10000))
    
    @staticmethod
    def compute_feature_coverage_score(data: np.ndarray) -> float:
        """Score based on feature coverage (completeness)."""
        # Proportion of non-missing values
        if data.size == 0:
            return 0.0
        missing_rate = np.isnan(data).sum() / data.size
        return 1.0 - missing_rate
    
    @staticmethod
    def compute_data_diversity_score(data: np.ndarray) -> float:
        """Score based on data diversity (variance)."""
        try:
            # Compute coefficient of variation for each feature
            cv_scores = []
            for col in range(data.shape[1]):
                col_data = data[:, col]
                col_data = col_data[~np.isnan(col_data)]  # Remove NaN
                if len(col_data) > 1 and np.std(col_data) > 0:
                    cv = np.std(col_data) / (np.abs(np.mean(col_data)) + 1e-8)
                    cv_scores.append(min(cv, 2.0))  # Cap at 2.0
                else:
                    cv_scores.append(0.0)
            
            return np.mean(cv_scores) / 2.0  # Normalize to [0, 1]
        except:
            return 0.0
    
    @staticmethod
    def compute_statistical_power_score(data: np.ndarray) -> float:
        """Score based on statistical power for detecting relationships."""
        try:
            n_samples, n_features = data.shape
            # Simple heuristic: ability to detect medium effect sizes
            min_samples_needed = 50 * n_features  # Rule of thumb
            power_score = min(1.0, n_samples / min_samples_needed)
            return power_score
        except:
            return 0.0
    
    @classmethod
    def compute_overall_quality(cls, data: np.ndarray) -> float:
        """Compute overall data quality score."""
        scores = {
            'sample_size': cls.compute_sample_size_score(data),
            'feature_coverage': cls.compute_feature_coverage_score(data),
            'data_diversity': cls.compute_data_diversity_score(data),
            'statistical_power': cls.compute_statistical_power_score(data)
        }
        
        # Weighted average (can be tuned based on importance)
        weights = {'sample_size': 0.3, 'feature_coverage': 0.3, 
                  'data_diversity': 0.2, 'statistical_power': 0.2}
        
        overall_score = sum(weights[k] * scores[k] for k in scores.keys())
        return overall_score


class FederatedClient:
    """Federated client for causal discovery."""
    
    def __init__(self, 
                 client_id: int, 
                 data: np.ndarray,
                 local_model: CausalDiscoveryModel,
                 config: FederatedConfig):
        self.client_id = client_id
        self.data = data
        self.local_model = local_model
        self.config = config
        self.privacy_mechanism = DifferentialPrivacyMechanism(
            config.differential_privacy_epsilon, 
            config.differential_privacy_delta
        )
        self.quality_score = DataQualityAssessment.compute_overall_quality(data)
        
    def local_training(self, global_model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform local training and return privatized results."""
        try:
            # Apply global model parameters if provided
            if global_model_params:
                self._update_local_model(global_model_params)
            
            # Train local model
            result = self.local_model.discover_causal_structure(self.data)
            
            # Apply differential privacy to causal matrix
            private_matrix = self.privacy_mechanism.privatize_causal_matrix(
                result.causal_matrix.astype(float)
            )
            
            # Compute local statistics with privacy
            local_stats = {
                'sample_size': self.privacy_mechanism.laplace_mechanism(len(self.data)),
                'feature_means': [self.privacy_mechanism.gaussian_mechanism(np.mean(self.data[:, i])) 
                                for i in range(self.data.shape[1])],
                'quality_score': self.quality_score
            }
            
            return {
                'client_id': self.client_id,
                'causal_matrix': private_matrix,
                'confidence_scores': result.confidence_scores,
                'local_stats': local_stats,
                'quality_score': self.quality_score
            }
            
        except Exception as e:
            logging.error(f"Error in local training for client {self.client_id}: {e}")
            return None
    
    def _update_local_model(self, global_params: Dict):
        """Update local model with global parameters."""
        # Implementation depends on the specific model type
        # For now, we'll use a simple parameter averaging approach
        if hasattr(self.local_model, 'hyperparameters'):
            self.local_model.hyperparameters.update(global_params)


class FederatedCausalDiscovery(CausalDiscoveryModel):
    """
    Privacy-Preserving Federated Causal Discovery.
    
    Implements federated learning for causal discovery with differential privacy,
    secure aggregation, and adaptive client selection based on data quality.
    """
    
    def __init__(self, config: Optional[FederatedConfig] = None):
        super().__init__()
        self.config = config or FederatedConfig()
        self.clients: List[FederatedClient] = []
        self.global_model_params = {}
        self.secure_aggregator = None
        self.round_history = []
        
    def add_client(self, 
                   client_id: int, 
                   data: np.ndarray,
                   local_model: Optional[CausalDiscoveryModel] = None) -> None:
        """Add a federated client with their local data."""
        
        if local_model is None:
            # Use mutual information discovery as default local model
            local_model = MutualInformationDiscovery()
        
        client = FederatedClient(client_id, data, local_model, self.config)
        self.clients.append(client)
        
        logging.info(f"Added client {client_id} with quality score: {client.quality_score:.3f}")
    
    def _select_clients(self) -> List[FederatedClient]:
        """Adaptive client selection based on data quality."""
        
        if not self.config.adaptive_client_selection:
            # Random selection
            selected_count = min(self.config.max_clients, len(self.clients))
            return np.random.choice(self.clients, selected_count, replace=False).tolist()
        
        # Quality-based selection
        quality_scores = [client.quality_score for client in self.clients]
        
        # Select top quality clients
        top_indices = np.argsort(quality_scores)[::-1]
        selected_clients = []
        
        for idx in top_indices:
            if (len(selected_clients) < self.config.max_clients and 
                quality_scores[idx] >= self.config.quality_threshold):
                selected_clients.append(self.clients[idx])
        
        # Ensure minimum number of clients
        while (len(selected_clients) < self.config.min_clients and 
               len(selected_clients) < len(self.clients)):
            remaining_indices = [i for i in top_indices 
                               if self.clients[i] not in selected_clients]
            if remaining_indices:
                selected_clients.append(self.clients[remaining_indices[0]])
            else:
                break
        
        return selected_clients
    
    def _aggregate_results(self, client_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate client results with secure aggregation."""
        
        if not client_results:
            raise ValueError("No client results to aggregate")
        
        # Filter valid results
        valid_results = [r for r in client_results if r is not None]
        if not valid_results:
            raise ValueError("No valid client results")
        
        # Initialize secure aggregation if using secure mode
        if self.config.secure_aggregation and self.secure_aggregator is None:
            self.secure_aggregator = SecureAggregation(len(valid_results))
        
        # Extract matrices and quality scores
        matrices = [r['causal_matrix'] for r in valid_results]
        quality_scores = [r['quality_score'] for r in valid_results]
        
        if self.config.secure_aggregation:
            # Secure aggregation (simplified demonstration)
            encrypted_matrices = []
            for i, matrix in enumerate(matrices):
                encrypted_matrix = np.zeros_like(matrix)
                for row in range(matrix.shape[0]):
                    for col in range(matrix.shape[1]):
                        encrypted_matrix[row, col] = self.secure_aggregator.encrypt_value(
                            matrix[row, col], i
                        )
                encrypted_matrices.append(encrypted_matrix)
            
            # Aggregate securely
            aggregated_matrix = self.secure_aggregator.aggregate_matrices(encrypted_matrices)
        else:
            # Quality-weighted aggregation
            total_weight = sum(quality_scores)
            if total_weight == 0:
                weights = [1.0 / len(quality_scores)] * len(quality_scores)
            else:
                weights = [score / total_weight for score in quality_scores]
            
            # Weighted average of causal matrices
            aggregated_matrix = np.zeros_like(matrices[0])
            for matrix, weight in zip(matrices, weights):
                aggregated_matrix += weight * matrix
        
        # Aggregate confidence scores
        confidence_scores = [r['confidence_scores'] for r in valid_results]
        avg_confidence = np.mean(confidence_scores, axis=0)
        
        # Compute consensus metrics
        consensus_score = self._compute_consensus(matrices)
        
        return {
            'aggregated_matrix': aggregated_matrix,
            'confidence_scores': avg_confidence,
            'consensus_score': consensus_score,
            'participating_clients': [r['client_id'] for r in valid_results],
            'quality_distribution': quality_scores
        }
    
    def _compute_consensus(self, matrices: List[np.ndarray]) -> float:
        """Compute consensus score among client results."""
        if len(matrices) <= 1:
            return 1.0
        
        # Compute pairwise agreements
        agreements = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                # Binary agreement on edge presence
                binary_i = (matrices[i] > 0.5).astype(int)
                binary_j = (matrices[j] > 0.5).astype(int)
                agreement = np.mean(binary_i == binary_j)
                agreements.append(agreement)
        
        return np.mean(agreements)
    
    def fit(self, data: Optional[np.ndarray] = None, **kwargs) -> 'FederatedCausalDiscovery':
        """
        Fit federated causal discovery model.
        Note: Data should be added via add_client() method before calling fit().
        """
        
        if not self.clients:
            raise ValueError("No clients added. Use add_client() to add federated clients first.")
        
        logging.info(f"Starting federated causal discovery with {len(self.clients)} clients")
        
        for round_num in range(self.config.communication_rounds):
            logging.info(f"Communication round {round_num + 1}/{self.config.communication_rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients()
            logging.info(f"Selected {len(selected_clients)} clients")
            
            # Parallel local training
            client_results = []
            with ThreadPoolExecutor(max_workers=min(4, len(selected_clients))) as executor:
                future_to_client = {
                    executor.submit(client.local_training, self.global_model_params): client
                    for client in selected_clients
                }
                
                for future in as_completed(future_to_client):
                    result = future.result()
                    if result:
                        client_results.append(result)
            
            # Aggregate results
            if client_results:
                round_result = self._aggregate_results(client_results)
                self.round_history.append(round_result)
                
                # Update global model parameters (simplified)
                self.global_model_params = {
                    'round': round_num,
                    'consensus_score': round_result['consensus_score']
                }
                
                logging.info(f"Round {round_num + 1} consensus score: {round_result['consensus_score']:.3f}")
                
                # Early stopping if high consensus
                if round_result['consensus_score'] >= self.config.consensus_threshold:
                    logging.info("High consensus reached, stopping early")
                    break
            else:
                logging.warning(f"No valid results in round {round_num + 1}")
        
        self._fitted_data = data
        self.is_fitted = True
        return self
    
    def discover_causal_structure(self, data: Optional[np.ndarray] = None, **kwargs) -> CausalResult:
        """Discover federated causal structure."""
        
        if not self.is_fitted:
            self.fit(data, **kwargs)
        
        if not self.round_history:
            raise ValueError("No federated learning rounds completed")
        
        # Use results from the last round
        final_result = self.round_history[-1]
        
        # Apply final thresholding for binary adjacency matrix
        threshold = 0.5
        binary_matrix = (final_result['aggregated_matrix'] > threshold).astype(int)
        
        # Create comprehensive result
        result = CausalResult(
            causal_matrix=binary_matrix,
            confidence_scores=final_result['confidence_scores'],
            method_name="Federated Causal Discovery with Privacy Preservation",
            metadata={
                'federated_config': self.config.__dict__,
                'communication_rounds': len(self.round_history),
                'final_consensus_score': final_result['consensus_score'],
                'participating_clients': final_result['participating_clients'],
                'quality_distribution': final_result['quality_distribution'],
                'privacy_epsilon': self.config.differential_privacy_epsilon,
                'privacy_delta': self.config.differential_privacy_delta,
                'secure_aggregation_used': self.config.secure_aggregation,
                'continuous_matrix': final_result['aggregated_matrix'],
                'consensus_history': [r['consensus_score'] for r in self.round_history]
            }
        )
        
        return result
    
    def discover(self, data: Optional[pd.DataFrame] = None) -> CausalResult:
        """Discover causal relationships."""
        if data is None:
            if not hasattr(self, '_fitted_data'):
                raise ValueError("No data provided and model has no fitted data")
            data = self._fitted_data
        
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        return self.discover_causal_structure(data)
    
    def privacy_analysis(self) -> Dict[str, Any]:
        """Analyze privacy guarantees of the federated system."""
        
        total_epsilon = self.config.differential_privacy_epsilon * self.config.communication_rounds
        
        return {
            'differential_privacy': {
                'epsilon_per_round': self.config.differential_privacy_epsilon,
                'delta_per_round': self.config.differential_privacy_delta,
                'total_epsilon': total_epsilon,
                'total_delta': self.config.differential_privacy_delta,
                'privacy_level': 'High' if total_epsilon < 1.0 else 'Medium' if total_epsilon < 10.0 else 'Low'
            },
            'secure_aggregation': {
                'enabled': self.config.secure_aggregation,
                'encryption_type': 'Additive Homomorphic (Simplified)',
                'key_distribution': 'Per-client private keys'
            },
            'data_minimization': {
                'local_training': True,
                'raw_data_sharing': False,
                'gradient_sharing': False,
                'model_sharing': True
            }
        }


# Demonstration function
def demonstrate_federated_causal_discovery():
    """Demonstrate federated causal discovery with privacy preservation."""
    
    print("🔒 Federated Causal Discovery with Privacy - Research Demo")
    print("=" * 65)
    
    # Generate synthetic datasets for multiple clients
    np.random.seed(42)
    n_features = 4
    
    # True causal structure
    true_structure = np.array([
        [0, 1, 0, 1],  # X0 -> X1, X3
        [0, 0, 1, 0],  # X1 -> X2
        [0, 0, 0, 1],  # X2 -> X3
        [0, 0, 0, 0]   # X3 (sink)
    ])
    
    print(f"True Causal Structure:\n{true_structure}")
    
    # Create federated datasets (different institutions)
    federated_config = FederatedConfig(
        differential_privacy_epsilon=0.5,
        secure_aggregation=True,
        adaptive_client_selection=True,
        communication_rounds=5,
        min_clients=3,
        max_clients=5
    )
    
    fed_discovery = FederatedCausalDiscovery(federated_config)
    
    # Add clients with varying data quality
    client_configs = [
        {'n_samples': 1000, 'noise_level': 0.1, 'name': 'Hospital A (High Quality)'},
        {'n_samples': 500, 'noise_level': 0.2, 'name': 'Hospital B (Medium Quality)'},
        {'n_samples': 800, 'noise_level': 0.15, 'name': 'Research Center C (High Quality)'},
        {'n_samples': 300, 'noise_level': 0.3, 'name': 'Clinic D (Lower Quality)'},
        {'n_samples': 600, 'noise_level': 0.25, 'name': 'University E (Medium Quality)'}
    ]
    
    print(f"\n🏥 Creating {len(client_configs)} Federated Clients:")
    
    for i, config in enumerate(client_configs):
        # Generate client data following true causal structure
        n_samples = config['n_samples']
        noise_level = config['noise_level']
        
        data = np.random.randn(n_samples, n_features)
        data[:, 1] += 0.5 * data[:, 0] + noise_level * np.random.randn(n_samples)
        data[:, 2] += 0.7 * data[:, 1] + noise_level * np.random.randn(n_samples)
        data[:, 3] += 0.6 * data[:, 0] + 0.4 * data[:, 2] + noise_level * np.random.randn(n_samples)
        
        # Add some missing data to simulate real-world conditions
        missing_rate = np.random.uniform(0.01, 0.05)
        mask = np.random.random(data.shape) < missing_rate
        data[mask] = np.nan
        
        fed_discovery.add_client(i, data)
        print(f"  • Client {i}: {config['name']}")
    
    # Run federated causal discovery
    print(f"\n🔄 Running Federated Causal Discovery...")
    result = fed_discovery.discover_causal_structure()
    
    print(f"\n📊 Results:")
    print(f"Discovered Structure:\n{result.causal_matrix}")
    print(f"Final Consensus Score: {result.metadata['final_consensus_score']:.3f}")
    print(f"Communication Rounds: {result.metadata['communication_rounds']}")
    print(f"Participating Clients: {result.metadata['participating_clients']}")
    
    # Evaluate accuracy
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    true_flat = true_structure.flatten()
    pred_flat = result.causal_matrix.flatten()
    
    print(f"\n🎯 Performance Metrics:")
    print(f"Accuracy: {accuracy_score(true_flat, pred_flat):.3f}")
    print(f"Precision: {precision_score(true_flat, pred_flat, zero_division=0):.3f}")
    print(f"Recall: {recall_score(true_flat, pred_flat, zero_division=0):.3f}")
    print(f"F1-Score: {f1_score(true_flat, pred_flat, zero_division=0):.3f}")
    
    # Privacy analysis
    privacy_analysis = fed_discovery.privacy_analysis()
    print(f"\n🔒 Privacy Analysis:")
    print(f"Privacy Level: {privacy_analysis['differential_privacy']['privacy_level']}")
    print(f"Total ε: {privacy_analysis['differential_privacy']['total_epsilon']:.2f}")
    print(f"Secure Aggregation: {privacy_analysis['secure_aggregation']['enabled']}")
    print(f"Raw Data Sharing: {privacy_analysis['data_minimization']['raw_data_sharing']}")
    
    # Quality distribution
    quality_scores = result.metadata['quality_distribution']
    print(f"\n📈 Client Quality Distribution:")
    for i, score in enumerate(quality_scores):
        print(f"  • Client {result.metadata['participating_clients'][i]}: {score:.3f}")
    
    return fed_discovery, result


if __name__ == "__main__":
    demonstrate_federated_causal_discovery()