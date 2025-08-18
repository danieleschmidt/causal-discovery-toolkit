"""
Self-Supervised Causal Representation Learning
============================================

Revolutionary self-supervised approach for learning causal representations without
explicit supervision, using contrastive learning and causal invariance principles.

Research Contributions:
- Contrastive causal representation learning
- Invariant causal mechanisms discovery
- Self-supervised structure learning
- Cross-domain causal transfer

Target Venues: ICML 2025, ICLR 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
import random

try:
    from .base import CausalDiscoveryModel, CausalResult
    from ..utils.validation import DataValidator
    from ..utils.metrics import CausalMetrics
except ImportError:
    from base import CausalDiscoveryModel, CausalResult
    from utils.validation import DataValidator
    from utils.metrics import CausalMetrics


@dataclass
class SelfSupervisedCausalConfig:
    """Configuration for Self-Supervised Causal Learning."""
    # Architecture
    encoder_dims: List[int] = None
    representation_dim: int = 256
    projection_dim: int = 128
    temperature: float = 0.07
    
    # Augmentation strategies
    noise_level: float = 0.1
    dropout_prob: float = 0.1
    masking_prob: float = 0.15
    intervention_prob: float = 0.2
    
    # Contrastive learning
    contrastive_weight: float = 1.0
    invariance_weight: float = 0.5
    reconstruction_weight: float = 0.3
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    warmup_epochs: int = 10
    
    # Causal structure
    sparsity_penalty: float = 0.1
    dag_penalty: float = 1.0
    structural_weight: float = 0.8


class CausalAugmentations:
    """Causal-aware data augmentations for self-supervised learning."""
    
    def __init__(self, config: SelfSupervisedCausalConfig):
        self.config = config
        
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise while preserving causal relationships."""
        noise = torch.randn_like(data) * self.config.noise_level
        return data + noise
    
    def random_masking(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random masking of variables for self-supervised learning."""
        batch_size, num_vars = data.shape
        mask = torch.rand(batch_size, num_vars) > self.config.masking_prob
        masked_data = data * mask.float()
        return masked_data, mask
    
    def simulate_intervention(self, data: torch.Tensor, 
                            variable_idx: Optional[int] = None) -> torch.Tensor:
        """Simulate interventional data for causal invariance learning."""
        if variable_idx is None:
            variable_idx = random.randint(0, data.size(1) - 1)
        
        intervened_data = data.clone()
        # Replace with random values from the same distribution
        random_values = torch.randn(data.size(0)) * data[:, variable_idx].std() + data[:, variable_idx].mean()
        intervened_data[:, variable_idx] = random_values
        
        return intervened_data
    
    def create_positive_pairs(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create positive pairs through causal-preserving augmentations."""
        # First augmentation: noise + dropout
        aug1 = self.add_noise(data)
        aug1 = F.dropout(aug1, p=self.config.dropout_prob, training=True)
        
        # Second augmentation: different noise + masking
        aug2 = self.add_noise(data)
        aug2, _ = self.random_masking(aug2)
        
        return aug1, aug2
    
    def create_negative_pairs(self, data: torch.Tensor) -> torch.Tensor:
        """Create negative pairs through causal-breaking interventions."""
        return self.simulate_intervention(data)


class CausalEncoder(nn.Module):
    """Encoder for learning causal representations."""
    
    def __init__(self, input_dim: int, config: SelfSupervisedCausalConfig):
        super().__init__()
        self.config = config
        
        # Encoder architecture
        encoder_dims = config.encoder_dims or [input_dim, 512, 256, config.representation_dim]
        
        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.BatchNorm1d(encoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*layers[:-1])  # Remove last dropout
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.representation_dim, config.representation_dim),
            nn.ReLU(),
            nn.Linear(config.representation_dim, config.projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning representations and projections."""
        representations = self.encoder(x)
        projections = self.projection_head(representations)
        return representations, projections


class CausalStructurePredictor(nn.Module):
    """Neural network for predicting causal structure."""
    
    def __init__(self, representation_dim: int, num_variables: int):
        super().__init__()
        self.num_variables = num_variables
        
        self.structure_predictor = nn.Sequential(
            nn.Linear(representation_dim, representation_dim // 2),
            nn.ReLU(),
            nn.Linear(representation_dim // 2, num_variables * num_variables),
            nn.Sigmoid()
        )
        
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Predict adjacency matrix from representations."""
        batch_size = representations.size(0)
        adjacency_logits = self.structure_predictor(representations)
        adjacency_matrix = adjacency_logits.view(batch_size, self.num_variables, self.num_variables)
        
        # Enforce DAG constraint (upper triangular)
        mask = torch.triu(torch.ones(self.num_variables, self.num_variables), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).to(adjacency_matrix.device)
        
        return adjacency_matrix * mask


class InvarianceLoss(nn.Module):
    """Loss function for learning invariant causal mechanisms."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, repr_original: torch.Tensor, 
                repr_intervened: torch.Tensor,
                intervention_targets: torch.Tensor) -> torch.Tensor:
        """Compute invariance loss between original and intervened representations."""
        # Only non-intervened variables should have similar representations
        non_intervened_mask = ~intervention_targets
        
        if non_intervened_mask.sum() == 0:
            return torch.tensor(0.0, device=repr_original.device)
        
        # Cosine similarity for non-intervened variables
        similarity = F.cosine_similarity(
            repr_original[non_intervened_mask], 
            repr_intervened[non_intervened_mask],
            dim=-1
        )
        
        # Encourage high similarity for non-intervened variables
        invariance_loss = -torch.mean(similarity)
        
        return invariance_loss


class SelfSupervisedCausalModel(CausalDiscoveryModel):
    """Self-supervised causal discovery through representation learning."""
    
    def __init__(self, 
                 config: Optional[SelfSupervisedCausalConfig] = None,
                 num_variables: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.config = config or SelfSupervisedCausalConfig()
        self.num_variables = num_variables
        
        # Will be initialized in fit()
        self.encoder = None
        self.structure_predictor = None
        self.augmentations = CausalAugmentations(self.config)
        self.invariance_loss_fn = InvarianceLoss(self.config.temperature)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def _initialize_model(self, input_dim: int):
        """Initialize model components after seeing data."""
        self.encoder = CausalEncoder(input_dim, self.config)
        self.structure_predictor = CausalStructurePredictor(
            self.config.representation_dim, 
            self.num_variables or input_dim
        )
        
    def contrastive_loss(self, projections1: torch.Tensor, 
                        projections2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for positive pairs."""
        # Normalize projections
        z1 = F.normalize(projections1, dim=-1)
        z2 = F.normalize(projections2, dim=-1)
        
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z1, z2.T) / self.config.temperature
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Contrastive loss
        loss_1to2 = F.cross_entropy(similarity_matrix, labels)
        loss_2to1 = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_1to2 + loss_2to1) / 2
    
    def reconstruction_loss(self, original_data: torch.Tensor,
                          representations: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss for masked variables."""
        # Simple linear decoder for reconstruction
        if not hasattr(self, 'decoder'):
            self.decoder = nn.Linear(
                self.config.representation_dim, 
                original_data.size(1)
            ).to(original_data.device)
        
        reconstructed = self.decoder(representations)
        
        # Only compute loss on masked variables
        masked_loss = F.mse_loss(
            reconstructed * (~mask).float(),
            original_data * (~mask).float(),
            reduction='mean'
        )
        
        return masked_loss
    
    def dag_constraint_loss(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """DAG constraint using matrix exponential trace."""
        batch_size = adjacency_matrix.size(0)
        dag_loss = 0
        
        for i in range(batch_size):
            adj = adjacency_matrix[i]
            # DAG constraint: tr(exp(A ⊙ A)) - d should be 0
            exp_adj = torch.matrix_exp(adj * adj)
            dag_loss += torch.trace(exp_adj) - adj.size(0)
            
        return dag_loss / batch_size
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            ground_truth: Optional[np.ndarray] = None,
            **kwargs) -> 'SelfSupervisedCausalModel':
        """Fit the self-supervised causal model."""
        try:
            # Validate and prepare data
            self.validator = DataValidator()
            validated_data = self.validator.validate(data)
            
            if validated_data is None:
                raise ValueError("Data validation failed")
            
            # Initialize model
            input_dim = validated_data.shape[1]
            self.num_variables = self.num_variables or input_dim
            self._initialize_model(input_dim)
            
            # Prepare data loader
            data_tensor = torch.FloatTensor(validated_data)
            dataset = TensorDataset(data_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            # Training setup
            all_params = list(self.encoder.parameters()) + list(self.structure_predictor.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
            
            # Training loop
            self.encoder.train()
            self.structure_predictor.train()
            training_losses = []
            
            for epoch in range(self.config.max_epochs):
                epoch_losses = {'total': 0, 'contrastive': 0, 'invariance': 0, 
                              'reconstruction': 0, 'structure': 0}
                num_batches = 0
                
                for (batch_data,) in dataloader:
                    optimizer.zero_grad()
                    
                    # Create augmented pairs
                    aug1, aug2 = self.augmentations.create_positive_pairs(batch_data)
                    intervened_data = self.augmentations.create_negative_pairs(batch_data)
                    masked_data, mask = self.augmentations.random_masking(batch_data)
                    
                    # Forward passes
                    repr1, proj1 = self.encoder(aug1)
                    repr2, proj2 = self.encoder(aug2)
                    repr_intervened, _ = self.encoder(intervened_data)
                    repr_masked, _ = self.encoder(masked_data)
                    
                    # Compute losses
                    # 1. Contrastive loss for positive pairs
                    contrastive_loss = self.contrastive_loss(proj1, proj2)
                    
                    # 2. Invariance loss for interventions
                    intervention_targets = torch.zeros(batch_data.size(1), dtype=torch.bool)
                    # Assume we know which variable was intervened (in practice, could be learned)
                    invariance_loss = self.invariance_loss_fn(
                        repr1, repr_intervened, intervention_targets
                    )
                    
                    # 3. Reconstruction loss for masked variables
                    reconstruction_loss = self.reconstruction_loss(batch_data, repr_masked, mask)
                    
                    # 4. Structure prediction loss
                    predicted_structure = self.structure_predictor(repr1)
                    sparsity_loss = torch.mean(torch.abs(predicted_structure))
                    dag_loss = self.dag_constraint_loss(predicted_structure)
                    structure_loss = self.config.sparsity_penalty * sparsity_loss + \
                                   self.config.dag_penalty * torch.abs(dag_loss)
                    
                    # Total loss
                    total_loss = (
                        self.config.contrastive_weight * contrastive_loss +
                        self.config.invariance_weight * invariance_loss +
                        self.config.reconstruction_weight * reconstruction_loss +
                        self.config.structural_weight * structure_loss
                    )
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    
                    # Track losses
                    epoch_losses['total'] += total_loss.item()
                    epoch_losses['contrastive'] += contrastive_loss.item()
                    epoch_losses['invariance'] += invariance_loss.item()
                    epoch_losses['reconstruction'] += reconstruction_loss.item()
                    epoch_losses['structure'] += structure_loss.item()
                    num_batches += 1
                
                # Average losses
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
                
                training_losses.append(epoch_losses)
                scheduler.step()
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Total={epoch_losses['total']:.4f}, "
                          f"Contrastive={epoch_losses['contrastive']:.4f}, "
                          f"Structure={epoch_losses['structure']:.4f}")
            
            self.is_trained = True
            self.training_history = training_losses
            
            return self
            
        except Exception as e:
            warnings.warn(f"Self-supervised causal model training failed: {e}")
            return self
    
    def discover(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> CausalResult:
        """Discover causal relationships using learned representations."""
        if not self.is_trained:
            warnings.warn("Model not trained. Fitting first...")
            self.fit(data)
        
        try:
            # Prepare data
            validated_data = self.validator.validate(data)
            data_tensor = torch.FloatTensor(validated_data)
            
            # Inference
            self.encoder.eval()
            self.structure_predictor.eval()
            
            with torch.no_grad():
                representations, _ = self.encoder(data_tensor)
                # Average representations across samples
                avg_representations = representations.mean(dim=0, keepdim=True)
                predicted_structure = self.structure_predictor(avg_representations)
                adjacency_matrix = predicted_structure.squeeze(0).numpy()
            
            # Apply threshold
            threshold = 0.3
            binary_adjacency = (adjacency_matrix > threshold).astype(int)
            
            # Create result
            metadata = {
                'method_used': 'Self-Supervised Causal Discovery',
                'model_config': self.config.__dict__,
                'final_loss': self.training_history[-1] if self.training_history else None,
                'num_epochs_trained': len(self.training_history),
                'threshold_used': threshold,
                'representation_dim': self.config.representation_dim,
                'self_supervised': True,
                'contrastive_learning': True
            }
            
            return CausalResult(
                adjacency_matrix=binary_adjacency,
                confidence_matrix=adjacency_matrix,
                metadata=metadata
            )
            
        except Exception as e:
            warnings.warn(f"Self-supervised causal discovery failed: {e}")
            # Return empty result
            n_vars = len(data.columns) if hasattr(data, 'columns') else data.shape[1]
            return CausalResult(
                adjacency_matrix=np.zeros((n_vars, n_vars)),
                confidence_matrix=np.zeros((n_vars, n_vars)),
                metadata={'method_used': 'Self-Supervised Causal Discovery', 'error': str(e)}
            )


# Export classes
__all__ = [
    'SelfSupervisedCausalModel',
    'SelfSupervisedCausalConfig',
    'CausalEncoder',
    'CausalStructurePredictor',
    'CausalAugmentations',
    'InvarianceLoss'
]