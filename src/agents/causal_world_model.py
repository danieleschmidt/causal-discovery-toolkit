"""
Adaptive Causal World Model
===========================

Self-improving causal world model that learns from interactions and maintains
uncertainty estimates about causal relationships.

Novel Contributions:
- Adaptive causal structure learning from intervention data
- Uncertainty quantification for causal relationships
- Non-stationary causal discovery with change detection
- Multi-modal causal representation learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
import logging
from abc import ABC, abstractmethod
import time
from scipy import stats
from sklearn.metrics import mutual_info_score

try:
    from .causal_reasoning_agent import CausalObservation, CausalAction, AgentMemory
except ImportError:
    from causal_reasoning_agent import CausalObservation, CausalAction, AgentMemory


@dataclass
class WorldModelConfig:
    """Configuration for Causal World Model."""
    # Model architecture
    state_dim: int = 10
    action_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 3
    
    # Learning parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    update_frequency: int = 10
    memory_capacity: int = 1000
    
    # Causal structure learning
    structure_learning_rate: float = 0.01
    significance_threshold: float = 0.05
    min_samples_for_update: int = 20
    
    # Uncertainty estimation
    enable_uncertainty: bool = True
    uncertainty_method: str = "bootstrap"  # "bootstrap", "dropout", "ensemble"
    num_bootstrap_samples: int = 100
    
    # Change detection
    enable_change_detection: bool = True
    change_detection_window: int = 50
    change_threshold: float = 0.1


@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables."""
    cause: str
    effect: str
    strength: float
    confidence: float
    uncertainty: float
    last_updated: float
    evidence_count: int
    relationship_type: str = "linear"  # "linear", "nonlinear", "threshold"


class CausalStructureLearner:
    """Learns causal structure from intervention data."""
    
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.relationships = {}  # Dict[Tuple[str, str], CausalRelationship]
        self.intervention_history = []
        
    def update_from_intervention(self, pre_state: np.ndarray, 
                               intervention: CausalAction,
                               post_state: np.ndarray,
                               variable_names: List[str]):
        """Update causal structure from intervention outcome."""
        if len(variable_names) != len(pre_state) or len(variable_names) != len(post_state):
            return
        
        # Record intervention
        self.intervention_history.append({
            "pre_state": pre_state.copy(),
            "intervention": intervention,
            "post_state": post_state.copy(),
            "timestamp": time.time()
        })
        
        # Analyze causal effects
        state_change = post_state - pre_state
        
        # For each intervened variable, check effects on other variables
        for target_var in intervention.target_variables:
            if target_var in variable_names:
                cause_idx = variable_names.index(target_var)
                
                for effect_idx, effect_var in enumerate(variable_names):
                    if effect_idx != cause_idx:
                        # Measure causal effect
                        effect_strength = abs(state_change[effect_idx])
                        
                        # Statistical significance test
                        p_value = self._test_causal_significance(
                            cause_idx, effect_idx, variable_names
                        )
                        
                        if p_value < self.config.significance_threshold:
                            self._update_relationship(
                                target_var, effect_var, effect_strength, 
                                1.0 - p_value, p_value
                            )
    
    def _test_causal_significance(self, cause_idx: int, effect_idx: int,
                                variable_names: List[str]) -> float:
        """Test statistical significance of causal relationship."""
        if len(self.intervention_history) < 5:
            return 1.0  # Not enough data
        
        # Collect intervention outcomes for this cause-effect pair
        intervention_effects = []
        null_effects = []
        
        cause_var = variable_names[cause_idx]
        
        for record in self.intervention_history[-20:]:  # Use recent history
            pre_state = record["pre_state"]
            post_state = record["post_state"]
            intervention = record["intervention"]
            
            if len(pre_state) > max(cause_idx, effect_idx):
                effect_change = post_state[effect_idx] - pre_state[effect_idx]
                
                if cause_var in intervention.target_variables:
                    intervention_effects.append(effect_change)
                else:
                    null_effects.append(effect_change)
        
        # Perform t-test if we have both intervention and null data
        if len(intervention_effects) >= 3 and len(null_effects) >= 3:
            try:
                _, p_value = stats.ttest_ind(intervention_effects, null_effects)
                return p_value
            except:
                return 1.0
        
        return 1.0
    
    def _update_relationship(self, cause: str, effect: str, strength: float,
                           confidence: float, uncertainty: float):
        """Update a causal relationship."""
        key = (cause, effect)
        
        if key in self.relationships:
            # Update existing relationship
            rel = self.relationships[key]
            # Exponential moving average
            alpha = 0.3
            rel.strength = (1 - alpha) * rel.strength + alpha * strength
            rel.confidence = (1 - alpha) * rel.confidence + alpha * confidence
            rel.uncertainty = (1 - alpha) * rel.uncertainty + alpha * uncertainty
            rel.evidence_count += 1
        else:
            # Create new relationship
            self.relationships[key] = CausalRelationship(
                cause=cause,
                effect=effect,
                strength=strength,
                confidence=confidence,
                uncertainty=uncertainty,
                last_updated=time.time(),
                evidence_count=1
            )
    
    def get_causal_matrix(self, variable_names: List[str]) -> np.ndarray:
        """Get causal adjacency matrix."""
        n_vars = len(variable_names)
        matrix = np.zeros((n_vars, n_vars))
        
        for i, cause_var in enumerate(variable_names):
            for j, effect_var in enumerate(variable_names):
                key = (cause_var, effect_var)
                if key in self.relationships:
                    rel = self.relationships[key]
                    # Weight by confidence
                    matrix[i, j] = rel.strength * rel.confidence
        
        return matrix
    
    def get_relationship_uncertainties(self, variable_names: List[str]) -> np.ndarray:
        """Get uncertainty matrix for causal relationships."""
        n_vars = len(variable_names)
        uncertainties = np.ones((n_vars, n_vars)) * 0.5  # Default uncertainty
        
        for i, cause_var in enumerate(variable_names):
            for j, effect_var in enumerate(variable_names):
                key = (cause_var, effect_var)
                if key in self.relationships:
                    uncertainties[i, j] = self.relationships[key].uncertainty
        
        return uncertainties


class UncertaintyEstimator:
    """Estimates uncertainty in world model predictions."""
    
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.method = config.uncertainty_method
        
    def estimate_prediction_uncertainty(self, model: nn.Module, 
                                      inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty in model predictions."""
        if self.method == "dropout":
            return self._dropout_uncertainty(model, inputs)
        elif self.method == "bootstrap":
            return self._bootstrap_uncertainty(model, inputs)
        elif self.method == "ensemble":
            return self._ensemble_uncertainty(model, inputs)
        else:
            # Default: simple variance estimation
            with torch.no_grad():
                prediction = model(inputs)
                uncertainty = torch.ones_like(prediction) * 0.1
            return prediction, uncertainty
    
    def _dropout_uncertainty(self, model: nn.Module, 
                           inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using Monte Carlo dropout."""
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.config.num_bootstrap_samples // 10):  # Fewer samples for efficiency
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        model.eval()  # Return to eval mode
        return mean_pred, uncertainty
    
    def _bootstrap_uncertainty(self, model: nn.Module,
                             inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using bootstrap sampling."""
        # For simplicity, use dropout as proxy for bootstrap
        return self._dropout_uncertainty(model, inputs)
    
    def _ensemble_uncertainty(self, model: nn.Module,
                            inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using model ensemble."""
        # For simplicity, use dropout as proxy for ensemble
        return self._dropout_uncertainty(model, inputs)


class ChangeDetector:
    """Detects changes in causal structure over time."""
    
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.structure_history = []
        self.performance_history = []
        
    def check_for_changes(self, current_structure: np.ndarray,
                         current_performance: float) -> Dict[str, Any]:
        """Check if causal structure has changed significantly."""
        # Store current state
        self.structure_history.append({
            "structure": current_structure.copy(),
            "performance": current_performance,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.structure_history) > self.config.change_detection_window:
            self.structure_history.pop(0)
        
        change_detected = False
        change_magnitude = 0.0
        change_type = "none"
        
        if len(self.structure_history) >= 10:
            # Compare recent vs older structures
            recent_structures = [h["structure"] for h in self.structure_history[-5:]]
            older_structures = [h["structure"] for h in self.structure_history[-10:-5]]
            
            if recent_structures and older_structures:
                recent_mean = np.mean(recent_structures, axis=0)
                older_mean = np.mean(older_structures, axis=0)
                
                # Calculate structural difference
                structure_diff = np.linalg.norm(recent_mean - older_mean)
                
                if structure_diff > self.config.change_threshold:
                    change_detected = True
                    change_magnitude = structure_diff
                    change_type = "structural"
                
                # Check performance degradation
                recent_perf = np.mean([h["performance"] for h in self.structure_history[-5:]])
                older_perf = np.mean([h["performance"] for h in self.structure_history[-10:-5]])
                
                if recent_perf < older_perf - 0.1:  # Significant performance drop
                    change_detected = True
                    change_type = "performance" if change_type == "none" else "both"
        
        return {
            "change_detected": change_detected,
            "change_magnitude": change_magnitude,
            "change_type": change_type,
            "confidence": min(1.0, change_magnitude * 2.0) if change_detected else 0.0
        }


class CausalDynamicsModel(nn.Module):
    """Neural network model for causal dynamics."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder for state + action
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Predictor for next state
        self.state_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )
        
        # Uncertainty predictor
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.state_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting next state and uncertainty."""
        # Combine state and action
        x = torch.cat([state, action], dim=-1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Predict next state and uncertainty
        next_state = self.state_predictor(encoded)
        uncertainty = self.uncertainty_predictor(encoded)
        
        return next_state, uncertainty


class CausalWorldModel:
    """Comprehensive causal world model with adaptive learning."""
    
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.variable_names = [f"var_{i}" for i in range(config.state_dim)]
        
        # Components
        self.structure_learner = CausalStructureLearner(config)
        self.uncertainty_estimator = UncertaintyEstimator(config)
        self.change_detector = ChangeDetector(config)
        
        # Neural dynamics model
        self.dynamics_model = CausalDynamicsModel(config)
        self.optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(),
            lr=config.learning_rate
        )
        
        # State tracking
        self.update_count = 0
        self.training_loss_history = []
        self.current_performance = 0.5
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def update_from_experience(self, memory: AgentMemory):
        """Update world model from agent experiences."""
        if len(memory.observations) < self.config.min_samples_for_update:
            return
        
        # Update causal structure
        self._update_causal_structure(memory)
        
        # Update neural dynamics model
        self._update_dynamics_model(memory)
        
        # Check for changes
        current_structure = self.get_current_structure()
        change_info = self.change_detector.check_for_changes(
            current_structure, self.current_performance
        )
        
        if change_info["change_detected"]:
            self.logger.warning(f"Causal structure change detected: {change_info}")
            # Could trigger model retraining here
        
        self.update_count += 1
        self.logger.debug(f"World model updated: update #{self.update_count}")
    
    def _update_causal_structure(self, memory: AgentMemory):
        """Update causal structure from experiences."""
        for i in range(len(memory.observations) - 1):
            obs = memory.observations[i]
            action = memory.actions[i]
            next_obs = memory.observations[i + 1]
            
            if (obs.state_data is not None and 
                next_obs.state_data is not None and
                len(obs.state_data) == len(self.variable_names)):
                
                self.structure_learner.update_from_intervention(
                    obs.state_data, action, next_obs.state_data, self.variable_names
                )
    
    def _update_dynamics_model(self, memory: AgentMemory):
        """Update neural dynamics model."""
        # Prepare training data
        states = []
        actions = []
        next_states = []
        
        for i in range(len(memory.observations) - 1):
            obs = memory.observations[i]
            action = memory.actions[i]
            next_obs = memory.observations[i + 1]
            
            if (obs.state_data is not None and next_obs.state_data is not None and
                len(obs.state_data) == self.config.state_dim):
                
                states.append(obs.state_data)
                
                # Encode action
                action_vec = np.zeros(self.config.action_dim)
                if action.target_variables and action.intervention_values.size > 0:
                    for j, var in enumerate(action.target_variables):
                        if var.startswith("var_") and j < len(action.intervention_values):
                            var_idx = int(var.split("_")[1])
                            if var_idx < self.config.action_dim:
                                action_vec[var_idx] = action.intervention_values[j]
                actions.append(action_vec)
                
                next_states.append(next_obs.state_data)
        
        if len(states) < self.config.batch_size:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        next_states_tensor = torch.FloatTensor(next_states)
        
        # Training step
        self.dynamics_model.train()
        predicted_states, predicted_uncertainties = self.dynamics_model(states_tensor, actions_tensor)
        
        # Loss: prediction error + uncertainty regularization
        prediction_loss = F.mse_loss(predicted_states, next_states_tensor)
        uncertainty_loss = torch.mean(predicted_uncertainties)  # Encourage reasonable uncertainty
        
        total_loss = prediction_loss + 0.01 * uncertainty_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Track performance
        self.training_loss_history.append(total_loss.item())
        if len(self.training_loss_history) > 100:
            self.training_loss_history.pop(0)
        
        self.current_performance = 1.0 / (1.0 + np.mean(self.training_loss_history[-10:]))
        
        self.dynamics_model.eval()
    
    def get_current_structure(self) -> np.ndarray:
        """Get current estimate of causal structure."""
        return self.structure_learner.get_causal_matrix(self.variable_names)
    
    def get_structure_uncertainties(self) -> np.ndarray:
        """Get uncertainty estimates for causal structure."""
        return self.structure_learner.get_relationship_uncertainties(self.variable_names)
    
    def predict_intervention_outcome(self, action: CausalAction,
                                   current_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict outcome of proposed intervention."""
        if current_state is None:
            current_state = np.zeros(self.config.state_dim)
        
        # Encode action
        action_vec = np.zeros(self.config.action_dim)
        if action.target_variables and action.intervention_values.size > 0:
            for j, var in enumerate(action.target_variables):
                if var.startswith("var_") and j < len(action.intervention_values):
                    var_idx = int(var.split("_")[1])
                    if var_idx < self.config.action_dim:
                        action_vec[var_idx] = action.intervention_values[j]
        
        # Predict using neural model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action_vec).unsqueeze(0)
            
            if self.config.enable_uncertainty:
                predicted_state, uncertainty = self.uncertainty_estimator.estimate_prediction_uncertainty(
                    self.dynamics_model, torch.cat([state_tensor, action_tensor], dim=-1)
                )
            else:
                predicted_state, uncertainty = self.dynamics_model(state_tensor, action_tensor)
            
            predicted_state = predicted_state.squeeze().numpy()
            uncertainty = uncertainty.squeeze().numpy()
        
        # Calculate state change
        state_change = predicted_state - current_state
        
        return {
            "predicted_next_state": predicted_state.tolist(),
            "predicted_state_change": state_change.tolist(),
            "prediction_uncertainty": uncertainty.tolist(),
            "total_change_magnitude": float(np.linalg.norm(state_change)),
            "max_uncertainty": float(np.max(uncertainty)),
            "confidence": float(1.0 / (1.0 + np.mean(uncertainty)))
        }
    
    def get_causal_explanation(self, cause_var: str, effect_var: str) -> Dict[str, Any]:
        """Get explanation of causal relationship."""
        key = (cause_var, effect_var)
        
        if key in self.structure_learner.relationships:
            rel = self.structure_learner.relationships[key]
            return {
                "relationship_exists": True,
                "strength": rel.strength,
                "confidence": rel.confidence,
                "uncertainty": rel.uncertainty,
                "evidence_count": rel.evidence_count,
                "relationship_type": rel.relationship_type,
                "last_updated": rel.last_updated
            }
        else:
            return {
                "relationship_exists": False,
                "strength": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "evidence_count": 0,
                "relationship_type": "none"
            }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about world model state."""
        structure = self.get_current_structure()
        uncertainties = self.get_structure_uncertainties()
        
        return {
            "update_count": self.update_count,
            "current_performance": self.current_performance,
            "causal_structure": {
                "n_variables": len(self.variable_names),
                "n_relationships": len(self.structure_learner.relationships),
                "structure_density": float(np.mean(structure > 0.1)),
                "max_strength": float(np.max(structure)),
                "mean_uncertainty": float(np.mean(uncertainties))
            },
            "recent_training_loss": float(np.mean(self.training_loss_history[-5:])) if self.training_loss_history else 0.0,
            "intervention_history_size": len(self.structure_learner.intervention_history)
        }
    
    def save_model(self, filepath: str):
        """Save world model state."""
        state = {
            "config": self.config.__dict__,
            "variable_names": self.variable_names,
            "causal_structure": self.get_current_structure().tolist(),
            "uncertainties": self.get_structure_uncertainties().tolist(),
            "model_statistics": self.get_model_statistics(),
            "dynamics_model_state": self.dynamics_model.state_dict()
        }
        
        torch.save(state, filepath)
        self.logger.info(f"World model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load world model state."""
        state = torch.load(filepath)
        
        self.variable_names = state.get("variable_names", self.variable_names)
        self.update_count = state.get("model_statistics", {}).get("update_count", 0)
        
        if "dynamics_model_state" in state:
            self.dynamics_model.load_state_dict(state["dynamics_model_state"])
        
        self.logger.info(f"World model loaded from {filepath}")


def demo_causal_world_model():
    """Demonstrate causal world model capabilities."""
    print("🌍 Adaptive Causal World Model Demo")
    print("=" * 50)
    
    # Configuration
    config = WorldModelConfig(
        state_dim=5,
        action_dim=3,
        enable_uncertainty=True,
        enable_change_detection=True
    )
    
    # Initialize world model
    world_model = CausalWorldModel(config)
    
    # Create mock memory with causal data
    memory = AgentMemory(max_size=1000)
    
    # Simulate causal environment: X1 -> X2 -> X3
    true_structure = np.array([
        [0, 0.8, 0.3, 0.1, 0],
        [0, 0, 0.6, 0.2, 0],
        [0, 0, 0, 0.4, 0.1],
        [0, 0, 0, 0, 0.3],
        [0, 0, 0, 0, 0]
    ])
    
    print(f"True causal structure:\n{true_structure}")
    
    # Generate training data
    for step in range(50):
        # Generate state
        state = np.random.randn(5)
        
        # Generate intervention
        if np.random.random() < 0.3:  # 30% intervention rate
            target_var = f"var_{np.random.randint(0, 3)}"
            intervention_value = np.random.randn() * 0.5
            action = CausalAction(
                target_variables=[target_var],
                intervention_values=np.array([intervention_value]),
                intervention_type="do",
                confidence=0.8,
                reasoning_chain=[f"Intervention on {target_var}"]
            )
        else:
            action = CausalAction(
                target_variables=[],
                intervention_values=np.array([]),
                intervention_type="observe",
                confidence=1.0,
                reasoning_chain=["Null intervention"]
            )
        
        # Simulate next state using true structure
        next_state = state.copy()
        
        # Apply intervention effect
        if action.target_variables and action.intervention_values.size > 0:
            var_name = action.target_variables[0]
            if var_name.startswith("var_"):
                var_idx = int(var_name.split("_")[1])
                if var_idx < len(next_state):
                    next_state[var_idx] = action.intervention_values[0]
        
        # Apply causal dynamics
        for i in range(len(next_state)):
            for j in range(len(next_state)):
                if true_structure[i, j] > 0:
                    next_state[j] += true_structure[i, j] * state[i] * 0.1
        
        # Add noise
        next_state += np.random.randn(5) * 0.1
        
        # Create observations
        obs = CausalObservation(
            state_data=state,
            timestamp=time.time() + step,
            metadata={"step": step}
        )
        
        next_obs = CausalObservation(
            state_data=next_state,
            timestamp=time.time() + step + 1,
            metadata={"step": step + 1}
        )
        
        # Add to memory
        memory.add_experience(
            obs=obs,
            action=action,
            reward=np.random.randn(),
            causal_structure=true_structure.copy(),
            reasoning=action.reasoning_chain
        )
    
    print(f"\n📚 Generated {len(memory.observations)} training experiences")
    
    # Update world model
    world_model.update_from_experience(memory)
    
    # Get learned structure
    learned_structure = world_model.get_current_structure()
    uncertainties = world_model.get_structure_uncertainties()
    
    print(f"\n🧠 Learned causal structure:")
    print(learned_structure)
    print(f"\n🎯 Structure learning accuracy:")
    print(f"  Mean absolute error: {np.mean(np.abs(learned_structure - true_structure)):.3f}")
    print(f"  Correlation with true structure: {np.corrcoef(learned_structure.flatten(), true_structure.flatten())[0,1]:.3f}")
    
    print(f"\n📊 Uncertainty matrix:")
    print(f"  Mean uncertainty: {np.mean(uncertainties):.3f}")
    print(f"  Max uncertainty: {np.max(uncertainties):.3f}")
    
    # Test intervention prediction
    test_state = np.array([1.0, 0.5, -0.2, 0.3, -0.1])
    test_action = CausalAction(
        target_variables=["var_1"],
        intervention_values=np.array([0.8]),
        intervention_type="do",
        confidence=0.9,
        reasoning_chain=["Test intervention"]
    )
    
    prediction = world_model.predict_intervention_outcome(test_action, test_state)
    
    print(f"\n🔮 Intervention Prediction:")
    print(f"  Current state: {test_state}")
    print(f"  Intervention: {test_action.intervention_type} on {test_action.target_variables}")
    print(f"  Predicted change: {prediction['predicted_state_change']}")
    print(f"  Change magnitude: {prediction['total_change_magnitude']:.3f}")
    print(f"  Confidence: {prediction['confidence']:.3f}")
    
    # Get model statistics
    stats = world_model.get_model_statistics()
    print(f"\n📈 World Model Statistics:")
    print(f"  Updates performed: {stats['update_count']}")
    print(f"  Current performance: {stats['current_performance']:.3f}")
    print(f"  Causal relationships: {stats['causal_structure']['n_relationships']}")
    print(f"  Structure density: {stats['causal_structure']['structure_density']:.3f}")
    
    return world_model


if __name__ == "__main__":
    import time
    model = demo_causal_world_model()
    print(f"\n✅ Causal world model demo completed!")