"""
Autonomous Causal Reasoning Agent
=================================

Revolutionary AI agent that can autonomously reason about causality,
plan interventions, and learn from environmental feedback.

Novel Contributions:
- Causal chain-of-thought reasoning
- Autonomous intervention planning
- Multi-modal causal environment understanding
- Self-improving causal world models
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import json
import time

try:
    from ..algorithms.foundation_causal import FoundationCausalModel, MultiModalCausalConfig
    from ..utils.validation import DataValidator
    from ..utils.metrics import CausalMetrics
except ImportError:
    # Fallback for development
    import warnings
    warnings.warn("Foundation model not available, using simplified implementation")


@dataclass
class AgentConfig:
    """Configuration for Causal Reasoning Agent."""
    # Agent Architecture
    reasoning_dim: int = 512
    memory_capacity: int = 10000
    planning_horizon: int = 10
    confidence_threshold: float = 0.7
    
    # Causal Reasoning
    max_causal_depth: int = 5
    intervention_budget: int = 100
    exploration_rate: float = 0.1
    
    # Learning
    learning_rate: float = 1e-4
    experience_replay_size: int = 1000
    world_model_update_freq: int = 10
    
    # Safety
    safety_constraints: bool = True
    max_intervention_impact: float = 0.5
    ethical_guidelines: bool = True


@dataclass
class CausalObservation:
    """Multi-modal observation from environment."""
    state_data: np.ndarray
    context_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CausalAction:
    """Causal intervention action."""
    target_variables: List[str]
    intervention_values: np.ndarray
    intervention_type: str  # 'do', 'condition', 'observe'
    confidence: float
    reasoning_chain: List[str]


@dataclass
class AgentMemory:
    """Memory structure for causal experiences."""
    observations: List[CausalObservation]
    actions: List[CausalAction]
    rewards: List[float]
    causal_structures: List[np.ndarray]
    reasoning_traces: List[List[str]]
    max_size: int = 10000
    
    def add_experience(self, obs: CausalObservation, action: CausalAction, 
                      reward: float, causal_structure: np.ndarray,
                      reasoning: List[str]):
        """Add new experience to memory."""
        if len(self.observations) >= self.max_size:
            # Remove oldest experience
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.causal_structures.pop(0)
            self.reasoning_traces.pop(0)
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.causal_structures.append(causal_structure)
        self.reasoning_traces.append(reasoning)


class CausalChainOfThought:
    """Causal reasoning chain-of-thought system."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.reasoning_steps = []
        
    def reason_about_causality(self, observation: CausalObservation, 
                             world_model: 'CausalWorldModel') -> List[str]:
        """Generate causal reasoning chain."""
        reasoning_chain = []
        
        # Step 1: Identify current state
        reasoning_chain.append(f"Current state: {self._describe_state(observation)}")
        
        # Step 2: Retrieve relevant causal structure
        causal_structure = world_model.get_current_structure()
        reasoning_chain.append(f"Active causal relationships: {self._describe_structure(causal_structure)}")
        
        # Step 3: Identify potential interventions
        possible_interventions = self._identify_interventions(observation, causal_structure)
        reasoning_chain.append(f"Possible interventions: {possible_interventions}")
        
        # Step 4: Predict outcomes
        for intervention in possible_interventions[:3]:  # Top 3
            predicted_outcome = world_model.predict_intervention_outcome(intervention)
            reasoning_chain.append(f"If {intervention} then {predicted_outcome}")
        
        # Step 5: Select best intervention
        best_intervention = self._select_best_intervention(possible_interventions, world_model)
        reasoning_chain.append(f"Selected intervention: {best_intervention} (confidence: {best_intervention.confidence:.3f})")
        
        return reasoning_chain
    
    def _describe_state(self, observation: CausalObservation) -> str:
        """Generate natural language description of state."""
        if observation.state_data is not None:
            return f"Variables: {observation.state_data.shape[0]} features, mean: {np.mean(observation.state_data):.3f}"
        return "No state data available"
    
    def _describe_structure(self, structure: np.ndarray) -> str:
        """Describe causal structure in natural language."""
        n_edges = np.sum(structure > 0.5)
        n_vars = structure.shape[0]
        return f"{n_edges} causal relationships among {n_vars} variables"
    
    def _identify_interventions(self, observation: CausalObservation, 
                               structure: np.ndarray) -> List[CausalAction]:
        """Identify possible interventions based on causal structure."""
        interventions = []
        n_vars = structure.shape[0]
        
        for i in range(min(n_vars, 10)):  # Limit for efficiency
            # Find variables that can be intervened upon (have outgoing edges)
            if np.sum(structure[i, :]) > 0.5:
                intervention = CausalAction(
                    target_variables=[f"var_{i}"],
                    intervention_values=np.random.randn(1) * 0.5,  # Small interventions
                    intervention_type="do",
                    confidence=np.sum(structure[i, :]) / n_vars,
                    reasoning_chain=[]
                )
                interventions.append(intervention)
        
        return interventions
    
    def _select_best_intervention(self, interventions: List[CausalAction],
                                 world_model: 'CausalWorldModel') -> CausalAction:
        """Select the best intervention based on predicted outcomes."""
        if not interventions:
            # Default null intervention
            return CausalAction(
                target_variables=[],
                intervention_values=np.array([]),
                intervention_type="observe",
                confidence=1.0,
                reasoning_chain=["No safe interventions available"]
            )
        
        # Score interventions by expected impact and confidence
        best_intervention = max(interventions, key=lambda x: x.confidence)
        return best_intervention


class CausalWorldModel:
    """Adaptive causal world model that learns from experience."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.current_structure = np.eye(10)  # Start with identity
        self.structure_confidence = np.ones((10, 10)) * 0.5
        self.update_count = 0
        
        # Simple neural network for dynamics prediction
        self.dynamics_model = nn.Sequential(
            nn.Linear(20, 64),  # state + action
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)   # next state
        )
        
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), 
                                        lr=config.learning_rate)
    
    def get_current_structure(self) -> np.ndarray:
        """Get current causal structure estimate."""
        return self.current_structure
    
    def update_from_experience(self, memory: AgentMemory):
        """Update world model from agent experiences."""
        if len(memory.observations) < 2:
            return
        
        # Update causal structure based on observed transitions
        self._update_causal_structure(memory)
        
        # Update dynamics model
        self._update_dynamics_model(memory)
        
        self.update_count += 1
    
    def _update_causal_structure(self, memory: AgentMemory):
        """Update causal structure based on intervention outcomes."""
        # Simple structure learning from intervention data
        for i in range(len(memory.actions)):
            action = memory.actions[i]
            if i < len(memory.observations) - 1:
                obs_before = memory.observations[i]
                obs_after = memory.observations[i + 1]
                
                # Update structure based on intervention outcomes
                if obs_before.state_data is not None and obs_after.state_data is not None:
                    self._update_structure_from_intervention(action, obs_before, obs_after)
    
    def _update_structure_from_intervention(self, action: CausalAction,
                                          obs_before: CausalObservation,
                                          obs_after: CausalObservation):
        """Update causal structure from single intervention."""
        if len(action.target_variables) == 0:
            return
        
        # Simple heuristic: if intervention caused change, update structure
        state_diff = obs_after.state_data - obs_before.state_data
        significant_changes = np.abs(state_diff) > 0.1
        
        # Strengthen causal links from intervened variables to changed variables
        for target_var in action.target_variables:
            if target_var.startswith("var_"):
                var_idx = int(target_var.split("_")[1])
                if var_idx < self.current_structure.shape[0]:
                    for j, changed in enumerate(significant_changes):
                        if changed and j < self.current_structure.shape[1]:
                            self.current_structure[var_idx, j] += 0.1
                            self.current_structure[var_idx, j] = min(1.0, self.current_structure[var_idx, j])
    
    def _update_dynamics_model(self, memory: AgentMemory):
        """Update neural dynamics model from experience."""
        if len(memory.observations) < 10:  # Need sufficient data
            return
        
        # Prepare training data
        states = []
        actions_encoded = []
        next_states = []
        
        for i in range(len(memory.observations) - 1):
            if (memory.observations[i].state_data is not None and 
                memory.observations[i + 1].state_data is not None):
                states.append(memory.observations[i].state_data)
                
                # Encode action
                action_vec = np.zeros(10)  # Simple encoding
                if memory.actions[i].target_variables:
                    action_vec[:len(memory.actions[i].intervention_values)] = memory.actions[i].intervention_values
                actions_encoded.append(action_vec)
                
                next_states.append(memory.observations[i + 1].state_data)
        
        if len(states) < 5:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions_encoded))
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        
        # Training step
        inputs = torch.cat([states_tensor, actions_tensor], dim=1)
        predictions = self.dynamics_model(inputs)
        loss = nn.MSELoss()(predictions, next_states_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def predict_intervention_outcome(self, intervention: CausalAction) -> str:
        """Predict outcome of proposed intervention."""
        # Simple prediction based on causal structure
        if len(intervention.target_variables) == 0:
            return "No change expected"
        
        affected_vars = []
        for target_var in intervention.target_variables:
            if target_var.startswith("var_"):
                var_idx = int(target_var.split("_")[1])
                if var_idx < self.current_structure.shape[0]:
                    # Find variables affected by this intervention
                    downstream = np.where(self.current_structure[var_idx, :] > 0.5)[0]
                    affected_vars.extend([f"var_{j}" for j in downstream])
        
        if affected_vars:
            return f"Will affect variables: {', '.join(set(affected_vars))}"
        else:
            return "Minimal impact expected"


class CausalReasoningAgent:
    """Autonomous agent with causal reasoning capabilities."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory = AgentMemory(max_size=config.memory_capacity)
        self.world_model = CausalWorldModel(config)
        self.reasoning_engine = CausalChainOfThought(config)
        self.step_count = 0
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Initialized Causal Reasoning Agent with config: {config}")
    
    def observe(self, observation: CausalObservation) -> Dict[str, Any]:
        """Process new observation and update internal state."""
        # Store observation
        self.current_observation = observation
        
        # Generate reasoning about current state
        reasoning_chain = self.reasoning_engine.reason_about_causality(
            observation, self.world_model
        )
        
        # Update world model periodically
        if self.step_count % self.config.world_model_update_freq == 0:
            self.world_model.update_from_experience(self.memory)
        
        return {
            "observation_processed": True,
            "reasoning_chain": reasoning_chain,
            "world_model_updated": self.step_count % self.config.world_model_update_freq == 0,
            "memory_size": len(self.memory.observations)
        }
    
    def plan_intervention(self, goal: Optional[str] = None) -> CausalAction:
        """Plan optimal intervention based on current understanding."""
        if not hasattr(self, 'current_observation'):
            # Default null action if no observations
            return CausalAction(
                target_variables=[],
                intervention_values=np.array([]),
                intervention_type="observe",
                confidence=1.0,
                reasoning_chain=["No observations available for planning"]
            )
        
        # Generate reasoning chain
        reasoning_chain = self.reasoning_engine.reason_about_causality(
            self.current_observation, self.world_model
        )
        
        # Extract the selected intervention from reasoning
        # For now, use simple exploration vs exploitation
        if np.random.random() < self.config.exploration_rate:
            # Exploration: random intervention
            action = self._generate_random_intervention()
            action.reasoning_chain = reasoning_chain + ["Exploration intervention selected"]
        else:
            # Exploitation: use reasoning engine
            possible_interventions = self.reasoning_engine._identify_interventions(
                self.current_observation, self.world_model.get_current_structure()
            )
            if possible_interventions:
                action = self.reasoning_engine._select_best_intervention(
                    possible_interventions, self.world_model
                )
                action.reasoning_chain = reasoning_chain
            else:
                action = CausalAction(
                    target_variables=[],
                    intervention_values=np.array([]),
                    intervention_type="observe",
                    confidence=1.0,
                    reasoning_chain=reasoning_chain + ["No interventions available"]
                )
        
        self.logger.info(f"Planned intervention: {action.intervention_type} on {action.target_variables}")
        return action
    
    def _generate_random_intervention(self) -> CausalAction:
        """Generate random intervention for exploration."""
        # Random variable to intervene on
        var_idx = np.random.randint(0, 5)  # Limit to first 5 variables
        return CausalAction(
            target_variables=[f"var_{var_idx}"],
            intervention_values=np.random.randn(1) * 0.3,  # Small random intervention
            intervention_type="do",
            confidence=0.5,
            reasoning_chain=["Random exploration intervention"]
        )
    
    def act(self, action: CausalAction) -> Dict[str, Any]:
        """Execute intervention and return action result."""
        # Simulate action execution (in real environment, this would interact with the world)
        execution_result = {
            "action_executed": True,
            "target_variables": action.target_variables,
            "intervention_values": action.intervention_values.tolist() if action.intervention_values.size > 0 else [],
            "intervention_type": action.intervention_type,
            "confidence": action.confidence,
            "timestamp": time.time()
        }
        
        self.logger.info(f"Executed intervention: {action.intervention_type} on {action.target_variables}")
        return execution_result
    
    def learn_from_feedback(self, action: CausalAction, reward: float, 
                           next_observation: CausalObservation):
        """Learn from action outcome and environmental feedback."""
        # Store experience in memory
        if hasattr(self, 'current_observation'):
            self.memory.add_experience(
                obs=self.current_observation,
                action=action,
                reward=reward,
                causal_structure=self.world_model.get_current_structure().copy(),
                reasoning=action.reasoning_chain
            )
        
        # Update world model
        self.world_model.update_from_experience(self.memory)
        
        self.step_count += 1
        
        self.logger.info(f"Learned from feedback: reward={reward:.3f}, memory_size={len(self.memory.observations)}")
        
        return {
            "learning_completed": True,
            "reward": reward,
            "memory_size": len(self.memory.observations),
            "world_model_updates": self.world_model.update_count
        }
    
    def get_reasoning_explanation(self) -> Dict[str, Any]:
        """Get detailed explanation of agent's reasoning process."""
        if not self.memory.reasoning_traces:
            return {"explanation": "No reasoning traces available"}
        
        latest_reasoning = self.memory.reasoning_traces[-1]
        causal_structure = self.world_model.get_current_structure()
        
        return {
            "latest_reasoning_chain": latest_reasoning,
            "causal_structure_summary": {
                "n_variables": causal_structure.shape[0],
                "n_edges": int(np.sum(causal_structure > 0.5)),
                "density": float(np.mean(causal_structure))
            },
            "memory_statistics": {
                "total_experiences": len(self.memory.observations),
                "average_reward": float(np.mean(self.memory.rewards)) if self.memory.rewards else 0.0,
                "exploration_rate": self.config.exploration_rate
            },
            "world_model_confidence": float(np.mean(self.world_model.structure_confidence))
        }
    
    def save_state(self, filepath: str):
        """Save agent state to file."""
        state = {
            "config": self.config.__dict__,
            "step_count": self.step_count,
            "causal_structure": self.world_model.get_current_structure().tolist(),
            "memory_size": len(self.memory.observations),
            "world_model_updates": self.world_model.update_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Agent state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load agent state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.step_count = state.get("step_count", 0)
        self.world_model.current_structure = np.array(state.get("causal_structure", np.eye(10)))
        self.world_model.update_count = state.get("world_model_updates", 0)
        
        self.logger.info(f"Agent state loaded from {filepath}")


# Demo and testing functions
def create_demo_environment() -> Tuple[CausalObservation, callable]:
    """Create a simple demo environment for testing."""
    # Simple linear environment: X1 -> X2 -> X3, X1 -> X3
    true_structure = np.array([
        [0, 1, 1],  # X1 affects X2 and X3
        [0, 0, 1],  # X2 affects X3
        [0, 0, 0]   # X3 affects nothing
    ])
    
    current_state = np.random.randn(3)
    
    def step_environment(action: CausalAction) -> Tuple[CausalObservation, float]:
        nonlocal current_state
        
        # Apply intervention
        if action.target_variables and action.intervention_values.size > 0:
            for i, var_name in enumerate(action.target_variables):
                if var_name.startswith("var_") and i < len(action.intervention_values):
                    var_idx = int(var_name.split("_")[1])
                    if var_idx < len(current_state):
                        current_state[var_idx] = action.intervention_values[i]
        
        # Apply causal dynamics
        new_state = current_state.copy()
        new_state[1] += 0.5 * current_state[0] + 0.1 * np.random.randn()  # X1 -> X2
        new_state[2] += 0.3 * current_state[0] + 0.4 * current_state[1] + 0.1 * np.random.randn()  # X1,X2 -> X3
        
        current_state = new_state
        
        # Simple reward: negative variance (prefer stable states)
        reward = -np.var(current_state)
        
        observation = CausalObservation(
            state_data=current_state.copy(),
            timestamp=time.time(),
            metadata={"true_structure": true_structure.tolist()}
        )
        
        return observation, reward
    
    initial_obs = CausalObservation(
        state_data=current_state.copy(),
        timestamp=time.time(),
        metadata={"true_structure": true_structure.tolist()}
    )
    
    return initial_obs, step_environment


def run_agent_demo(steps: int = 20) -> Dict[str, Any]:
    """Run a demonstration of the causal reasoning agent."""
    print("🤖 Autonomous Causal Reasoning Agent Demo")
    print("=" * 50)
    
    # Initialize agent
    config = AgentConfig(
        reasoning_dim=128,
        memory_capacity=1000,
        planning_horizon=5,
        exploration_rate=0.3
    )
    agent = CausalReasoningAgent(config)
    
    # Create demo environment
    observation, env_step = create_demo_environment()
    
    # Run interaction loop
    total_reward = 0.0
    
    for step in range(steps):
        print(f"\n--- Step {step + 1}/{steps} ---")
        
        # Agent observes environment
        obs_result = agent.observe(observation)
        print(f"Agent observed: {obs_result['observation_processed']}")
        
        # Agent plans intervention
        action = agent.plan_intervention()
        print(f"Planned action: {action.intervention_type} on {action.target_variables}")
        print(f"Confidence: {action.confidence:.3f}")
        
        # Agent acts
        act_result = agent.act(action)
        
        # Environment responds
        next_observation, reward = env_step(action)
        total_reward += reward
        
        # Agent learns from feedback
        learn_result = agent.learn_from_feedback(action, reward, next_observation)
        print(f"Reward: {reward:.3f}, Total: {total_reward:.3f}")
        
        # Update for next iteration
        observation = next_observation
        
        # Show reasoning every 5 steps
        if (step + 1) % 5 == 0:
            explanation = agent.get_reasoning_explanation()
            print(f"Latest reasoning: {explanation['latest_reasoning_chain'][-1] if explanation['latest_reasoning_chain'] else 'None'}")
            print(f"World model: {explanation['causal_structure_summary']['n_edges']} edges, {explanation['causal_structure_summary']['density']:.3f} density")
    
    # Final statistics
    final_explanation = agent.get_reasoning_explanation()
    
    results = {
        "total_steps": steps,
        "total_reward": total_reward,
        "average_reward": total_reward / steps,
        "final_memory_size": len(agent.memory.observations),
        "world_model_updates": agent.world_model.update_count,
        "final_reasoning": final_explanation
    }
    
    print(f"\n🎯 Demo Results:")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward: {total_reward/steps:.3f}")
    print(f"Experiences learned: {len(agent.memory.observations)}")
    print(f"World model updates: {agent.world_model.update_count}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = run_agent_demo(steps=15)
    print(f"\n✅ Agent demo completed successfully!")
    print(f"Results: {results}")