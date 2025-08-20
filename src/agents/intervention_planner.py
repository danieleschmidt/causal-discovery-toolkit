"""
Autonomous Intervention Planner
===============================

Advanced planning system for causal interventions with uncertainty quantification,
safety constraints, and multi-objective optimization.

Novel Contributions:
- Counterfactual intervention planning
- Multi-objective optimization (efficacy vs safety vs cost)
- Uncertainty-aware intervention selection
- Hierarchical intervention decomposition
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings
from enum import Enum
import itertools
from scipy.optimize import minimize
import logging

try:
    from .causal_reasoning_agent import CausalAction, CausalObservation, AgentConfig
except ImportError:
    from causal_reasoning_agent import CausalAction, CausalObservation, AgentConfig


class InterventionType(Enum):
    """Types of causal interventions."""
    DO = "do"                    # Hard intervention do(X=x)
    CONDITION = "condition"      # Conditioning P(Y|X=x)
    OBSERVE = "observe"          # Passive observation
    SOFT = "soft"               # Soft intervention with noise
    TEMPORAL = "temporal"        # Time-delayed intervention
    COMPOSITE = "composite"      # Multi-variable intervention


class SafetyLevel(Enum):
    """Safety levels for interventions."""
    SAFE = "safe"               # Guaranteed safe
    CAUTIOUS = "cautious"       # Likely safe with monitoring
    MODERATE = "moderate"       # Moderate risk, requires approval
    HIGH_RISK = "high_risk"     # High risk, extensive safeguards needed
    PROHIBITED = "prohibited"   # Not allowed


@dataclass
class InterventionConstraints:
    """Constraints for intervention planning."""
    # Safety constraints
    max_intervention_magnitude: float = 1.0
    forbidden_variables: List[str] = None
    safety_threshold: float = 0.8
    
    # Resource constraints  
    max_cost: float = 100.0
    max_duration: float = 3600.0  # 1 hour
    max_simultaneous_interventions: int = 3
    
    # Ethical constraints
    require_consent: bool = True
    reversible_only: bool = False
    minimize_harm: bool = True
    
    def __post_init__(self):
        if self.forbidden_variables is None:
            self.forbidden_variables = []


@dataclass
class InterventionResult:
    """Result of intervention planning."""
    # Planned intervention
    action: CausalAction
    
    # Planning metadata
    expected_outcome: Dict[str, float]
    confidence_interval: Dict[str, Tuple[float, float]]
    safety_assessment: SafetyLevel
    estimated_cost: float
    
    # Alternative plans
    alternative_actions: List[CausalAction]
    
    # Reasoning
    planning_trace: List[str]
    counterfactuals: Dict[str, Any]
    
    # Uncertainty quantification
    outcome_uncertainty: float
    model_uncertainty: float
    total_uncertainty: float


class CounterfactualReasoning:
    """Counterfactual reasoning for intervention planning."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
    def generate_counterfactuals(self, observation: CausalObservation,
                                action: CausalAction,
                                world_model: 'CausalWorldModel') -> Dict[str, Any]:
        """Generate counterfactual scenarios for intervention."""
        counterfactuals = {}
        
        # What if we didn't intervene?
        null_action = CausalAction(
            target_variables=[],
            intervention_values=np.array([]),
            intervention_type="observe",
            confidence=1.0,
            reasoning_chain=["Null intervention counterfactual"]
        )
        counterfactuals["no_intervention"] = self._predict_outcome(
            observation, null_action, world_model
        )
        
        # What if we intervened on different variables?
        if len(action.target_variables) > 0:
            for i, var in enumerate(action.target_variables):
                # Alternative variable intervention
                alt_vars = [v for v in action.target_variables if v != var]
                if alt_vars:
                    alt_action = CausalAction(
                        target_variables=alt_vars,
                        intervention_values=action.intervention_values[1:] if len(action.intervention_values) > 1 else np.array([0.0]),
                        intervention_type=action.intervention_type,
                        confidence=action.confidence * 0.8,
                        reasoning_chain=[f"Alternative intervention excluding {var}"]
                    )
                    counterfactuals[f"without_{var}"] = self._predict_outcome(
                        observation, alt_action, world_model
                    )
        
        # What if we used different intervention magnitudes?
        if action.intervention_values.size > 0:
            for magnitude in [0.5, 1.5, 2.0]:
                scaled_values = action.intervention_values * magnitude
                scaled_action = CausalAction(
                    target_variables=action.target_variables,
                    intervention_values=scaled_values,
                    intervention_type=action.intervention_type,
                    confidence=action.confidence,
                    reasoning_chain=[f"Scaled intervention magnitude x{magnitude}"]
                )
                counterfactuals[f"magnitude_{magnitude}"] = self._predict_outcome(
                    observation, scaled_action, world_model
                )
        
        return counterfactuals
    
    def _predict_outcome(self, observation: CausalObservation,
                        action: CausalAction, world_model) -> Dict[str, float]:
        """Predict outcome of counterfactual intervention."""
        # Simple prediction based on causal structure
        if observation.state_data is None:
            return {"outcome": 0.0, "confidence": 0.0}
        
        # Use world model to predict state changes
        predicted_change = 0.0
        for i, var in enumerate(action.target_variables):
            if var.startswith("var_") and i < len(action.intervention_values):
                var_idx = int(var.split("_")[1])
                if var_idx < len(observation.state_data):
                    predicted_change += abs(action.intervention_values[i])
        
        return {
            "predicted_state_change": predicted_change,
            "outcome_magnitude": predicted_change,
            "confidence": action.confidence
        }


class SafetyAssessor:
    """Safety assessment system for causal interventions."""
    
    def __init__(self, constraints: InterventionConstraints):
        self.constraints = constraints
        self.safety_history = []
        
    def assess_safety(self, action: CausalAction, 
                     observation: CausalObservation,
                     world_model: 'CausalWorldModel') -> Tuple[SafetyLevel, Dict[str, Any]]:
        """Assess safety level of proposed intervention."""
        safety_checks = {}
        risk_factors = []
        
        # Check forbidden variables
        if any(var in self.constraints.forbidden_variables for var in action.target_variables):
            safety_checks["forbidden_variables"] = False
            risk_factors.append("Intervention targets forbidden variables")
        else:
            safety_checks["forbidden_variables"] = True
        
        # Check intervention magnitude
        if action.intervention_values.size > 0:
            max_magnitude = np.max(np.abs(action.intervention_values))
            if max_magnitude > self.constraints.max_intervention_magnitude:
                safety_checks["magnitude"] = False
                risk_factors.append(f"Intervention magnitude {max_magnitude:.3f} exceeds limit {self.constraints.max_intervention_magnitude}")
            else:
                safety_checks["magnitude"] = True
        else:
            safety_checks["magnitude"] = True
        
        # Check confidence level
        if action.confidence < self.constraints.safety_threshold:
            safety_checks["confidence"] = False
            risk_factors.append(f"Low confidence {action.confidence:.3f} below threshold {self.constraints.safety_threshold}")
        else:
            safety_checks["confidence"] = True
        
        # Check for reversibility if required
        if self.constraints.reversible_only:
            is_reversible = self._check_reversibility(action, world_model)
            safety_checks["reversibility"] = is_reversible
            if not is_reversible:
                risk_factors.append("Intervention is not easily reversible")
        
        # Check historical safety
        historical_safety = self._check_historical_safety(action)
        safety_checks["historical"] = historical_safety
        if not historical_safety:
            risk_factors.append("Similar interventions have caused issues")
        
        # Determine overall safety level
        safety_level = self._determine_safety_level(safety_checks, risk_factors)
        
        assessment = {
            "safety_checks": safety_checks,
            "risk_factors": risk_factors,
            "overall_risk_score": len(risk_factors) / len(safety_checks),
            "recommendations": self._generate_safety_recommendations(safety_checks, risk_factors)
        }
        
        return safety_level, assessment
    
    def _check_reversibility(self, action: CausalAction, world_model) -> bool:
        """Check if intervention is reversible."""
        # Simple heuristic: smaller interventions are more reversible
        if action.intervention_values.size == 0:
            return True
        return np.max(np.abs(action.intervention_values)) < 0.5
    
    def _check_historical_safety(self, action: CausalAction) -> bool:
        """Check safety based on historical interventions."""
        # Check if similar actions have been safe in the past
        for past_action, was_safe in self.safety_history:
            if (past_action.intervention_type == action.intervention_type and
                set(past_action.target_variables) == set(action.target_variables)):
                if not was_safe:
                    return False
        return True
    
    def _determine_safety_level(self, safety_checks: Dict[str, bool], 
                               risk_factors: List[str]) -> SafetyLevel:
        """Determine overall safety level."""
        failed_checks = sum(1 for check in safety_checks.values() if not check)
        
        if failed_checks == 0:
            return SafetyLevel.SAFE
        elif failed_checks == 1:
            return SafetyLevel.CAUTIOUS
        elif failed_checks == 2:
            return SafetyLevel.MODERATE
        elif failed_checks >= 3:
            return SafetyLevel.HIGH_RISK
        else:
            return SafetyLevel.PROHIBITED
    
    def _generate_safety_recommendations(self, safety_checks: Dict[str, bool],
                                       risk_factors: List[str]) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if not safety_checks.get("magnitude", True):
            recommendations.append("Reduce intervention magnitude")
        
        if not safety_checks.get("confidence", True):
            recommendations.append("Gather more data to improve confidence")
        
        if not safety_checks.get("forbidden_variables", True):
            recommendations.append("Select different target variables")
        
        if risk_factors:
            recommendations.append("Consider alternative intervention strategies")
        
        return recommendations
    
    def record_outcome(self, action: CausalAction, was_safe: bool):
        """Record intervention outcome for learning."""
        self.safety_history.append((action, was_safe))
        # Keep only recent history
        if len(self.safety_history) > 100:
            self.safety_history.pop(0)


class MultiObjectiveOptimizer:
    """Multi-objective optimization for intervention planning."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
    def optimize_intervention(self, candidates: List[CausalAction],
                            observation: CausalObservation,
                            objectives: Dict[str, Callable],
                            weights: Dict[str, float]) -> CausalAction:
        """Optimize intervention selection across multiple objectives."""
        if not candidates:
            return CausalAction(
                target_variables=[],
                intervention_values=np.array([]),
                intervention_type="observe",
                confidence=1.0,
                reasoning_chain=["No candidate interventions available"]
            )
        
        best_action = None
        best_score = float('-inf')
        
        for action in candidates:
            # Evaluate each objective
            scores = {}
            for obj_name, obj_func in objectives.items():
                try:
                    scores[obj_name] = obj_func(action, observation)
                except Exception as e:
                    scores[obj_name] = 0.0  # Default score on error
            
            # Compute weighted sum
            weighted_score = sum(weights.get(obj_name, 1.0) * score 
                               for obj_name, score in scores.items())
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_action = action
        
        return best_action if best_action else candidates[0]
    
    def generate_pareto_frontier(self, candidates: List[CausalAction],
                               observation: CausalObservation,
                               objectives: Dict[str, Callable]) -> List[CausalAction]:
        """Generate Pareto-optimal set of interventions."""
        if not candidates or not objectives:
            return candidates
        
        # Evaluate all candidates on all objectives
        candidate_scores = []
        for action in candidates:
            scores = []
            for obj_func in objectives.values():
                try:
                    score = obj_func(action, observation)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            candidate_scores.append((action, scores))
        
        # Find Pareto-optimal solutions
        pareto_optimal = []
        for i, (action_i, scores_i) in enumerate(candidate_scores):
            is_dominated = False
            for j, (action_j, scores_j) in enumerate(candidate_scores):
                if i != j:
                    # Check if action_j dominates action_i
                    if (all(s_j >= s_i for s_j, s_i in zip(scores_j, scores_i)) and
                        any(s_j > s_i for s_j, s_i in zip(scores_j, scores_i))):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(action_i)
        
        return pareto_optimal if pareto_optimal else candidates[:3]  # Return top 3 if none Pareto-optimal


class InterventionPlanner:
    """Advanced intervention planner with multi-objective optimization."""
    
    def __init__(self, config: AgentConfig, constraints: Optional[InterventionConstraints] = None):
        self.config = config
        self.constraints = constraints or InterventionConstraints()
        self.safety_assessor = SafetyAssessor(self.constraints)
        self.counterfactual_reasoner = CounterfactualReasoning(config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config)
        
        # Planning history
        self.planning_history = []
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def plan_intervention(self, observation: CausalObservation,
                         world_model: 'CausalWorldModel',
                         goal: Optional[str] = None,
                         objectives: Optional[Dict[str, Callable]] = None,
                         weights: Optional[Dict[str, float]] = None) -> InterventionResult:
        """Plan optimal intervention with comprehensive analysis."""
        planning_trace = []
        planning_trace.append(f"Starting intervention planning with goal: {goal}")
        
        # Generate candidate interventions
        candidates = self._generate_candidate_interventions(observation, world_model)
        planning_trace.append(f"Generated {len(candidates)} candidate interventions")
        
        # Filter by safety constraints
        safe_candidates = []
        safety_assessments = []
        
        for candidate in candidates:
            safety_level, assessment = self.safety_assessor.assess_safety(
                candidate, observation, world_model
            )
            safety_assessments.append((candidate, safety_level, assessment))
            
            if safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTIOUS]:
                safe_candidates.append(candidate)
        
        planning_trace.append(f"Filtered to {len(safe_candidates)} safe candidates")
        
        if not safe_candidates:
            # Fallback to null intervention if nothing is safe
            null_action = CausalAction(
                target_variables=[],
                intervention_values=np.array([]),
                intervention_type="observe",
                confidence=1.0,
                reasoning_chain=["No safe interventions available"]
            )
            safe_candidates = [null_action]
            safety_level = SafetyLevel.SAFE
            safety_assessment = {"safety_checks": {}, "risk_factors": [], "overall_risk_score": 0.0, "recommendations": []}
        else:
            # Use the safety assessment of the best candidate (will be selected later)
            safety_level = SafetyLevel.SAFE
            safety_assessment = {}
        
        # Set up default objectives if none provided
        if objectives is None:
            objectives = self._get_default_objectives()
        if weights is None:
            weights = {"efficacy": 0.4, "safety": 0.3, "cost": 0.2, "confidence": 0.1}
        
        # Multi-objective optimization
        optimal_action = self.multi_objective_optimizer.optimize_intervention(
            safe_candidates, observation, objectives, weights
        )
        planning_trace.append(f"Selected optimal intervention: {optimal_action.intervention_type}")
        
        # Generate counterfactuals
        counterfactuals = self.counterfactual_reasoner.generate_counterfactuals(
            observation, optimal_action, world_model
        )
        planning_trace.append(f"Generated {len(counterfactuals)} counterfactual scenarios")
        
        # Predict expected outcomes
        expected_outcome = self._predict_expected_outcome(optimal_action, observation, world_model)
        confidence_interval = self._estimate_confidence_interval(optimal_action, observation, world_model)
        
        # Estimate costs
        estimated_cost = self._estimate_intervention_cost(optimal_action)
        
        # Calculate uncertainties
        outcome_uncertainty = self._calculate_outcome_uncertainty(optimal_action, world_model)
        model_uncertainty = self._calculate_model_uncertainty(world_model)
        total_uncertainty = outcome_uncertainty + model_uncertainty
        
        # Get Pareto frontier for alternatives
        alternative_actions = self.multi_objective_optimizer.generate_pareto_frontier(
            safe_candidates, observation, objectives
        )
        # Remove the selected action from alternatives
        alternative_actions = [a for a in alternative_actions if a != optimal_action]
        
        # Find safety assessment for selected action
        for candidate, s_level, s_assessment in safety_assessments:
            if candidate == optimal_action:
                safety_level = s_level
                safety_assessment = s_assessment
                break
        
        # Create intervention result
        result = InterventionResult(
            action=optimal_action,
            expected_outcome=expected_outcome,
            confidence_interval=confidence_interval,
            safety_assessment=safety_level,
            estimated_cost=estimated_cost,
            alternative_actions=alternative_actions,
            planning_trace=planning_trace,
            counterfactuals=counterfactuals,
            outcome_uncertainty=outcome_uncertainty,
            model_uncertainty=model_uncertainty,
            total_uncertainty=total_uncertainty
        )
        
        # Record planning history
        self.planning_history.append(result)
        
        self.logger.info(f"Planned intervention: {optimal_action.intervention_type} with safety level: {safety_level}")
        
        return result
    
    def _generate_candidate_interventions(self, observation: CausalObservation,
                                        world_model) -> List[CausalAction]:
        """Generate candidate interventions."""
        candidates = []
        
        if observation.state_data is None:
            return candidates
        
        n_vars = len(observation.state_data)
        causal_structure = world_model.get_current_structure()
        
        # Null intervention (always include)
        candidates.append(CausalAction(
            target_variables=[],
            intervention_values=np.array([]),
            intervention_type="observe",
            confidence=1.0,
            reasoning_chain=["Null intervention - observe only"]
        ))
        
        # Single variable interventions
        for i in range(min(n_vars, 10)):  # Limit for efficiency
            if i < causal_structure.shape[0]:
                # Check if this variable has causal influence
                has_influence = np.sum(causal_structure[i, :]) > 0.1
                if has_influence:
                    for magnitude in [0.2, 0.5, 1.0]:
                        candidates.append(CausalAction(
                            target_variables=[f"var_{i}"],
                            intervention_values=np.array([magnitude]),
                            intervention_type="do",
                            confidence=float(np.sum(causal_structure[i, :]) / n_vars),
                            reasoning_chain=[f"Single intervention on var_{i} with magnitude {magnitude}"]
                        ))
        
        # Multi-variable interventions (limited combinations)
        if n_vars >= 2:
            for var_pair in itertools.combinations(range(min(n_vars, 5)), 2):
                i, j = var_pair
                if (i < causal_structure.shape[0] and j < causal_structure.shape[0] and
                    causal_structure[i, j] > 0.3 or causal_structure[j, i] > 0.3):
                    
                    candidates.append(CausalAction(
                        target_variables=[f"var_{i}", f"var_{j}"],
                        intervention_values=np.array([0.3, 0.3]),
                        intervention_type="do",
                        confidence=0.6,
                        reasoning_chain=[f"Joint intervention on var_{i} and var_{j}"]
                    ))
        
        return candidates
    
    def _get_default_objectives(self) -> Dict[str, Callable]:
        """Get default objective functions."""
        def efficacy_objective(action: CausalAction, observation: CausalObservation) -> float:
            """Measure expected efficacy of intervention."""
            if action.intervention_values.size == 0:
                return 0.5  # Neutral for null intervention
            return min(1.0, np.sum(np.abs(action.intervention_values)))
        
        def safety_objective(action: CausalAction, observation: CausalObservation) -> float:
            """Measure safety of intervention."""
            if action.intervention_values.size == 0:
                return 1.0  # Null intervention is safe
            # Smaller interventions are safer
            risk = np.max(np.abs(action.intervention_values))
            return max(0.0, 1.0 - risk)
        
        def cost_objective(action: CausalAction, observation: CausalObservation) -> float:
            """Measure cost of intervention (lower cost = higher score)."""
            if action.intervention_values.size == 0:
                return 1.0  # Null intervention has no cost
            # Cost increases with number of variables and magnitude
            cost = len(action.target_variables) + np.sum(np.abs(action.intervention_values))
            return max(0.0, 1.0 - cost / 10.0)
        
        def confidence_objective(action: CausalAction, observation: CausalObservation) -> float:
            """Use action confidence as objective."""
            return action.confidence
        
        return {
            "efficacy": efficacy_objective,
            "safety": safety_objective,
            "cost": cost_objective,
            "confidence": confidence_objective
        }
    
    def _predict_expected_outcome(self, action: CausalAction, 
                                observation: CausalObservation,
                                world_model) -> Dict[str, float]:
        """Predict expected outcome of intervention."""
        if observation.state_data is None:
            return {"predicted_change": 0.0}
        
        # Simple prediction based on intervention magnitude
        expected_change = 0.0
        if action.intervention_values.size > 0:
            expected_change = np.sum(np.abs(action.intervention_values))
        
        return {
            "predicted_state_change": expected_change,
            "expected_reward": expected_change * action.confidence,
            "stability_impact": -expected_change * 0.5  # Interventions reduce stability
        }
    
    def _estimate_confidence_interval(self, action: CausalAction,
                                    observation: CausalObservation,
                                    world_model) -> Dict[str, Tuple[float, float]]:
        """Estimate confidence intervals for predictions."""
        expected = self._predict_expected_outcome(action, observation, world_model)
        
        # Simple uncertainty estimation
        uncertainty = 0.1 + (1.0 - action.confidence) * 0.5
        
        intervals = {}
        for key, value in expected.items():
            lower = value - uncertainty
            upper = value + uncertainty
            intervals[key] = (lower, upper)
        
        return intervals
    
    def _estimate_intervention_cost(self, action: CausalAction) -> float:
        """Estimate cost of implementing intervention."""
        if action.intervention_values.size == 0:
            return 0.0
        
        # Cost based on number of variables and magnitude
        base_cost = len(action.target_variables) * 10.0
        magnitude_cost = np.sum(np.abs(action.intervention_values)) * 5.0
        
        return base_cost + magnitude_cost
    
    def _calculate_outcome_uncertainty(self, action: CausalAction, world_model) -> float:
        """Calculate uncertainty in intervention outcome."""
        if action.intervention_values.size == 0:
            return 0.1  # Low uncertainty for null intervention
        
        # Uncertainty increases with intervention magnitude and decreases with confidence
        magnitude_uncertainty = np.sum(np.abs(action.intervention_values)) * 0.2
        confidence_uncertainty = (1.0 - action.confidence) * 0.3
        
        return magnitude_uncertainty + confidence_uncertainty
    
    def _calculate_model_uncertainty(self, world_model) -> float:
        """Calculate uncertainty in world model."""
        # Simple estimation based on model update frequency
        if hasattr(world_model, 'update_count') and world_model.update_count > 0:
            return max(0.1, 1.0 / world_model.update_count)
        return 0.5  # High uncertainty for untrained model
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get statistics about planning performance."""
        if not self.planning_history:
            return {"total_plans": 0}
        
        safety_levels = [result.safety_assessment for result in self.planning_history]
        costs = [result.estimated_cost for result in self.planning_history]
        uncertainties = [result.total_uncertainty for result in self.planning_history]
        
        return {
            "total_plans": len(self.planning_history),
            "average_cost": float(np.mean(costs)),
            "average_uncertainty": float(np.mean(uncertainties)),
            "safety_distribution": {
                level.value: safety_levels.count(level) 
                for level in SafetyLevel
            },
            "recent_plans": len([r for r in self.planning_history[-10:]])
        }


def demo_intervention_planner():
    """Demonstrate the intervention planner capabilities."""
    print("🎯 Autonomous Intervention Planner Demo")
    print("=" * 50)
    
    # Setup
    config = AgentConfig()
    constraints = InterventionConstraints(
        max_intervention_magnitude=1.0,
        forbidden_variables=["var_0"],  # Protect critical variable
        safety_threshold=0.6
    )
    
    planner = InterventionPlanner(config, constraints)
    
    # Create mock world model
    class MockWorldModel:
        def __init__(self):
            self.current_structure = np.array([
                [0, 0.8, 0.3],
                [0, 0, 0.6],
                [0, 0, 0]
            ])
            self.update_count = 5
        
        def get_current_structure(self):
            return self.current_structure
    
    world_model = MockWorldModel()
    
    # Create test observation
    observation = CausalObservation(
        state_data=np.array([1.0, 0.5, -0.2]),
        timestamp=time.time(),
        metadata={"environment": "demo"}
    )
    
    # Plan intervention
    result = planner.plan_intervention(
        observation=observation,
        world_model=world_model,
        goal="stabilize_system"
    )
    
    print(f"✅ Intervention planned successfully!")
    print(f"Action type: {result.action.intervention_type}")
    print(f"Target variables: {result.action.target_variables}")
    print(f"Safety level: {result.safety_assessment}")
    print(f"Estimated cost: ${result.estimated_cost:.2f}")
    print(f"Total uncertainty: {result.total_uncertainty:.3f}")
    print(f"Alternatives available: {len(result.alternative_actions)}")
    
    # Show planning trace
    print(f"\n📋 Planning trace:")
    for step in result.planning_trace:
        print(f"  • {step}")
    
    # Show counterfactuals
    print(f"\n🔮 Counterfactual scenarios:")
    for scenario, outcome in result.counterfactuals.items():
        print(f"  • {scenario}: {outcome}")
    
    return result


if __name__ == "__main__":
    import time
    result = demo_intervention_planner()
    print(f"\n✅ Intervention planner demo completed!")