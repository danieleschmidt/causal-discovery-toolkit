"""
Autonomous Causal AI Agents
===========================

Revolutionary autonomous agents that can reason about causality, plan interventions,
and learn from their actions in complex environments.

Research Breakthrough:
- Causal reasoning + autonomous decision making
- Multi-modal environment understanding
- Intervention planning with uncertainty quantification
- Self-improving causal world models

Target Venues: Nature AI 2026, ICML 2026, AAAI 2026
"""

try:
    from .causal_reasoning_agent import CausalReasoningAgent, AgentConfig
    from .intervention_planner import InterventionPlanner, InterventionResult
    from .causal_world_model import CausalWorldModel, WorldModelConfig
    from .agent_memory import AgentMemory, CausalMemoryBank
except ImportError as e:
    # For development, these will be implemented progressively
    import warnings
    warnings.warn(f"Agent modules not fully available: {e}")

__version__ = "0.1.0"
__author__ = "Terragon Labs - Autonomous SDLC"

__all__ = [
    "CausalReasoningAgent",
    "AgentConfig", 
    "InterventionPlanner",
    "InterventionResult",
    "CausalWorldModel",
    "WorldModelConfig",
    "AgentMemory",
    "CausalMemoryBank"
]