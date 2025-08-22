"""Causal discovery algorithms."""

from .base import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
from .robust import RobustSimpleLinearCausalModel
from .optimized import OptimizedCausalModel, AdaptiveScalingManager
from .bayesian_network import BayesianNetworkDiscovery, ConstraintBasedDiscovery
from .information_theory import MutualInformationDiscovery, TransferEntropyDiscovery
from .distributed_discovery import DistributedCausalDiscovery, StreamingCausalDiscovery, MemoryEfficientDiscovery
from .robust_ensemble import RobustEnsembleDiscovery, EnsembleResult
from .bioneuro_olfactory import OlfactoryNeuralCausalModel, MultiModalOlfactoryCausalModel, BioneuroFusionResult, OlfactoryNeuralSignal
from .quantum_causal import QuantumCausalDiscovery, QuantumEntanglementCausal
from .neuromorphic_causal import SpikingNeuralCausal, ReservoirComputingCausal
from .topological_causal import PersistentHomologyCausal, AlgebraicTopologyCausal
# Breakthrough research algorithms
from .llm_enhanced_causal import (
    LLMEnhancedCausalDiscovery, 
    LLMInterface, 
    OpenAIInterface,
    MultiAgentLLMConsensus,
    LLMCausalResponse,
    ConfidenceLevel,
    discover_causal_relationships_with_llm
)
from .rl_causal_agent import (
    RLCausalAgent,
    CausalAction,
    ActionType,
    CausalState,
    RewardFunction,
    CurriculumLearning,
    discover_causality_with_rl
)

# Optional torch-dependent imports
try:
    from .foundation_causal import FoundationCausalModel, MetaLearningCausalDiscovery, MultiModalCausalConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from .self_supervised_causal import SelfSupervisedCausalModel, SelfSupervisedCausalConfig
    SELF_SUPERVISED_AVAILABLE = True
except ImportError:
    SELF_SUPERVISED_AVAILABLE = False

__all__ = [
    "CausalDiscoveryModel",
    "SimpleLinearCausalModel", 
    "RobustSimpleLinearCausalModel",
    "OptimizedCausalModel",
    "AdaptiveScalingManager",
    "CausalResult",
    "BayesianNetworkDiscovery",
    "ConstraintBasedDiscovery", 
    "MutualInformationDiscovery",
    "TransferEntropyDiscovery",
    "DistributedCausalDiscovery",
    "StreamingCausalDiscovery", 
    "MemoryEfficientDiscovery",
    "RobustEnsembleDiscovery",
    "EnsembleResult",
    "OlfactoryNeuralCausalModel",
    "MultiModalOlfactoryCausalModel", 
    "BioneuroFusionResult",
    "OlfactoryNeuralSignal",
    "QuantumCausalDiscovery",
    "QuantumEntanglementCausal",
    "SpikingNeuralCausal",
    "ReservoirComputingCausal",
    "PersistentHomologyCausal",
    "AlgebraicTopologyCausal",
    # Breakthrough research algorithms
    "LLMEnhancedCausalDiscovery",
    "LLMInterface", 
    "OpenAIInterface",
    "MultiAgentLLMConsensus",
    "LLMCausalResponse",
    "ConfidenceLevel",
    "discover_causal_relationships_with_llm",
    "RLCausalAgent",
    "CausalAction",
    "ActionType", 
    "CausalState",
    "RewardFunction",
    "CurriculumLearning",
    "discover_causality_with_rl",
    # Add torch-dependent exports only if available
] + (["FoundationCausalModel", "MetaLearningCausalDiscovery", "MultiModalCausalConfig"] if TORCH_AVAILABLE else []) + (["SelfSupervisedCausalModel", "SelfSupervisedCausalConfig"] if SELF_SUPERVISED_AVAILABLE else [])