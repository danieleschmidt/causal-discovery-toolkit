"""
causal - Pearl do-calculus and causal discovery for knowledge graphs.

Core modules:
    CausalGraph        - DAG wrapper with do-calculus operations
    PCAlgorithm        - PC algorithm skeleton-finding via conditional independence
    DoCalculusReasoner - Three rules of Pearl's do-calculus
    KGCausalBridge     - DocGraph KG edge list → causal queries
"""

from .graph import CausalGraph
from .pc_algorithm import PCAlgorithm
from .do_calculus import DoCalculusReasoner
from .kg_bridge import KGCausalBridge

__all__ = ["CausalGraph", "PCAlgorithm", "DoCalculusReasoner", "KGCausalBridge"]
__version__ = "1.0.0"
