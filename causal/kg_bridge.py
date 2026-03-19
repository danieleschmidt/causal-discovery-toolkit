"""
KGCausalBridge: Connect DocGraph knowledge graphs to causal inference.

DocGraph produces a KG as a list of (subject, predicate, object) triples or
(source, target, metadata) edges.  This bridge:

  1. Ingests a KG edge list and builds a CausalGraph from it.
  2. Optionally uses attached data to fit empirical distributions.
  3. Provides high-level causal queries over the KG:
       - "Does entity A causally influence entity B?"
       - "What is the causal effect of A on B, adjusted for confounders?"
       - "Given observed KG state, what would happen if we intervened on A?"

The KG → Causal DAG mapping:
  - Entities (nodes in KG) become nodes in CausalGraph.
  - KG edges are treated as *candidate* causal edges (not proven).
  - The graph can optionally be refined by PC algorithm if data is available.
  - Causal direction: assumed from KG edge direction (source → target),
    which in legal/document KGs typically represents influence, citation,
    or temporal precedence.

DocGraph integration:
    DocGraph builds entity-relationship KGs from documents (e.g. legal filings,
    scientific papers).  Typical entities: organizations, persons, events,
    outcomes.  Typical relations: "filed_by", "caused_by", "resulted_in",
    "influenced", "preceded".

    This module answers questions like:
      "Did Company A's action (filing) cause Outcome B (penalty)?"
      "Is there a causal chain from Event X to Outcome Y?"
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from .graph import CausalGraph
from .do_calculus import DoCalculusReasoner
from .pc_algorithm import PCAlgorithm


# Predicates that imply causal direction (source causes target)
CAUSAL_PREDICATES = frozenset({
    "causes", "caused_by_inverse", "results_in", "leads_to",
    "influences", "affects", "triggers", "enables", "prevents",
    "filed_against", "resulted_in", "preceded_by_inverse",
    "contributed_to", "imposed_on", "applies_to",
})

# Predicates that imply reverse causal direction (target causes source)
REVERSE_CAUSAL_PREDICATES = frozenset({
    "caused_by", "resulted_from", "preceded_by", "triggered_by",
    "influenced_by", "imposed_by",
})

# Predicates that are non-causal (symmetric / structural)
NON_CAUSAL_PREDICATES = frozenset({
    "is_a", "same_as", "related_to", "co_occurs_with",
    "mentioned_with", "appears_in",
})


class KGCausalBridge:
    """
    Build a CausalGraph from a DocGraph KG and run causal queries.

    Parameters
    ----------
    causal_predicates : set of str, optional
        Predicates to treat as causal (source → target).
        Defaults to CAUSAL_PREDICATES.
    reverse_predicates : set of str, optional
        Predicates where target causes source.
    drop_non_causal : bool
        If True (default), ignore edges with non-causal predicates.
    pc_refine : bool
        If True and data is provided, refine graph with PC algorithm.
    pc_alpha : float
        Significance level for PC algorithm CI tests (default 0.05).

    Examples
    --------
    >>> bridge = KGCausalBridge()
    >>> edges = [
    ...     ("CompanyA", "filed_against", "Regulator"),
    ...     ("Regulator", "imposed_on", "CompanyA"),
    ...     ("Regulator", "resulted_in", "Penalty"),
    ... ]
    >>> bridge.build_from_triples(edges)
    >>> result = bridge.does_cause("CompanyA", "Penalty")
    >>> print(result["explanation"])
    """

    def __init__(
        self,
        causal_predicates: Optional[Set[str]] = None,
        reverse_predicates: Optional[Set[str]] = None,
        drop_non_causal: bool = True,
        pc_refine: bool = False,
        pc_alpha: float = 0.05,
    ) -> None:
        self.causal_predicates = causal_predicates or set(CAUSAL_PREDICATES)
        self.reverse_predicates = reverse_predicates or set(REVERSE_CAUSAL_PREDICATES)
        self.drop_non_causal = drop_non_causal
        self.pc_refine = pc_refine
        self.pc_alpha = pc_alpha

        self.causal_graph: Optional[CausalGraph] = None
        self.reasoner: Optional[DoCalculusReasoner] = None
        self._entity_metadata: Dict[str, Any] = {}
        self._edge_metadata: List[Dict] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_triples(
        self,
        triples: Iterable[Tuple],
        entity_metadata: Optional[Dict[str, Any]] = None,
    ) -> "KGCausalBridge":
        """
        Build CausalGraph from KG triples (subject, predicate, object).

        Parameters
        ----------
        triples : iterable of (subj, pred, obj) or (subj, obj) tuples
            3-tuples use predicate to determine direction.
            2-tuples assume subj → obj direction.
        entity_metadata : dict {entity: metadata_dict}, optional
            Arbitrary metadata about each entity.

        Returns
        -------
        self
        """
        nodes = set()
        edges = []
        edge_meta = []

        for triple in triples:
            if len(triple) == 2:
                subj, obj = triple
                pred = "causes"  # assume causal
            elif len(triple) == 3:
                subj, pred, obj = triple
            else:
                # Longer tuples: take first 3
                subj, pred, obj = triple[0], triple[1], triple[2]

            pred_lower = pred.lower().replace(" ", "_")
            nodes.add(subj)
            nodes.add(obj)

            if pred_lower in self.reverse_predicates:
                # Reverse: obj → subj
                edges.append((obj, subj))
                edge_meta.append({"from": obj, "to": subj, "predicate": pred, "original": (subj, pred, obj)})
            elif pred_lower in NON_CAUSAL_PREDICATES and self.drop_non_causal:
                # Skip non-causal
                continue
            else:
                # Default: subj → obj
                edges.append((subj, obj))
                edge_meta.append({"from": subj, "to": obj, "predicate": pred, "original": (subj, pred, obj)})

        self._entity_metadata = entity_metadata or {}
        self._edge_metadata = edge_meta

        # Build CausalGraph (handle cycles by dropping offending edges)
        cg = CausalGraph(nodes=list(nodes))
        for u, v in edges:
            try:
                cg.add_edge(u, v)
            except ValueError:
                # Would create cycle — skip (log for debugging)
                pass

        self.causal_graph = cg
        self.reasoner = DoCalculusReasoner(cg)
        return self

    def build_from_edge_list(
        self,
        edges: Iterable[Tuple[str, str]],
        node_attrs: Optional[Dict[str, Any]] = None,
    ) -> "KGCausalBridge":
        """
        Build from a simple (source, target) edge list.

        Parameters
        ----------
        edges : iterable of (str, str)
        node_attrs : optional dict {node: attrs}
        """
        triples = [(u, "causes", v) for u, v in edges]
        return self.build_from_triples(triples, entity_metadata=node_attrs)

    def refine_with_data(
        self,
        data: np.ndarray,
        variable_names: List[str],
    ) -> "KGCausalBridge":
        """
        Optionally refine the causal graph skeleton using PC algorithm.

        The PC algorithm's skeleton is used to *prune* edges from the KG-derived
        graph that are not supported by data.  Orientation from KG is preserved
        where consistent with PC results.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        variable_names : list of str  (must be a subset of graph nodes)
        """
        if self.causal_graph is None:
            raise RuntimeError("Build the graph first with build_from_triples().")

        pc = PCAlgorithm(alpha=self.pc_alpha)
        pc.fit(data, variable_names)

        # Prune edges not in PC skeleton
        pc_skeleton = pc.get_skeleton_edges()  # frozensets
        nodes_in_data = set(variable_names)

        edges_to_remove = []
        for u, v in list(self.causal_graph.edges):
            if u in nodes_in_data and v in nodes_in_data:
                if frozenset({u, v}) not in pc_skeleton:
                    edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self.causal_graph.remove_edge(u, v)

        # Attach data for empirical estimation
        self.causal_graph.fit_data(data, variable_names)
        self.reasoner = DoCalculusReasoner(self.causal_graph)
        self._pc_result = pc
        return self

    # ------------------------------------------------------------------
    # High-level causal queries
    # ------------------------------------------------------------------

    def does_cause(
        self,
        cause_entity: str,
        effect_entity: str,
        verbose: bool = True,
    ) -> dict:
        """
        Query: Does *cause_entity* causally influence *effect_entity*?

        Returns structural and identifiability analysis.
        """
        if self.reasoner is None:
            raise RuntimeError("Build the graph first.")
        if cause_entity not in self.causal_graph.nodes:
            raise KeyError(f"Entity {cause_entity!r} not in KG.")
        if effect_entity not in self.causal_graph.nodes:
            raise KeyError(f"Entity {effect_entity!r} not in KG.")

        result = self.reasoner.causal_effect_query(
            cause=cause_entity,
            effect=effect_entity,
            verbose=verbose,
        )

        # Add KG-specific context
        result["kg_entity_metadata"] = {
            "cause": self._entity_metadata.get(cause_entity, {}),
            "effect": self._entity_metadata.get(effect_entity, {}),
        }
        # Find supporting KG edges on the causal path
        result["supporting_kg_edges"] = self._find_supporting_edges(
            cause_entity, effect_entity
        )
        return result

    def causal_effect(
        self,
        cause: str,
        effect: str,
        adjustment_set: Optional[Set[str]] = None,
    ) -> dict:
        """
        Identify and (if data is available) estimate E[effect | do(cause=1)].

        Parameters
        ----------
        cause : str
        effect : str
        adjustment_set : set of str, optional
            If None, uses parents of cause as default adjustment set.
        """
        if self.reasoner is None:
            raise RuntimeError("Build the graph first.")

        id_result = self.reasoner.identify(y={effect}, x_do={cause})

        result = {
            "cause": cause,
            "effect": effect,
            "identifiable": id_result["identifiable"],
            "formula": id_result.get("formula"),
            "method": id_result.get("method"),
        }

        # If data attached, give numerical estimate
        if hasattr(self.causal_graph, "_data"):
            try:
                adj = adjustment_set or id_result.get("adjustment_set") or set()
                # Estimate E[Y | do(X=1)] - E[Y | do(X=0)] as ATE
                interventions_1 = {cause: 1.0}
                interventions_0 = {cause: 0.0}
                e1 = self.causal_graph.interventional_mean(effect, interventions_1)
                e0 = self.causal_graph.interventional_mean(effect, interventions_0)
                result["ate_estimate"] = e1 - e0
                result["e_y_do_x1"] = e1
                result["e_y_do_x0"] = e0
            except Exception as exc:
                result["estimation_error"] = str(exc)

        return result

    def intervention_query(
        self,
        interventions: Dict[str, float],
        outcome: str,
    ) -> dict:
        """
        Answer: "What is the expected outcome if we intervene on these entities?"

        Parameters
        ----------
        interventions : dict {entity: value}
        outcome : str
        """
        if self.reasoner is None:
            raise RuntimeError("Build the graph first.")

        mutilated = self.causal_graph.do(interventions)

        # Identify in mutilated graph what can be computed
        analysis = {
            "interventions": interventions,
            "outcome": outcome,
            "mutilated_graph_nodes": mutilated.nodes,
            "mutilated_graph_edges": mutilated.edges,
            "edges_removed": [
                e for e in self.causal_graph.edges
                if e not in mutilated.edges
            ],
        }

        # Data-based estimate
        if hasattr(self.causal_graph, "_data"):
            try:
                estimate = self.causal_graph.interventional_mean(
                    outcome, interventions
                )
                analysis["interventional_mean"] = estimate
            except Exception as exc:
                analysis["estimation_error"] = str(exc)

        return analysis

    def path_analysis(self, source: str, target: str) -> dict:
        """
        Enumerate all directed paths from source to target and classify them.

        Returns
        -------
        dict with 'direct_paths', 'mediators', 'confounders'
        """
        import networkx as nx

        if self.causal_graph is None:
            raise RuntimeError("Build the graph first.")

        g = self.causal_graph.to_networkx()

        if source not in g or target not in g:
            return {"error": f"Source or target not in graph."}

        try:
            paths = list(nx.all_simple_paths(g, source, target, cutoff=8))
        except Exception:
            paths = []

        # Classify nodes on paths
        mediators = set()
        for path in paths:
            for node in path[1:-1]:  # intermediate
                mediators.add(node)

        # Confounders: common ancestors of source and target (not on directed path)
        anc_src = self.causal_graph.ancestors(source)
        anc_tgt = self.causal_graph.ancestors(target)
        confounders = anc_src & anc_tgt

        return {
            "source": source,
            "target": target,
            "directed_paths": [" → ".join(p) for p in paths],
            "n_paths": len(paths),
            "mediators": sorted(mediators),
            "common_ancestors_confounders": sorted(confounders),
        }

    def get_causal_subgraph(
        self, nodes: Iterable[str]
    ) -> "KGCausalBridge":
        """
        Return a new KGCausalBridge restricted to *nodes* and edges between them.
        """
        nodes_set = set(nodes)
        edges = [
            (u, v) for u, v in self.causal_graph.edges
            if u in nodes_set and v in nodes_set
        ]
        new_bridge = KGCausalBridge(
            causal_predicates=self.causal_predicates,
            reverse_predicates=self.reverse_predicates,
        )
        new_bridge.build_from_edge_list(edges)
        return new_bridge

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_supporting_edges(self, source: str, target: str) -> List[Dict]:
        """Find KG edge metadata on paths from source to target."""
        import networkx as nx
        g = self.causal_graph.to_networkx()
        on_path_edges = set()
        try:
            for path in nx.all_simple_paths(g, source, target, cutoff=6):
                for i in range(len(path) - 1):
                    on_path_edges.add((path[i], path[i + 1]))
        except Exception:
            pass
        return [
            m for m in self._edge_metadata
            if (m["from"], m["to"]) in on_path_edges
        ]

    def summary(self) -> str:
        """Human-readable summary of the KG causal structure."""
        if self.causal_graph is None:
            return "KGCausalBridge: no graph built yet."
        g = self.causal_graph
        lines = [
            f"KGCausalBridge Summary",
            f"  Entities (nodes): {len(g.nodes)}",
            f"  Causal edges:     {len(g.edges)}",
            f"  Valid DAG:        {g.is_dag()}",
            "",
            "Edges:",
        ]
        for u, v in g.edges:
            lines.append(f"  {u} → {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self.causal_graph.nodes) if self.causal_graph else 0
        e = len(self.causal_graph.edges) if self.causal_graph else 0
        return f"KGCausalBridge(nodes={n}, edges={e})"
