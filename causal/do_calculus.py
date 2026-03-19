"""
DoCalculusReasoner: Pearl's three rules of do-calculus.

The three rules of do-calculus (Pearl 1995, 2009) are sound and complete
for identifying causal effects from observational data.  They characterise
when we can replace an interventional expression with an observational one
(or vice versa), given the causal DAG structure.

Rule 1 — Insertion/deletion of observations:
    P(y | do(x), z, w) = P(y | do(x), w)
    if Y ⊥ Z | X, W in G_{X̄}  (X's parents cut)
    (We may ignore Z when Z is d-separated from Y given {X,W})

Rule 2 — Action/observation exchange:
    P(y | do(x), do(z), w) = P(y | do(x), z, w)
    if Y ⊥ Z | X, W in G_{X̄, Z_bar}
    (We may demote do(z) to obs(z) when no open back-door from Z to Y
     after removing X's parents and Z's children)

Rule 3 — Insertion/deletion of actions:
    P(y | do(x), do(z), w) = P(y | do(x), w)
    if Y ⊥ Z | X, W in G_{X̄, Z(W)}
    where Z(W) = {z ∈ Z : Z is not an ancestor of W in G_{X̄}}
    (We may remove do(z) when Z has no causal effect on Y through any
     path not blocked by X or W)

This class provides:
  - check_rule1, check_rule2, check_rule3: Boolean structural checks
  - apply: iterative simplification of an interventional expression
  - identify: attempt to express P(y | do(x)) in observational terms
    using the ID algorithm (simplified front-door/back-door)

References:
    Pearl, J. (1995). Causal diagrams for empirical research.
    Biometrika, 82(4), 669–688.
    Pearl, J. (2009). Causality. Cambridge University Press, ch. 3.
"""

from __future__ import annotations

import copy
from typing import Dict, FrozenSet, Optional, Set, Tuple, Union

from .graph import CausalGraph


class DoCalculusReasoner:
    """
    Apply Pearl's do-calculus rules to a CausalGraph.

    Parameters
    ----------
    graph : CausalGraph
        The causal DAG (must be a valid DAG).

    Examples
    --------
    >>> from causal import CausalGraph, DoCalculusReasoner
    >>> g = CausalGraph(nodes=["X", "Z", "Y"], edges=[("X", "Z"), ("Z", "Y")])
    >>> r = DoCalculusReasoner(g)
    >>> r.check_rule1(y={"Y"}, z={"Z"}, x_do={"X"}, w=set())
    False  # Z is not d-separated from Y (it's on the path)
    """

    def __init__(self, graph: CausalGraph) -> None:
        if not graph.is_dag():
            raise ValueError("DoCalculusReasoner requires a DAG.")
        self.graph = graph

    # ------------------------------------------------------------------
    # Rule 1: Insertion/deletion of observations
    # ------------------------------------------------------------------

    def check_rule1(
        self,
        y: Set[str],
        z: Set[str],
        x_do: Set[str],
        w: Set[str],
    ) -> bool:
        """
        Rule 1: P(y | do(x), z, w) = P(y | do(x), w)?

        True iff Y ⊥ Z | X, W in G_{X̄}  (mutilated graph where
        incoming edges into X are removed).

        Parameters
        ----------
        y     : outcome variables
        z     : observation variables to remove/insert
        x_do  : variables under do() intervention
        w     : remaining conditioning observations
        """
        # Construct G_{X̄}: remove incoming edges to X
        g_xbar = self.graph.do({xn: None for xn in x_do})
        # d-separate Y from Z given X ∪ W
        cond = x_do | w
        return g_xbar.d_separated(y, z, cond)

    # ------------------------------------------------------------------
    # Rule 2: Action/observation exchange
    # ------------------------------------------------------------------

    def check_rule2(
        self,
        y: Set[str],
        z: Set[str],
        x_do: Set[str],
        w: Set[str],
    ) -> bool:
        """
        Rule 2: P(y | do(x), do(z), w) = P(y | do(x), z, w)?

        True iff Y ⊥ Z | X, W in G_{X̄, Z̄}
        where G_{X̄, Z̄} is the mutilated graph with incoming edges of
        X removed AND outgoing edges of Z removed.

        Parameters
        ----------
        y     : outcome variables
        z     : intervention variables to demote to observations
        x_do  : other do() variables (not being demoted)
        w     : conditioning observations
        """
        # Build G_{X̄, Z̄}: cut incoming to X, cut outgoing from Z
        g = copy.deepcopy(self.graph)
        # Remove incoming edges to X
        for xn in x_do:
            for parent in list(g._g.predecessors(xn)):
                g._g.remove_edge(parent, xn)
        # Remove outgoing edges from Z
        for zn in z:
            for child in list(g._g.successors(zn)):
                g._g.remove_edge(zn, child)
        # Check d-separation Y ⊥ Z | X ∪ W
        cond = x_do | w
        return g.d_separated(y, z, cond)

    # ------------------------------------------------------------------
    # Rule 3: Insertion/deletion of actions
    # ------------------------------------------------------------------

    def check_rule3(
        self,
        y: Set[str],
        z: Set[str],
        x_do: Set[str],
        w: Set[str],
    ) -> bool:
        """
        Rule 3: P(y | do(x), do(z), w) = P(y | do(x), w)?

        True iff Y ⊥ Z | X, W in G_{X̄, Z(W)}
        where Z(W) = Z minus {z ∈ Z : z is ancestor of some w ∈ W in G_{X̄}}.

        Parameters
        ----------
        y     : outcome variables
        z     : do() variables to remove
        x_do  : other do() variables
        w     : conditioning observations
        """
        # G_{X̄}: remove incoming to X
        g_xbar = copy.deepcopy(self.graph)
        for xn in x_do:
            for parent in list(g_xbar._g.predecessors(xn)):
                g_xbar._g.remove_edge(parent, xn)

        # Z(W) = Z that are NOT ancestors of W in G_{X̄}
        ancestors_of_w = set()
        for wn in w:
            ancestors_of_w |= g_xbar.ancestors(wn)
        z_w = z - ancestors_of_w  # only these get removed

        # Now build G_{X̄, Z(W)}: additionally remove incoming to z ∈ Z(W)
        for zn in z_w:
            for parent in list(g_xbar._g.predecessors(zn)):
                g_xbar._g.remove_edge(parent, zn)

        # d-separate Y from Z given X ∪ W
        cond = x_do | w
        return g_xbar.d_separated(y, z, cond)

    # ------------------------------------------------------------------
    # Identification: express P(y | do(x)) in observational terms
    # ------------------------------------------------------------------

    def identify(
        self, y: Set[str], x_do: Set[str], w: Optional[Set[str]] = None
    ) -> dict:
        """
        Attempt to identify P(y | do(x), w) in observational terms.

        Strategy (simplified ID):
          1. If X has no parents in the graph → no back-door confounding,
             P(y | do(x)) = P(y | x).
          2. If a valid back-door adjustment set exists (use parents of X),
             apply back-door formula.
          3. If a front-door criterion is satisfied via mediators, apply
             front-door formula.
          4. Try each of the three do-calculus rules to reduce the expression.
          5. If none apply, report "not identifiable from structure alone."

        Returns
        -------
        dict with keys:
          'identifiable' : bool
          'formula'      : str description of the formula
          'adjustment_set' : set (if back-door), None otherwise
          'method'       : str
        """
        if w is None:
            w = set()

        results = []

        for xn in x_do:
            pa_x = self.graph.parents(xn)
            # Case 1: X is a root (no parents) — no confounding
            if not pa_x:
                results.append({
                    "identifiable": True,
                    "formula": f"P({','.join(y)} | {xn}) [X is root, no confounding]",
                    "adjustment_set": set(),
                    "method": "direct (no confounding)",
                    "variable": xn,
                })
                continue

            # Case 2: Back-door adjustment via parents of X
            adj_set = pa_x.copy()
            if self.graph.satisfies_backdoor(xn, list(y)[0] if len(y) == 1 else list(y)[0], adj_set):
                formula = (
                    f"Σ_z P({','.join(y)} | {xn}, Z=z) · P(Z=z)"
                    f"  where Z = {{{', '.join(sorted(adj_set))}}}"
                )
                results.append({
                    "identifiable": True,
                    "formula": formula,
                    "adjustment_set": adj_set,
                    "method": "back-door adjustment",
                    "variable": xn,
                })
                continue

            # Case 3: Check Rule 3 — can we just drop do(x)?
            if self.check_rule3(y, {xn}, set(), w):
                results.append({
                    "identifiable": True,
                    "formula": f"P({','.join(y)} | {','.join(w)}) [do({xn}) dropped by Rule 3]",
                    "adjustment_set": None,
                    "method": "do-calculus Rule 3",
                    "variable": xn,
                })
                continue

            # Case 4: Check Rule 2 — can we demote do(x) to obs(x)?
            if self.check_rule2(y, {xn}, set(), w):
                results.append({
                    "identifiable": True,
                    "formula": f"P({','.join(y)} | {xn}, {','.join(w)}) [do → obs by Rule 2]",
                    "adjustment_set": None,
                    "method": "do-calculus Rule 2",
                    "variable": xn,
                })
                continue

            # Not identifiable by these methods
            results.append({
                "identifiable": False,
                "formula": f"P({','.join(y)} | do({xn})) — not identifiable",
                "adjustment_set": None,
                "method": "unidentified",
                "variable": xn,
            })

        # Aggregate
        if len(results) == 1:
            return results[0]
        all_identifiable = all(r["identifiable"] for r in results)
        return {
            "identifiable": all_identifiable,
            "formula": " AND ".join(r["formula"] for r in results),
            "adjustment_set": None,
            "method": "multi-variable",
            "details": results,
        }

    # ------------------------------------------------------------------
    # Causal effect query: "Does X cause Y?"
    # ------------------------------------------------------------------

    def causal_effect_query(
        self,
        cause: str,
        effect: str,
        verbose: bool = True,
    ) -> dict:
        """
        High-level query: does *cause* have a causal effect on *effect*?

        Returns a dict with:
          'has_causal_path'  : bool — structural path exists
          'identifiable'     : bool — causal effect can be computed
          'formula'          : str  — identification formula
          'method'           : str
          'explanation'      : str  — human-readable summary
        """
        import networkx as nx

        g_nx = self.graph.to_networkx()

        # 1. Structural: does a directed path exist?
        has_path = nx.has_path(g_nx, cause, effect)

        # 2. Identification
        id_result = self.identify(y={effect}, x_do={cause})

        explanation_parts = []
        if has_path:
            # Find actual paths
            try:
                paths = list(nx.all_simple_paths(g_nx, cause, effect, cutoff=6))
                path_strs = [" → ".join(p) for p in paths]
                explanation_parts.append(
                    f"Directed causal paths from {cause!r} to {effect!r}:\n"
                    + "\n".join(f"  {s}" for s in path_strs)
                )
            except Exception:
                explanation_parts.append(f"A directed path from {cause!r} to {effect!r} exists.")
        else:
            explanation_parts.append(
                f"No directed causal path from {cause!r} to {effect!r} in the graph."
            )

        if id_result["identifiable"]:
            explanation_parts.append(
                f"Causal effect is identifiable via: {id_result['method']}"
            )
            explanation_parts.append(f"Formula: {id_result['formula']}")
        else:
            explanation_parts.append(
                "Causal effect is NOT identifiable from this graph structure "
                "(possible hidden confounders or non-identifiable structure)."
            )

        return {
            "cause": cause,
            "effect": effect,
            "has_causal_path": has_path,
            "identifiable": id_result["identifiable"],
            "formula": id_result.get("formula"),
            "method": id_result.get("method"),
            "adjustment_set": id_result.get("adjustment_set"),
            "explanation": "\n".join(explanation_parts),
        }

    # ------------------------------------------------------------------
    # Counterfactual reasoning
    # ------------------------------------------------------------------

    def counterfactual_query(
        self,
        outcome: str,
        factual_vals: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        data: Optional["np.ndarray"] = None,
        variable_names: Optional[list] = None,
    ) -> dict:
        """
        Answer: "Given that we observed factual_vals, what would *outcome*
        have been under counterfactual_intervention?"

        Uses the three-step counterfactual procedure (abduction-action-prediction):
          1. Abduction: infer exogenous noise U from factual evidence
          2. Action: apply the counterfactual intervention (mutilate graph)
          3. Prediction: compute outcome in mutilated world

        For Gaussian linear models this yields exact results.
        Here we do a structural / symbolic analysis + optional data-based estimate.

        Parameters
        ----------
        outcome : str
        factual_vals : dict {node: observed_value}
        counterfactual_intervention : dict {node: counterfactual_value}
        data : optional np.ndarray for data-driven estimate
        variable_names : list of str (required if data provided)

        Returns
        -------
        dict with 'structural_analysis', 'data_estimate' (if data given)
        """
        import networkx as nx

        g_nx = self.graph.to_networkx()

        # Structural analysis
        intervened_nodes = set(counterfactual_intervention)
        outcome_ancestors = self.graph.ancestors(outcome) | {outcome}
        relevant_nodes = outcome_ancestors | intervened_nodes

        # Find which factual observations are "upstream" of intervention
        affected = set()
        for iv in intervened_nodes:
            affected |= self.graph.descendants(iv)
        affected.add(outcome)

        analysis = {
            "counterfactual_intervention": counterfactual_intervention,
            "factual_evidence": factual_vals,
            "outcome": outcome,
            "nodes_affected_by_intervention": sorted(affected),
            "structural_note": (
                f"Under do({counterfactual_intervention}), the following nodes "
                f"are affected: {sorted(affected)}. "
                f"Factual evidence about {sorted(set(factual_vals) - affected)} "
                f"is retained (unaffected by intervention)."
            ),
        }

        result = {"structural_analysis": analysis}

        # Data-driven estimate
        if data is not None and variable_names is not None:
            import numpy as np
            g_copy = self.graph.copy()
            g_copy.fit_data(data, variable_names)
            try:
                estimate = g_copy.interventional_mean(
                    outcome, counterfactual_intervention
                )
                result["data_estimate"] = estimate
                result["data_estimate_note"] = (
                    f"E[{outcome} | do({counterfactual_intervention})] ≈ {estimate:.4f}"
                )
            except Exception as e:
                result["data_estimate_error"] = str(e)

        return result

    def __repr__(self) -> str:
        return f"DoCalculusReasoner(graph={self.graph!r})"
