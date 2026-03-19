"""
Tests for DoCalculusReasoner (3 rules of do-calculus),
PCAlgorithm (skeleton-finding), and KGCausalBridge.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal import CausalGraph, PCAlgorithm, DoCalculusReasoner, KGCausalBridge


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def chain():
    """X → Y → Z."""
    return CausalGraph(nodes=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z")])


@pytest.fixture
def fork():
    """X ← Z → Y (common cause)."""
    return CausalGraph(nodes=["X", "Y", "Z"], edges=[("Z", "X"), ("Z", "Y")])


@pytest.fixture
def collider():
    """X → Z ← Y."""
    return CausalGraph(nodes=["X", "Y", "Z"], edges=[("X", "Z"), ("Y", "Z")])


@pytest.fixture
def frontdoor_graph():
    """
    Classic front-door graph (Pearl 2009):
        X → M → Y
        X ← U → Y  (U = hidden confounder, not explicit in graph)
    Without U in the graph, this is just X → M → Y.
    We add a backdoor X ← C → Y to make it more interesting.
        C → X, C → Y, X → M → Y
    """
    return CausalGraph(
        nodes=["C", "X", "M", "Y"],
        edges=[("C", "X"), ("C", "Y"), ("X", "M"), ("M", "Y")],
    )


@pytest.fixture
def legal_graph():
    return CausalGraph(
        nodes=["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"],
        edges=[
            ("Lobbying",    "Legislation"),
            ("Lobbying",    "Enforcement"),
            ("Legislation", "Penalty"),
            ("Violation",   "Enforcement"),
            ("Enforcement", "Penalty"),
        ],
    )


@pytest.fixture
def linear_data():
    """Synthetic linear Gaussian data from the legal graph SEM."""
    np.random.seed(42)
    N = 800
    lob = np.random.normal(0, 1, N)
    leg = 0.6 * lob + np.random.normal(0, 0.5, N)
    vio = np.random.normal(0, 1, N)
    enf = 0.5 * lob + 0.7 * vio + np.random.normal(0, 0.5, N)
    pen = 0.4 * leg + 0.8 * enf + np.random.normal(0, 0.3, N)
    data = np.column_stack([lob, leg, vio, enf, pen])
    names = ["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"]
    return data, names


# ============================================================
# DoCalculusReasoner: construction
# ============================================================

class TestReasonerConstruction:
    def test_requires_dag(self):
        """Cyclic graph should raise."""
        g = CausalGraph(nodes=["A", "B"])
        # Manually add cycle to underlying networkx (bypass our guard)
        g._g.add_edge("A", "B")
        g._g.add_edge("B", "A")
        with pytest.raises(ValueError, match="DAG"):
            DoCalculusReasoner(g)

    def test_valid_graph(self, chain):
        r = DoCalculusReasoner(chain)
        assert r is not None


# ============================================================
# Rule 1: Insertion/deletion of observations
# ============================================================

class TestRule1:
    def test_rule1_chain_blocked_by_middle(self, chain):
        """
        G_{X̄}: cut into X (X has no parents anyway).
        Y is in the middle of chain X→Y→Z.
        P(Z | do(X), Y) = P(Z | do(X))?  No — Y is the mediator!
        """
        r = DoCalculusReasoner(chain)
        # Y is NOT d-separated from Z given X in G_{X̄}
        result = r.check_rule1(y={"Z"}, z={"Y"}, x_do={"X"}, w=set())
        assert not result  # can't ignore Y

    def test_rule1_fork_can_ignore_irrelevant(self, fork):
        """
        Fork: X ← Z → Y.  After do(Z), X is root (cut in) and
        Y is also cut (Z has no parents).  X ⊥ Y | ∅ in G_{Z̄}.
        Check if we can ignore X when conditioning on P(Y | do(Z), X).
        """
        r = DoCalculusReasoner(fork)
        # G_{Z̄}: no incoming to Z (Z has no parents, so no change).
        # Is X d-separated from Y given Z in G_{Z̄}?
        result = r.check_rule1(y={"Y"}, z={"X"}, x_do={"Z"}, w=set())
        # In G_{Z̄} (same as G), Z → X and Z → Y.  X ⊥ Y | Z → True
        assert result

    def test_rule1_independent_nodes(self):
        """Two completely disconnected nodes: always d-separated."""
        g = CausalGraph(nodes=["A", "B", "C"], edges=[("A", "B")])
        r = DoCalculusReasoner(g)
        # C has no connection to B; P(B | do(A), C) = P(B | do(A))
        result = r.check_rule1(y={"B"}, z={"C"}, x_do={"A"}, w=set())
        assert result


# ============================================================
# Rule 2: Action/observation exchange
# ============================================================

class TestRule2:
    def test_rule2_root_node(self, chain):
        """
        X is a root in chain X→Y→Z.  In G_{X̄,Ȳ} (cut outgoing of X,
        cut incoming of nothing), X ⊥ Y given ∅? 
        After cutting X→Y (outgoing of X), X and Y are disconnected → True.
        """
        r = DoCalculusReasoner(chain)
        # Demote do(X) to obs(X): P(Z|do(X)) = P(Z|X)?
        result = r.check_rule2(y={"Z"}, z={"X"}, x_do=set(), w=set())
        assert result  # X is exogenous (root), no backdoor

    def test_rule2_fork_violation(self, fork):
        """
        Fork: X ← Z → Y.
        Can we demote do(X) to obs(X)?
        G_{X̄,Z̄}: cut outgoing of X (none), cut incoming of X (Z→X removed).
        But we're asking about do(Z) → obs(Z): cut outgoing of Z.
        In G_{Z̄} (Z's outgoing removed), Z and Y are d-separated.
        """
        r = DoCalculusReasoner(fork)
        # P(Y|do(Z)) = P(Y|Z)? Cut outgoing of Z.  Z→Y removed.
        # Z ⊥ Y | ∅ in G_{Z̄}? Yes (no paths left).
        result = r.check_rule2(y={"Y"}, z={"Z"}, x_do=set(), w=set())
        assert result

    def test_rule2_chain_middle(self, chain):
        """
        Chain X→Y→Z: demote do(Y) to obs(Y)?
        G_{Ȳ}: cut outgoing of Y (Y→Z removed).
        Y ⊥ Z | ∅? After cutting Y→Z, Y and Z are disconnected → True.
        """
        r = DoCalculusReasoner(chain)
        result = r.check_rule2(y={"Z"}, z={"Y"}, x_do=set(), w=set())
        assert result


# ============================================================
# Rule 3: Insertion/deletion of actions
# ============================================================

class TestRule3:
    def test_rule3_disconnected_cause(self):
        """
        A → B → C, and D is unconnected.
        P(C | do(D)) = P(C)? D has no effect on C → True.
        """
        g = CausalGraph(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C")],
        )
        r = DoCalculusReasoner(g)
        result = r.check_rule3(y={"C"}, z={"D"}, x_do=set(), w=set())
        assert result  # D disconnected from C

    def test_rule3_root_intervention_blocked(self, chain):
        """
        Chain X→Y→Z: P(Z | do(X)) = P(Z)?
        X IS an ancestor of Z, so rule 3 should NOT apply (can't drop do(X)).
        """
        r = DoCalculusReasoner(chain)
        result = r.check_rule3(y={"Z"}, z={"X"}, x_do=set(), w=set())
        # X causes Z, so rule 3 fails (can't drop do(X))
        assert not result

    def test_rule3_upstream_blocked(self, legal_graph):
        """
        Can we drop do(Lobbying) from P(Penalty | do(Lobbying))?
        Lobbying causes Penalty through multiple paths → should fail.
        """
        r = DoCalculusReasoner(legal_graph)
        result = r.check_rule3(
            y={"Penalty"}, z={"Lobbying"}, x_do=set(), w=set()
        )
        assert not result  # Lobbying causally affects Penalty


# ============================================================
# Identification
# ============================================================

class TestIdentification:
    def test_identify_root_cause(self, chain):
        """X is a root — P(Z | do(X)) = P(Z | X)."""
        r = DoCalculusReasoner(chain)
        result = r.identify(y={"Z"}, x_do={"X"})
        assert result["identifiable"]
        assert "direct" in result["method"] or "confounding" in result["method"]

    def test_identify_with_backdoor_adjustment(self, frontdoor_graph):
        """C → X → M → Y, C → Y.  P(Y | do(X)) via back-door adj on C."""
        r = DoCalculusReasoner(frontdoor_graph)
        result = r.identify(y={"Y"}, x_do={"X"})
        assert result["identifiable"]
        assert "back-door" in result["method"] or "direct" in result["method"]

    def test_causal_effect_query_has_path(self, legal_graph):
        r = DoCalculusReasoner(legal_graph)
        result = r.causal_effect_query("Lobbying", "Penalty")
        assert result["has_causal_path"]
        assert result["identifiable"]

    def test_causal_effect_query_no_path(self, legal_graph):
        """Penalty has no descendants in legal_graph."""
        r = DoCalculusReasoner(legal_graph)
        result = r.causal_effect_query("Penalty", "Lobbying")
        assert not result["has_causal_path"]

    def test_causal_effect_violation_to_penalty(self, legal_graph):
        r = DoCalculusReasoner(legal_graph)
        result = r.causal_effect_query("Violation", "Penalty")
        assert result["has_causal_path"]
        assert result["identifiable"]


# ============================================================
# PC Algorithm
# ============================================================

class TestPCAlgorithm:
    def test_fit_returns_self(self, linear_data):
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01)
        result = pc.fit(data, names)
        assert result is pc

    def test_skeleton_found(self, linear_data):
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
        pc.fit(data, names)
        assert pc.skeleton_ is not None
        assert len(pc.skeleton_) > 0

    def test_true_edges_in_skeleton(self, linear_data):
        """True skeleton edges should be discovered."""
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
        pc.fit(data, names)
        skeleton = pc.skeleton_

        # True edges in the legal graph:
        true_edges = [
            frozenset({"Lobbying", "Legislation"}),
            frozenset({"Lobbying", "Enforcement"}),
            frozenset({"Legislation", "Penalty"}),
            frozenset({"Violation", "Enforcement"}),
            frozenset({"Enforcement", "Penalty"}),
        ]
        found = sum(1 for e in true_edges if e in skeleton)
        # Expect at least 4 of 5 true edges to be found
        assert found >= 4, f"Only {found}/5 true edges found: {skeleton}"

    def test_false_edges_not_in_skeleton(self, linear_data):
        """Non-adjacent pairs should be separated."""
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
        pc.fit(data, names)
        skeleton = pc.skeleton_

        # Lobbying and Violation are d-separated (no common ancestor)
        assert frozenset({"Lobbying", "Violation"}) not in skeleton

    def test_separating_sets_recorded(self, linear_data):
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
        pc.fit(data, names)
        # Some separating sets should be stored
        assert len(pc.separating_sets_) > 0

    def test_causal_graph_is_dag(self, linear_data):
        data, names = linear_data
        pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
        pc.fit(data, names)
        assert pc.causal_graph_.is_dag()

    def test_small_graph_independent(self):
        """Completely independent variables → no edges in skeleton."""
        np.random.seed(0)
        data = np.random.normal(0, 1, (300, 3))
        pc = PCAlgorithm(alpha=0.01)
        pc.fit(data, ["A", "B", "C"])
        # With independent data, skeleton should be empty or very sparse
        # (false positives possible at alpha=0.01 but rare)
        assert len(pc.skeleton_) <= 2  # allow at most 2 spurious edges

    def test_fully_connected_chain(self):
        """Chain X→Y→Z — PC should find all 2 skeleton edges."""
        np.random.seed(1)
        N = 500
        x = np.random.normal(0, 1, N)
        y = 0.8 * x + np.random.normal(0, 0.3, N)
        z = 0.8 * y + np.random.normal(0, 0.3, N)
        data = np.column_stack([x, y, z])
        pc = PCAlgorithm(alpha=0.01)
        pc.fit(data, ["X", "Y", "Z"])
        assert frozenset({"X", "Y"}) in pc.skeleton_
        assert frozenset({"Y", "Z"}) in pc.skeleton_


# ============================================================
# KGCausalBridge
# ============================================================

class TestKGCausalBridge:
    @pytest.fixture
    def legal_bridge(self):
        bridge = KGCausalBridge()
        bridge.build_from_triples([
            ("CompanyA",    "resulted_in",   "Filing"),
            ("Filing",      "triggered",     "Investigation"),
            ("Investigation","resulted_in",  "Penalty"),
            ("Violation",   "leads_to",      "Investigation"),
            ("Regulator",   "imposed_on",    "CompanyA"),
        ], entity_metadata={
            "CompanyA": {"type": "organization"},
            "Penalty":  {"type": "outcome"},
        })
        return bridge

    def test_build_creates_dag(self, legal_bridge):
        assert legal_bridge.causal_graph is not None
        assert legal_bridge.causal_graph.is_dag()

    def test_nodes_in_graph(self, legal_bridge):
        nodes = set(legal_bridge.causal_graph.nodes)
        assert "CompanyA" in nodes
        assert "Penalty" in nodes
        assert "Investigation" in nodes

    def test_does_cause_has_path(self, legal_bridge):
        result = legal_bridge.does_cause("CompanyA", "Penalty")
        assert result["has_causal_path"]

    def test_does_cause_no_path(self, legal_bridge):
        result = legal_bridge.does_cause("Penalty", "CompanyA")
        assert not result["has_causal_path"]

    def test_path_analysis(self, legal_bridge):
        pa = legal_bridge.path_analysis("Filing", "Penalty")
        assert "Filing → Investigation → Penalty" in pa["directed_paths"]
        assert "Investigation" in pa["mediators"]

    def test_reverse_predicate(self):
        """caused_by inverts the edge direction."""
        bridge = KGCausalBridge()
        bridge.build_from_triples([
            ("Penalty", "caused_by", "Violation"),
        ])
        # Edge should be Violation → Penalty
        assert ("Violation", "Penalty") in bridge.causal_graph.edges
        assert ("Penalty", "Violation") not in bridge.causal_graph.edges

    def test_edge_list_build(self):
        bridge = KGCausalBridge()
        bridge.build_from_edge_list([("A", "B"), ("B", "C")])
        assert ("A", "B") in bridge.causal_graph.edges
        assert ("B", "C") in bridge.causal_graph.edges

    def test_intervention_query(self, legal_bridge):
        result = legal_bridge.intervention_query(
            interventions={"Investigation": 1.0},
            outcome="Penalty",
        )
        assert "edges_removed" in result
        # Filing → Investigation and Violation → Investigation should be cut
        removed = result["edges_removed"]
        assert any("Investigation" in str(e) for e in removed)

    def test_summary_string(self, legal_bridge):
        s = legal_bridge.summary()
        assert "Entities" in s
        assert "Causal edges" in s

    def test_subgraph(self, legal_bridge):
        sub = legal_bridge.get_causal_subgraph(["Filing", "Investigation", "Penalty"])
        assert set(sub.causal_graph.nodes) == {"Filing", "Investigation", "Penalty"}

    def test_unknown_cause_raises(self, legal_bridge):
        with pytest.raises(KeyError):
            legal_bridge.does_cause("UNKNOWN_ENTITY", "Penalty")

    def test_cycle_edge_skipped(self):
        """Edges that would create a cycle should be silently skipped."""
        bridge = KGCausalBridge()
        # A → B → A would cycle
        bridge.build_from_triples([
            ("A", "causes", "B"),
            ("B", "causes", "A"),  # would create cycle
        ])
        # Graph should still be a valid DAG
        assert bridge.causal_graph.is_dag()


# ============================================================
# Counterfactual reasoning
# ============================================================

class TestCounterfactual:
    def test_counterfactual_structural_analysis(self, legal_graph):
        r = DoCalculusReasoner(legal_graph)
        result = r.counterfactual_query(
            outcome="Penalty",
            factual_vals={"Lobbying": 1.0, "Violation": 0.5},
            counterfactual_intervention={"Violation": 2.0},
        )
        assert "structural_analysis" in result
        analysis = result["structural_analysis"]
        # Violation affects Enforcement and Penalty
        assert "Enforcement" in analysis["nodes_affected_by_intervention"]
        assert "Penalty" in analysis["nodes_affected_by_intervention"]

    def test_counterfactual_with_data(self, legal_graph, linear_data):
        data, names = linear_data
        r = DoCalculusReasoner(legal_graph)
        result = r.counterfactual_query(
            outcome="Penalty",
            factual_vals={"Lobbying": 1.0},
            counterfactual_intervention={"Enforcement": 3.0},
            data=data,
            variable_names=names,
        )
        assert "data_estimate" in result
        # High Enforcement → positive Penalty estimate
        assert result["data_estimate"] > 0
