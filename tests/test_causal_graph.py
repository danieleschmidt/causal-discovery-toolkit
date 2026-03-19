"""
Tests for CausalGraph: structural queries, do-operator, d-separation,
back-door criterion, and data-driven interventional estimation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal import CausalGraph


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def chain_graph():
    """X → Y → Z (simple chain)."""
    return CausalGraph(
        nodes=["X", "Y", "Z"],
        edges=[("X", "Y"), ("Y", "Z")],
    )


@pytest.fixture
def fork_graph():
    """X ← Z → Y (common cause / fork)."""
    return CausalGraph(
        nodes=["X", "Y", "Z"],
        edges=[("Z", "X"), ("Z", "Y")],
    )


@pytest.fixture
def collider_graph():
    """X → Z ← Y (collider)."""
    return CausalGraph(
        nodes=["X", "Y", "Z"],
        edges=[("X", "Z"), ("Y", "Z")],
    )


@pytest.fixture
def legal_graph():
    """Legal regulatory scenario (5 nodes)."""
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


# ============================================================
# Construction tests
# ============================================================

class TestConstruction:
    def test_empty_graph(self):
        g = CausalGraph()
        assert g.nodes == []
        assert g.edges == []

    def test_nodes_and_edges(self, chain_graph):
        g = chain_graph
        assert set(g.nodes) == {"X", "Y", "Z"}
        assert ("X", "Y") in g.edges
        assert ("Y", "Z") in g.edges

    def test_add_node(self):
        g = CausalGraph()
        g.add_node("A")
        assert "A" in g.nodes

    def test_add_edge(self):
        g = CausalGraph(nodes=["A", "B"])
        g.add_edge("A", "B")
        assert ("A", "B") in g.edges

    def test_cycle_rejected(self):
        g = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        with pytest.raises(ValueError, match="cycle"):
            g.add_edge("B", "A")

    def test_is_dag(self, chain_graph):
        assert chain_graph.is_dag()


# ============================================================
# Structural query tests
# ============================================================

class TestStructuralQueries:
    def test_parents_chain(self, chain_graph):
        assert chain_graph.parents("X") == set()
        assert chain_graph.parents("Y") == {"X"}
        assert chain_graph.parents("Z") == {"Y"}

    def test_children_chain(self, chain_graph):
        assert chain_graph.children("X") == {"Y"}
        assert chain_graph.children("Y") == {"Z"}
        assert chain_graph.children("Z") == set()

    def test_ancestors(self, chain_graph):
        assert chain_graph.ancestors("Z") == {"X", "Y"}
        assert chain_graph.ancestors("X") == set()

    def test_descendants(self, chain_graph):
        assert chain_graph.descendants("X") == {"Y", "Z"}
        assert chain_graph.descendants("Z") == set()

    def test_markov_blanket_chain_middle(self, chain_graph):
        # MB(Y) = parents(Y) ∪ children(Y) ∪ coparents-of-children(Y)
        mb = chain_graph.markov_blanket("Y")
        assert "X" in mb  # parent
        assert "Z" in mb  # child

    def test_markov_blanket_collider_center(self, collider_graph):
        # Z is collider, MB(Z) = {X, Y} (both parents, no children)
        mb = collider_graph.markov_blanket("Z")
        assert "X" in mb
        assert "Y" in mb

    def test_topological_order(self, legal_graph):
        order = legal_graph.topological_order()
        # Penalty must come after everything else
        assert order.index("Penalty") > order.index("Legislation")
        assert order.index("Penalty") > order.index("Enforcement")
        assert order.index("Legislation") > order.index("Lobbying")
        assert order.index("Enforcement") > order.index("Lobbying")


# ============================================================
# d-separation tests
# ============================================================

class TestDSeparation:
    def test_chain_blocked_by_middle(self, chain_graph):
        # X — Y — Z: X ⊥ Z | Y
        assert chain_graph.d_separated({"X"}, {"Z"}, {"Y"})

    def test_chain_open_without_conditioning(self, chain_graph):
        # X and Z are NOT d-separated by ∅
        assert not chain_graph.d_separated({"X"}, {"Z"}, set())

    def test_fork_blocked_by_confounder(self, fork_graph):
        # X ← Z → Y: X ⊥ Y | Z
        assert fork_graph.d_separated({"X"}, {"Y"}, {"Z"})

    def test_fork_open_without_conditioning(self, fork_graph):
        # X and Y are NOT d-separated by ∅ (Z is confounder)
        assert not fork_graph.d_separated({"X"}, {"Y"}, set())

    def test_collider_closed_without_conditioning(self, collider_graph):
        # X → Z ← Y: X ⊥ Y | ∅ (collider blocks)
        assert collider_graph.d_separated({"X"}, {"Y"}, set())

    def test_collider_opens_when_conditioned(self, collider_graph):
        # X → Z ← Y: X NOT ⊥ Y | Z (conditioning on collider opens path)
        assert not collider_graph.d_separated({"X"}, {"Y"}, {"Z"})

    def test_legal_graph_lobbying_violation_independent(self, legal_graph):
        # Lobbying and Violation have no common ancestors → d-separated by ∅
        assert legal_graph.d_separated({"Lobbying"}, {"Violation"}, set())

    def test_legal_graph_lobbying_penalty_not_dsep(self, legal_graph):
        # There is a directed path Lobbying → Legislation → Penalty
        assert not legal_graph.d_separated({"Lobbying"}, {"Penalty"}, set())


# ============================================================
# do-operator tests
# ============================================================

class TestDoOperator:
    def test_do_removes_incoming_edges(self, fork_graph):
        """do(X) cuts edges into X — fork graph Z → X should be removed."""
        mutilated = fork_graph.do({"X": 1})
        # Z → X should be gone
        assert ("Z", "X") not in mutilated.edges
        # Z → Y should remain
        assert ("Z", "Y") in mutilated.edges

    def test_do_preserves_outgoing_edges(self, chain_graph):
        mutilated = chain_graph.do({"X": 0})
        # X → Y should still exist
        assert ("X", "Y") in mutilated.edges

    def test_do_does_not_modify_original(self, fork_graph):
        original_edges = set(fork_graph.edges)
        _ = fork_graph.do({"X": 1})
        assert set(fork_graph.edges) == original_edges

    def test_do_stores_intervention_value(self, fork_graph):
        mutilated = fork_graph.do({"X": 5})
        assert mutilated.interventions == {"X": 5}

    def test_do_unknown_node_raises(self, chain_graph):
        with pytest.raises(KeyError):
            chain_graph.do({"UNKNOWN": 1})

    def test_do_makes_intervention_d_separated(self, fork_graph):
        """After do(X), X should be d-separated from Z (no incoming paths to X)."""
        mutilated = fork_graph.do({"X": 1})
        # In mutilated graph, Z no longer has edge to X
        assert mutilated.d_separated({"X"}, {"Z"}, set())


# ============================================================
# Back-door criterion tests
# ============================================================

class TestBackdoor:
    def test_parents_satisfy_backdoor(self, legal_graph):
        """Parents of Violation (empty) = trivially satisfies back-door for V→P."""
        # Violation is a root — back-door adjustment with ∅
        assert legal_graph.satisfies_backdoor(
            "Violation", "Penalty", set()
        )

    def test_nonempty_adjustment(self, legal_graph):
        """Parents of Enforcement = {Lobbying, Violation} should block back-doors."""
        adj = legal_graph.parents("Enforcement")
        # Enforcement is not a root, so check parents
        result = legal_graph.satisfies_backdoor("Enforcement", "Penalty", adj)
        assert result  # {Lobbying, Violation} is a valid adjustment set

    def test_descendant_invalidates_backdoor(self, legal_graph):
        """Descendants of X cannot be in the adjustment set."""
        # Legislation and Enforcement are descendants of Lobbying
        invalid_adj = {"Legislation"}
        result = legal_graph.satisfies_backdoor("Lobbying", "Penalty", invalid_adj)
        assert not result  # Legislation is a descendant of Lobbying → invalid


# ============================================================
# Data-driven estimation
# ============================================================

class TestDataDrivenEstimation:
    @pytest.fixture
    def data_and_graph(self, legal_graph):
        """Generate synthetic data from linear SEM."""
        np.random.seed(99)
        N = 1000
        lob = np.random.normal(0, 1, N)
        leg = 0.6 * lob + np.random.normal(0, 0.5, N)
        vio = np.random.normal(0, 1, N)
        enf = 0.5 * lob + 0.7 * vio + np.random.normal(0, 0.5, N)
        pen = 0.4 * leg + 0.8 * enf + np.random.normal(0, 0.3, N)
        data = np.column_stack([lob, leg, vio, enf, pen])
        var_names = ["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"]
        legal_graph.fit_data(data, var_names)
        return legal_graph, data, var_names

    def test_observational_mean_reasonable(self, data_and_graph):
        g, data, names = data_and_graph
        mean = g.observational_mean("Penalty")
        # Penalty should be near 0 (all inputs are ~N(0,1))
        assert abs(mean) < 0.5

    def test_interventional_higher_enforcement_higher_penalty(self, data_and_graph):
        g, data, names = data_and_graph
        hi = g.interventional_mean("Penalty", {"Enforcement": 2.0})
        lo = g.interventional_mean("Penalty", {"Enforcement": -2.0})
        # Higher enforcement → higher penalty (positive coefficient 0.8)
        assert hi > lo

    def test_no_data_raises(self):
        g = CausalGraph(nodes=["A", "B"], edges=[("A", "B")])
        with pytest.raises(RuntimeError, match="fit_data"):
            g.observational_mean("B")
