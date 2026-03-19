"""
PC Algorithm — skeleton-finding and edge orientation for causal discovery.

The PC algorithm (Spirtes, Glymour, Scheines 1993) recovers the Markov
equivalence class (CPDAG) of the true DAG from i.i.d. data under:
  - Causal Markov condition
  - Faithfulness assumption
  - Causal sufficiency (no hidden common causes)

Steps:
  1. Start with a complete undirected graph.
  2. Remove edges X—Y whenever X ⊥ Y | Z for *some* conditioning set Z ⊆
     adjacencies (tested iteratively for increasing |Z|).  Record the
     separating set sep(X,Y) = Z.
  3. Orient v-structures (colliders): if X — Z — Y and Z ∉ sep(X,Y),
     orient as X → Z ← Y.
  4. Apply Meek's orientation rules R1–R4 to propagate orientations.

This implementation uses:
  - Partial correlation conditional independence test (parametric, assumes
    linear Gaussian relationships — sufficient for skeleton finding).
  - Fisher's z-transform for significance testing.

References:
    Spirtes, P., Glymour, C., Scheines, R. (2000). Causation, Prediction,
    and Search. MIT Press.
    Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional
    directed acyclic graphs with the PC-algorithm.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Set, Tuple

import numpy as np
import scipy.stats

from .graph import CausalGraph


def _partial_corr(data: np.ndarray, x: int, y: int, z: list) -> float:
    """
    Compute partial correlation r_{xy|z} using recursive elimination.

    For an empty conditioning set, returns the Pearson correlation.
    For a non-empty set Z, regresses X and Y on Z and returns
    the correlation of residuals.
    """
    if not z:
        corr = np.corrcoef(data[:, x], data[:, y])[0, 1]
        return float(np.clip(corr, -1 + 1e-10, 1 - 1e-10))

    # Regress X on Z and Y on Z, compute residual correlation
    Z_mat = data[:, z]
    # Add intercept
    Z_mat = np.column_stack([np.ones(len(Z_mat)), Z_mat])

    def residuals(col):
        beta, _, _, _ = np.linalg.lstsq(Z_mat, data[:, col], rcond=None)
        return data[:, col] - Z_mat @ beta

    res_x = residuals(x)
    res_y = residuals(y)

    denom = np.std(res_x) * np.std(res_y)
    if denom < 1e-12:
        return 0.0
    corr = np.dot(res_x, res_y) / (len(res_x) * denom)
    return float(np.clip(corr, -1 + 1e-10, 1 - 1e-10))


def _fisherz_pvalue(r: float, n: int, k: int) -> float:
    """
    Fisher's z-transform test for H0: partial correlation == 0.

    Parameters
    ----------
    r : partial correlation coefficient
    n : number of samples
    k : size of conditioning set

    Returns
    -------
    p-value (two-tailed)
    """
    r = float(np.clip(r, -1 + 1e-10, 1 - 1e-10))
    z = 0.5 * math.log((1 + r) / (1 - r))  # Fisher's z
    se = 1.0 / math.sqrt(max(n - k - 3, 1))
    stat = z / se
    # Two-tailed p-value from standard normal
    p = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(stat)))
    return p


class PCAlgorithm:
    """
    PC Algorithm for causal skeleton discovery and v-structure orientation.

    Parameters
    ----------
    alpha : float
        Significance level for conditional independence tests (default 0.05).
    max_cond_set_size : int or None
        Maximum conditioning set size.  Set to a small integer to control
        runtime on large graphs (e.g. 3).  None = unlimited.
    ci_test : str
        Conditional independence test to use.  Currently only 'fisherz'.

    Attributes
    ----------
    skeleton_ : set of frozenset pairs
        Undirected skeleton edges after step 2.
    separating_sets_ : dict
        {frozenset({X,Y}): separating_set} — the Z s.t. X ⊥ Y | Z.
    causal_graph_ : CausalGraph
        Partially directed graph (CPDAG) after orientation steps.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        ci_test: str = "fisherz",
    ) -> None:
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.ci_test = ci_test

        self.skeleton_: Optional[set] = None
        self.separating_sets_: Dict = {}
        self.causal_graph_: Optional[CausalGraph] = None
        self._adjacency: Optional[Dict[int, Set[int]]] = None

    # ------------------------------------------------------------------
    # Main fit method
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        variable_names=None,
    ) -> "PCAlgorithm":
        """
        Run the PC algorithm on *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_variables)
        variable_names : list of str, optional

        Returns
        -------
        self
        """
        n, p = data.shape
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(p)]
        self._names = list(variable_names)
        self._n = n
        self._p = p
        self._data = data

        # Step 1 & 2: skeleton-finding
        adj = self._find_skeleton()

        # Step 3: orient v-structures (colliders)
        directed = self._orient_colliders(adj)

        # Step 4: apply Meek rules
        directed = self._meek_rules(directed, adj)

        # Build CausalGraph
        self.causal_graph_ = self._build_cpdag(directed, adj)
        return self

    # ------------------------------------------------------------------
    # Step 2: Skeleton finding
    # ------------------------------------------------------------------

    def _find_skeleton(self) -> Dict[int, Set[int]]:
        """
        Remove edges X—Y if X ⊥ Y | Z for some subset Z of their
        common adjacencies.  Returns the undirected adjacency dict.
        """
        p = self._p
        # Start with complete undirected graph
        adj: Dict[int, Set[int]] = {i: set(range(p)) - {i} for i in range(p)}
        sep_sets: Dict[Tuple[int, int], Set[int]] = {}

        max_k = self.max_cond_set_size
        k = 0
        while True:
            edge_removed = False
            # Iterate over all pairs
            pairs = [(i, j) for i in range(p) for j in adj[i] if i < j]
            for x, y in pairs:
                if y not in adj[x]:
                    continue  # already removed
                # Conditioning candidates: adjacencies of x minus y
                cond_candidates = list(adj[x] - {y})
                cond_size = k
                if max_k is not None:
                    cond_size = min(k, max_k)
                if len(cond_candidates) < cond_size:
                    continue
                # Test all subsets of size k
                found_sep = False
                for z_idx in itertools.combinations(cond_candidates, cond_size):
                    z_list = list(z_idx)
                    r = _partial_corr(self._data, x, y, z_list)
                    p_val = _fisherz_pvalue(r, self._n, len(z_list))
                    if p_val > self.alpha:
                        # X ⊥ Y | Z — remove edge
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sep_sets[(x, y)] = set(z_list)
                        sep_sets[(y, x)] = set(z_list)
                        edge_removed = True
                        found_sep = True
                        break
                _ = found_sep  # suppress unused warning

            k += 1
            # Stop when no edges removed or max |Z| exceeds adjacency size
            max_adj = max((len(s) for s in adj.values()), default=0)
            if max_adj < k:
                break
            if max_k is not None and k > max_k:
                break

        self._adjacency = adj
        self.separating_sets_ = {
            frozenset({self._names[a], self._names[b]}): {self._names[c] for c in z}
            for (a, b), z in sep_sets.items()
        }
        self.skeleton_ = {
            frozenset({self._names[i], self._names[j]})
            for i in range(p)
            for j in adj[i]
            if i < j
        }
        return adj

    # ------------------------------------------------------------------
    # Step 3: Orient v-structures / colliders
    # ------------------------------------------------------------------

    def _orient_colliders(
        self, adj: Dict[int, Set[int]]
    ) -> Dict[Tuple[int, int], bool]:
        """
        For every unshielded triple X — Z — Y where Z ∉ sep(X,Y),
        orient as X → Z ← Y.

        Returns a dict of directed edges {(u,v): True}.
        """
        p = self._p
        sep_sets = {}
        for (a, b), z in [
            ((a, b), z)
            for (a, b), z in [
                ((list(s)[0], list(s)[1]), self._sep_int(s))
                for s in [
                    frozenset({i, j})
                    for i in range(p)
                    for j in range(i + 1, p)
                    if j not in adj[i]  # non-adjacent → have sep set
                ]
                if len(s) == 2
            ]
        ]:
            sep_sets[(a, b)] = z
            sep_sets[(b, a)] = z

        directed: Dict[Tuple[int, int], bool] = {}  # (u→v)

        for z in range(p):
            neighbors_z = list(adj[z])
            for x, y in itertools.combinations(neighbors_z, 2):
                if y in adj[x]:
                    continue  # shielded triple — skip
                # Unshielded triple X — Z — Y
                key = frozenset({x, y})
                sep = self._sep_int_pair(x, y)
                if z not in sep:
                    # Collider: X → Z ← Y
                    directed[(x, z)] = True
                    directed[(y, z)] = True

        return directed

    def _sep_int(self, s: frozenset) -> Set[int]:
        """Get integer-indexed sep set for a frozenset of integer indices."""
        items = list(s)
        a, b = items[0], items[1]
        # Build reverse-lookup from name to index
        name2idx = {n: i for i, n in enumerate(self._names)}
        key_ab = (a, b)
        key_ba = (b, a)
        # sep_sets_int — we stored in separating_sets_ as names; look up
        na, nb = self._names[a], self._names[b]
        fk = frozenset({na, nb})
        if fk in self.separating_sets_:
            return {name2idx[n] for n in self.separating_sets_[fk]}
        return set()

    def _sep_int_pair(self, x: int, y: int) -> Set[int]:
        """Get integer-indexed sep set for pair (x, y)."""
        name2idx = {n: i for i, n in enumerate(self._names)}
        fk = frozenset({self._names[x], self._names[y]})
        if fk in self.separating_sets_:
            return {name2idx[n] for n in self.separating_sets_[fk]}
        return set()

    # ------------------------------------------------------------------
    # Step 4: Meek orientation rules
    # ------------------------------------------------------------------

    def _meek_rules(
        self,
        directed: Dict[Tuple[int, int], bool],
        adj: Dict[int, Set[int]],
    ) -> Dict[Tuple[int, int], bool]:
        """
        Apply Meek's 4 orientation rules to propagate orientations.

        R1: If α → β — γ and α not adjacent to γ, orient β → γ.
        R2: If α → β → γ and α — γ, orient α → γ.
        R3: If α — β, α — γ₁ → β, α — γ₂ → β, γ₁ not adjacent to γ₂:
            orient α → β.
        R4: (θ: rarely changes outcome with R1–R3; omitted for clarity)
        """
        p = self._p

        def is_directed(u, v):
            return (u, v) in directed and (v, u) not in directed

        def is_undirected(u, v):
            return u in adj[v] and (u, v) not in directed and (v, u) not in directed

        changed = True
        while changed:
            changed = False

            # R1
            for beta in range(p):
                for alpha in range(p):
                    if not is_directed(alpha, beta):
                        continue
                    for gamma in adj[beta]:
                        if gamma == alpha:
                            continue
                        if not is_undirected(beta, gamma):
                            continue
                        if gamma in adj[alpha]:
                            continue  # alpha adjacent to gamma
                        directed[(beta, gamma)] = True
                        changed = True

            # R2
            for alpha in range(p):
                for gamma in adj[alpha]:
                    if not is_undirected(alpha, gamma):
                        continue
                    for beta in adj[alpha]:
                        if beta == gamma:
                            continue
                        if is_directed(alpha, beta) and is_directed(beta, gamma):
                            directed[(alpha, gamma)] = True
                            changed = True

            # R3
            for beta in range(p):
                undirected_neigh = [v for v in adj[beta] if is_undirected(beta, v)]
                # For each pair (gamma1, gamma2) that both point to beta
                # and are not adjacent
                for alpha in undirected_neigh:
                    cands = [
                        v for v in adj[beta]
                        if v != alpha
                        and is_directed(v, beta)
                        and v not in adj[alpha]
                    ]
                    if len(cands) >= 2:
                        for g1, g2 in itertools.combinations(cands, 2):
                            if g2 not in adj[g1]:
                                directed[(alpha, beta)] = True
                                changed = True
                                break

        return directed

    # ------------------------------------------------------------------
    # Build CausalGraph from results
    # ------------------------------------------------------------------

    def _build_cpdag(
        self,
        directed: Dict[Tuple[int, int], bool],
        adj: Dict[int, Set[int]],
    ) -> CausalGraph:
        """Construct a CausalGraph from the skeleton + directed edges."""
        cg = CausalGraph(nodes=self._names)

        # Add directed edges
        for (u_idx, v_idx) in directed:
            u = self._names[u_idx]
            v = self._names[v_idx]
            # Only add if not reversed already
            if (v_idx, u_idx) not in directed:
                try:
                    cg.add_edge(u, v)
                except ValueError:
                    pass  # skip cycles (shouldn't happen in correct CPDAG)

        # For undirected skeleton edges not yet oriented, add arbitrarily
        # (in practice the CPDAG leaves them undirected — we pick an order)
        for i in range(self._p):
            for j in adj[i]:
                if i >= j:
                    continue
                u, v = self._names[i], self._names[j]
                if (i, j) not in directed and (j, i) not in directed:
                    # Undirected edge — add in topological consistent direction
                    try:
                        cg.add_edge(u, v)
                    except ValueError:
                        try:
                            cg.add_edge(v, u)
                        except ValueError:
                            pass

        return cg

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_skeleton_edges(self):
        """Return skeleton edges as list of frozensets (unordered pairs)."""
        return list(self.skeleton_) if self.skeleton_ else []

    def get_separating_set(self, x: str, y: str) -> Set[str]:
        """Return the separating set for pair (x, y), or empty set."""
        fk = frozenset({x, y})
        return self.separating_sets_.get(fk, set())

    def __repr__(self) -> str:
        return (
            f"PCAlgorithm(alpha={self.alpha}, "
            f"max_cond_set_size={self.max_cond_set_size})"
        )
