# causal-discovery-toolkit

**Pearl do-calculus and causal discovery for knowledge graphs.**

DocGraph's causal inference module — given a knowledge graph built from documents, reason causally about entity relationships: *"Did Company A's action cause Outcome B?"*

---

## What This Is

This toolkit implements the mathematical foundations of Pearl's causal inference framework:

- **Structural Causal Models** via directed acyclic graphs (DAGs)
- **Do-calculus** — Pearl's three rules for interventional reasoning
- **PC Algorithm** — data-driven causal skeleton discovery
- **DocGraph integration** — KG edge list → causal queries

All in pure Python with `networkx`, `numpy`, and `scipy`. No heavy ML dependencies.

---

## Quick Start

```python
from causal import CausalGraph, DoCalculusReasoner, KGCausalBridge

# 1. Build a causal graph
g = CausalGraph(
    nodes=["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"],
    edges=[
        ("Lobbying",    "Legislation"),
        ("Lobbying",    "Enforcement"),
        ("Legislation", "Penalty"),
        ("Violation",   "Enforcement"),
        ("Enforcement", "Penalty"),
    ],
)

# 2. Apply the do-operator (intervention)
mutilated = g.do({"Enforcement": 2.0})  # Forces Enforcement = 2, cuts its parents

# 3. Answer causal queries
reasoner = DoCalculusReasoner(g)
result = reasoner.causal_effect_query("Violation", "Penalty")
print(result["explanation"])
# → Directed causal paths: Violation → Enforcement → Penalty
# → Identifiable via: direct (no confounding)

# 4. Integrate DocGraph KG
bridge = KGCausalBridge()
bridge.build_from_triples([
    ("CompanyA", "resulted_in",  "Filing"),
    ("Filing",   "triggered",    "Investigation"),
    ("Investigation", "resulted_in", "Penalty"),
    ("Violation", "leads_to",    "Investigation"),
])
print(bridge.does_cause("CompanyA", "Penalty")["has_causal_path"])  # True
```

Run the full demo:
```bash
python demo.py
```

---

## Core Modules

### `CausalGraph`

Wraps a NetworkX DiGraph with causal inference operations.

```python
g = CausalGraph(nodes=[...], edges=[...])

# Structural queries
g.parents("Y")           # direct parents
g.ancestors("Y")         # all upstream nodes
g.descendants("X")       # all downstream nodes
g.markov_blanket("Y")    # MB = parents ∪ children ∪ co-parents

# d-separation (independence oracle)
g.d_separated({"X"}, {"Y"}, {"Z"})  # is X ⊥ Y | Z?

# Do-operator (Pearl's intervention calculus)
g_mutilated = g.do({"X": 1.0})  # returns new graph with X's parents cut

# Identification
g.satisfies_backdoor("X", "Y", adjustment_set)
g.satisfies_frontdoor("X", "Y", mediator_set)

# Data-driven estimation
g.fit_data(data_array, variable_names)
g.observational_mean("Y")
g.interventional_mean("Y", {"X": 1.0})   # back-door adjusted
```

### `DoCalculusReasoner`

Implements Pearl's three rules of do-calculus.

**Rule 1 — Insertion/deletion of observations:**
```
P(y | do(x), z, w) = P(y | do(x), w)
iff Y ⊥ Z | X,W in G_{X̄}
```

**Rule 2 — Action/observation exchange:**
```
P(y | do(x), do(z), w) = P(y | do(x), z, w)
iff Y ⊥ Z | X,W in G_{X̄, Z̄}   (Z's outgoing cut)
```

**Rule 3 — Insertion/deletion of actions:**
```
P(y | do(x), do(z), w) = P(y | do(x), w)
iff Y ⊥ Z | X,W in G_{X̄, Z(W)}   (Z(W) = Z not-ancestors-of-W)
```

```python
r = DoCalculusReasoner(g)

# Rule checks
r.check_rule1(y={"Y"}, z={"Z"}, x_do={"X"}, w=set())
r.check_rule2(y={"Y"}, z={"Z"}, x_do={"X"}, w=set())
r.check_rule3(y={"Y"}, z={"Z"}, x_do={"X"}, w=set())

# Identification (back-door, front-door, or do-calculus)
r.identify(y={"Penalty"}, x_do={"Lobbying"})

# High-level causal query
r.causal_effect_query("Lobbying", "Penalty")
# → {has_causal_path, identifiable, formula, method, explanation}

# Counterfactual reasoning
r.counterfactual_query(
    outcome="Penalty",
    factual_vals={"Lobbying": 1.0},
    counterfactual_intervention={"Enforcement": 3.0},
    data=data, variable_names=names
)
```

### `PCAlgorithm`

Data-driven causal skeleton discovery (Spirtes, Glymour, Scheines 1993).

```python
pc = PCAlgorithm(alpha=0.05, max_cond_set_size=3)
pc.fit(data_array, variable_names)

pc.skeleton_           # set of frozenset edges
pc.separating_sets_    # {frozenset({X,Y}): sep_set}
pc.causal_graph_       # CausalGraph (CPDAG)

pc.get_skeleton_edges()
pc.get_separating_set("X", "Y")
```

Uses Fisher's z-transform conditional independence test (partial correlation).  
Handles v-structure orientation (colliders) and Meek's R1–R3 propagation rules.

### `KGCausalBridge`

Connects DocGraph knowledge graphs to causal inference.

```python
bridge = KGCausalBridge()

# From KG triples (subject, predicate, object)
bridge.build_from_triples([
    ("CompanyA", "resulted_in",  "Penalty"),
    ("Penalty",  "caused_by",    "Violation"),   # → reversed: Violation → Penalty
])

# From edge list
bridge.build_from_edge_list([("A", "B"), ("B", "C")])

# Refine with data (uses PC algorithm)
bridge.refine_with_data(data, variable_names)

# Queries
bridge.does_cause("A", "C")         # causal path + identification
bridge.causal_effect("A", "C")      # E[C | do(A=1)] - E[C | do(A=0)]
bridge.intervention_query({"A": 1.0}, outcome="C")
bridge.path_analysis("A", "C")      # enumerate directed paths + mediators
bridge.get_causal_subgraph(["A","B","C"])
```

**Predicate handling:**
- `resulted_in`, `leads_to`, `influences`, `causes`, … → forward edge (source → target)
- `caused_by`, `resulted_from`, `preceded_by`, … → reverse edge (target → source)
- `is_a`, `same_as`, `related_to`, … → dropped (non-causal)

---

## DocGraph Integration

DocGraph builds entity-relationship KGs from legal and scientific documents.  
This toolkit is designed to consume DocGraph's output and answer causal questions:

```python
# DocGraph produces: list of (entity, relation, entity) triples
from docgraph import DocGraph  # hypothetical
from causal import KGCausalBridge

dg = DocGraph(corpus="sec_filings/")
kg_edges = dg.extract_relations()  # → [(subj, pred, obj), ...]

bridge = KGCausalBridge()
bridge.build_from_triples(kg_edges)

# Answer causal queries
print(bridge.does_cause("CompanyA", "StockDrop")["explanation"])
print(bridge.path_analysis("Filing", "Penalty"))

# If you have panel data alongside the KG:
bridge.refine_with_data(panel_data, variable_names)
ate = bridge.causal_effect("LobbyingSpend", "RegulatoryPenalty")["ate_estimate"]
```

---

## Installation

```bash
# Dependencies (all available in Anaconda)
pip install networkx numpy scipy

# Run tests
python -m pytest tests/test_causal_graph.py tests/test_reasoner.py -v

# Run demo
python demo.py
```

---

## Tests

```
tests/test_causal_graph.py   — CausalGraph: d-sep, do-operator, back-door, data estimation
tests/test_reasoner.py       — DoCalculusReasoner (3 rules), PCAlgorithm, KGCausalBridge
```

All 71 tests pass.

---

## Theory

**d-separation** (Pearl, 1988): A graphical criterion for reading off conditional independencies from a DAG.  Used as the oracle for all rule checks.

**Do-calculus** (Pearl, 1995): Three inference rules that are *sound and complete* for identifying causal effects from observational data, given a causal DAG.

**PC algorithm** (Spirtes et al., 1993): Recovers the Markov equivalence class (CPDAG) of the true DAG from i.i.d. data under causal Markov + faithfulness + causal sufficiency.

**Back-door criterion** (Pearl, 2009): If Z blocks all back-door paths from X to Y and contains no descendants of X, then P(Y|do(X)) = Σ_z P(Y|X,z)P(z).

### References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*, 82(4), 669–688.
- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
- Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional DAGs with the PC-algorithm. *JMLR*, 8, 613–636.

---

## Repository Structure

```
causal/
  __init__.py         — public API
  graph.py            — CausalGraph (DAG + do-calculus)
  pc_algorithm.py     — PCAlgorithm (skeleton discovery)
  do_calculus.py      — DoCalculusReasoner (3 rules)
  kg_bridge.py        — KGCausalBridge (DocGraph integration)
tests/
  test_causal_graph.py
  test_reasoner.py
demo.py               — end-to-end example
```
