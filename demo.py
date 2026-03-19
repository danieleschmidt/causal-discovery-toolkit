#!/usr/bin/env python3
"""
demo.py — Causal Discovery Toolkit: Pearl do-calculus over knowledge graphs.

This demo uses a small legal-entity causal graph (5 nodes) representing:

    Lobbying → Legislation → Penalty
                    ↘
    Violation → Enforcement → Penalty
                    ↑
                Lobbying (also influences enforcement directly)

Questions answered:
    1. Did "Lobbying" cause the "Penalty"?
    2. What is the causal effect of "Violation" on "Penalty"?
    3. What would happen to "Penalty" if we intervened and forced Enforcement=high?

Also runs PC algorithm skeleton discovery on synthetic data drawn from the
true causal model.
"""

import sys
import numpy as np
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(__file__))

from causal import CausalGraph, PCAlgorithm, DoCalculusReasoner, KGCausalBridge

np.random.seed(42)


# ============================================================
# 1. Build a causal graph for a legal-entity scenario
# ============================================================

print("=" * 60)
print("STEP 1: Build the Causal Graph")
print("=" * 60)

# Nodes represent variables in a legal regulatory scenario
# Lobbying  → Legislation  → Penalty
# Lobbying  → Enforcement  → Penalty
# Violation → Enforcement

g = CausalGraph(
    nodes=["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"],
    edges=[
        ("Lobbying",   "Legislation"),
        ("Lobbying",   "Enforcement"),
        ("Legislation","Penalty"),
        ("Violation",  "Enforcement"),
        ("Enforcement","Penalty"),
    ],
)

print(g)
print(f"\nIs valid DAG: {g.is_dag()}")
print(f"Topological order: {g.topological_order()}")

print("\nStructural queries:")
print(f"  Parents of Penalty:      {g.parents('Penalty')}")
print(f"  Ancestors of Penalty:    {g.ancestors('Penalty')}")
print(f"  Descendants of Lobbying: {g.descendants('Lobbying')}")
print(f"  Markov blanket(Enforcement): {g.markov_blanket('Enforcement')}")


# ============================================================
# 2. Do-calculus: apply the do-operator
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Do-Calculus Reasoner")
print("=" * 60)

reasoner = DoCalculusReasoner(g)

# Query 1: Does Lobbying cause Penalty?
print("\nQuery 1: Does Lobbying → Penalty (structurally)?")
q1 = reasoner.causal_effect_query("Lobbying", "Penalty")
print(q1["explanation"])

# Query 2: Does Violation cause Penalty?
print("\nQuery 2: Does Violation → Penalty?")
q2 = reasoner.causal_effect_query("Violation", "Penalty")
print(q2["explanation"])

# Query 3: Rule checks
print("\nDo-calculus rule checks:")
print("  Rule 1 — Can we ignore Legislation when computing P(Penalty | do(Lobbying), Legislation)?")
r1 = reasoner.check_rule1(
    y={"Penalty"}, z={"Legislation"},
    x_do={"Lobbying"}, w=set()
)
print(f"  P(Penalty|do(Lobbying), Legislation) = P(Penalty|do(Lobbying))? {r1}")

print("  Rule 2 — Demote do(Violation) to obs(Violation)?")
r2 = reasoner.check_rule2(
    y={"Penalty"}, z={"Violation"},
    x_do=set(), w=set()
)
print(f"  P(Penalty|do(Violation)) = P(Penalty|Violation)? {r2}")

print("  Rule 3 — Drop do(Legislation) from P(Penalty|do(Lobbying), do(Legislation))?")
r3 = reasoner.check_rule3(
    y={"Penalty"}, z={"Legislation"},
    x_do={"Lobbying"}, w=set()
)
print(f"  P(Penalty|do(Lobbying),do(Legislation)) = P(Penalty|do(Lobbying))? {r3}")

# Identification
print("\nIdentification of P(Penalty | do(Lobbying)):")
id_result = reasoner.identify(y={"Penalty"}, x_do={"Lobbying"})
print(f"  Identifiable: {id_result['identifiable']}")
print(f"  Method:       {id_result['method']}")
print(f"  Formula:      {id_result['formula']}")


# ============================================================
# 3. Generate synthetic data from the causal model
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Generate Synthetic Data & PC Algorithm")
print("=" * 60)

N = 500
# Linear Gaussian structural equation model:
#   Lobbying   ~ N(0,1)
#   Legislation = 0.6*Lobbying + noise
#   Violation  ~ N(0,1)  (exogenous)
#   Enforcement = 0.5*Lobbying + 0.7*Violation + noise
#   Penalty     = 0.4*Legislation + 0.8*Enforcement + noise

lob = np.random.normal(0, 1, N)
leg = 0.6 * lob + np.random.normal(0, 0.5, N)
vio = np.random.normal(0, 1, N)
enf = 0.5 * lob + 0.7 * vio + np.random.normal(0, 0.5, N)
pen = 0.4 * leg + 0.8 * enf + np.random.normal(0, 0.3, N)

var_names = ["Lobbying", "Legislation", "Violation", "Enforcement", "Penalty"]
data = np.column_stack([lob, leg, vio, enf, pen])

print(f"Data shape: {data.shape}")
print(f"Variable order: {var_names}")

# Run PC algorithm
print("\nRunning PC Algorithm (alpha=0.01)...")
pc = PCAlgorithm(alpha=0.01, max_cond_set_size=3)
pc.fit(data, var_names)

print(f"\nSkeleton edges discovered by PC:")
for edge in sorted(pc.get_skeleton_edges(), key=lambda e: str(e)):
    u, v = list(edge)
    print(f"  {u} — {v}")

print(f"\nSeparating sets (selected):")
for pair, sep in list(pc.separating_sets_.items())[:5]:
    print(f"  sep({', '.join(sorted(pair))}) = {sep or '∅'}")

print(f"\nDiscovered CPDAG (PC algorithm output):")
print(pc.causal_graph_)


# ============================================================
# 4. Data-driven interventional estimation
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Interventional Mean Estimation")
print("=" * 60)

g.fit_data(data, var_names)

obs_mean = g.observational_mean("Penalty")
print(f"E[Penalty] (observational) = {obs_mean:.4f}")

# E[Penalty | do(Enforcement = 2.0)]
int_mean_high = g.interventional_mean("Penalty", {"Enforcement": 2.0})
int_mean_low  = g.interventional_mean("Penalty", {"Enforcement": -2.0})
print(f"E[Penalty | do(Enforcement=+2)] = {int_mean_high:.4f}")
print(f"E[Penalty | do(Enforcement=-2)] = {int_mean_low:.4f}")
print(f"ATE (high vs low Enforcement)   = {int_mean_high - int_mean_low:.4f}")


# ============================================================
# 5. KGCausalBridge: DocGraph integration
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: KGCausalBridge (DocGraph Integration)")
print("=" * 60)

# Simulate a DocGraph knowledge graph edge list
kg_triples = [
    ("CompanyA",    "resulted_in",   "Filing"),
    ("Filing",      "triggered",     "Investigation"),
    ("Investigation","resulted_in",  "Penalty"),
    ("Regulator",   "imposed_on",    "CompanyA"),
    ("CompanyA",    "caused_by_inverse", "Lobbying"),  # non-standard
    ("Lobbying",    "influences",    "Regulation"),
    ("Regulation",  "affects",       "Penalty"),
    ("Violation",   "leads_to",      "Investigation"),
]

bridge = KGCausalBridge()
bridge.build_from_triples(kg_triples, entity_metadata={
    "CompanyA":      {"type": "organization", "sector": "finance"},
    "Penalty":       {"type": "outcome", "severity": "high"},
    "Regulator":     {"type": "agency"},
    "Investigation": {"type": "process"},
})

print(bridge.summary())

# Query 1: Does CompanyA cause Penalty?
print("\nKG Query 1: Does CompanyA → Penalty?")
r = bridge.does_cause("CompanyA", "Penalty")
print(f"  Has causal path: {r['has_causal_path']}")
print(f"  Identifiable:    {r['identifiable']}")
print(f"  Method:          {r['method']}")
print(f"  Supporting KG edges:")
for e in r["supporting_kg_edges"]:
    print(f"    {e['from']} --[{e['predicate']}]--> {e['to']}")

# Query 2: Path analysis
print("\nKG Path Analysis: Filing → Penalty")
pa = bridge.path_analysis("Filing", "Penalty")
print(f"  Directed paths: {pa['directed_paths']}")
print(f"  Mediators: {pa['mediators']}")

# Query 3: Intervention
print("\nKG Intervention: do(Investigation = 1)")
inv_result = bridge.intervention_query(
    interventions={"Investigation": 1.0},
    outcome="Penalty",
)
print(f"  Edges removed by intervention: {inv_result['edges_removed']}")

print("\n" + "=" * 60)
print("Demo complete. All causal queries answered successfully.")
print("=" * 60)
