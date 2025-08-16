#!/usr/bin/env python3
"""Demonstration of next-generation causal discovery algorithms."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

# Import next-generation algorithms
from src.algorithms.quantum_causal import QuantumCausalDiscovery, QuantumEntanglementCausal
from src.algorithms.neuromorphic_causal import SpikingNeuralCausal, ReservoirComputingCausal
from src.algorithms.topological_causal import PersistentHomologyCausal, AlgebraicTopologyCausal


def generate_complex_causal_data(n_samples: int = 500, noise_level: float = 0.1) -> pd.DataFrame:
    """Generate complex synthetic data with non-linear causal relationships."""
    np.random.seed(42)
    
    # Base variables
    X1 = np.random.normal(0, 1, n_samples)
    
    # Non-linear causal relationships
    X2 = 0.5 * X1**2 + 0.3 * np.sin(X1) + noise_level * np.random.normal(0, 1, n_samples)
    X3 = 0.4 * X1 * X2 + 0.2 * np.cos(X2) + noise_level * np.random.normal(0, 1, n_samples)
    
    # Quantum-like entangled variables
    X4 = 0.3 * (X1 + X2) + 0.2 * np.exp(-0.1 * X3**2) + noise_level * np.random.normal(0, 1, n_samples)
    
    # Oscillatory dynamics
    t = np.linspace(0, 10, n_samples)
    X5 = 0.2 * X1 + 0.3 * np.sin(2 * np.pi * 0.1 * t + X2) + noise_level * np.random.normal(0, 1, n_samples)
    
    # Topologically complex relationship
    X6 = 0.1 * (X1**3 - 3*X1*X2**2) + 0.2 * X4 + noise_level * np.random.normal(0, 1, n_samples)
    
    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6
    })
    
    return data


def demonstrate_quantum_causal_discovery():
    """Demonstrate quantum-inspired causal discovery algorithms."""
    print("🌌 QUANTUM CAUSAL DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Generate test data
    data = generate_complex_causal_data(n_samples=300)
    print(f"Generated dataset with {len(data)} samples and {len(data.columns)} variables")
    
    # Quantum Causal Discovery
    print("\n🔬 Testing QuantumCausalDiscovery...")
    quantum_model = QuantumCausalDiscovery(
        n_qubits=6,
        coherence_threshold=0.6,
        entanglement_threshold=0.4,
        measurement_basis='computational'
    )
    
    start_time = time.time()
    quantum_model.fit(data)
    quantum_result = quantum_model.predict(data)
    quantum_time = time.time() - start_time
    
    print(f"⏱️  Quantum discovery completed in {quantum_time:.3f} seconds")
    print(f"🧮 Quantum coherence: {quantum_result.metadata['quantum_coherence']:.4f}")
    print(f"🔗 Quantum entanglement: {quantum_result.metadata['quantum_entanglement']:.4f}")
    print(f"📊 Discovered {np.sum(quantum_result.adjacency_matrix)} causal connections")
    
    # Quantum Entanglement Causal
    print("\n🌀 Testing QuantumEntanglementCausal...")
    entanglement_model = QuantumEntanglementCausal(
        bell_state_threshold=0.7,
        epr_correlation_strength=0.8,
        quantum_channel_noise=0.15
    )
    
    start_time = time.time()
    entanglement_model.fit(data)
    entanglement_result = entanglement_model.predict(data)
    entanglement_time = time.time() - start_time
    
    print(f"⏱️  Entanglement discovery completed in {entanglement_time:.3f} seconds")
    print(f"🔔 Bell states used: {entanglement_result.metadata['bell_states_used']}")
    print(f"📊 Discovered {np.sum(entanglement_result.adjacency_matrix)} causal connections")
    
    return {
        'quantum': quantum_result,
        'entanglement': entanglement_result,
        'timing': {'quantum': quantum_time, 'entanglement': entanglement_time}
    }


def demonstrate_neuromorphic_causal_discovery():
    """Demonstrate neuromorphic causal discovery algorithms."""
    print("\n🧠 NEUROMORPHIC CAUSAL DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Generate test data
    data = generate_complex_causal_data(n_samples=200)  # Smaller for neuromorphic simulation
    print(f"Generated dataset with {len(data)} samples and {len(data.columns)} variables")
    
    # Spiking Neural Causal
    print("\n🔋 Testing SpikingNeuralCausal...")
    spiking_model = SpikingNeuralCausal(
        membrane_time_constant=15.0,
        synaptic_time_constant=5.0,
        threshold_voltage=-50.0,
        plasticity_rate=0.02,
        stdp_window=25.0
    )
    
    start_time = time.time()
    spiking_model.fit(data)
    spiking_result = spiking_model.predict(data)
    spiking_time = time.time() - start_time
    
    print(f"⏱️  Spiking neural discovery completed in {spiking_time:.3f} seconds")
    print(f"🧠 Average firing rate: {spiking_result.metadata['average_firing_rate_hz']:.2f} Hz")
    print(f"🔗 Average synaptic weight: {spiking_result.metadata['average_synaptic_weight']:.4f}")
    print(f"📊 Discovered {np.sum(spiking_result.adjacency_matrix)} causal connections")
    
    # Reservoir Computing Causal
    print("\n🌊 Testing ReservoirComputingCausal...")
    reservoir_model = ReservoirComputingCausal(
        reservoir_size=80,
        spectral_radius=0.95,
        input_scaling=0.15,
        leak_rate=0.2
    )
    
    start_time = time.time()
    reservoir_model.fit(data)
    reservoir_result = reservoir_model.predict(data)
    reservoir_time = time.time() - start_time
    
    print(f"⏱️  Reservoir computing discovery completed in {reservoir_time:.3f} seconds")
    print(f"🌊 Reservoir activity: {reservoir_result.metadata['reservoir_activity']:.4f}")
    print(f"🎯 State diversity: {reservoir_result.metadata['state_diversity']:.4f}")
    print(f"📊 Discovered {np.sum(reservoir_result.adjacency_matrix)} causal connections")
    
    return {
        'spiking': spiking_result,
        'reservoir': reservoir_result,
        'timing': {'spiking': spiking_time, 'reservoir': reservoir_time}
    }


def demonstrate_topological_causal_discovery():
    """Demonstrate topological causal discovery algorithms."""
    print("\n🔺 TOPOLOGICAL CAUSAL DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Generate test data
    data = generate_complex_causal_data(n_samples=150)  # Smaller for topology computation
    print(f"Generated dataset with {len(data)} samples and {len(data.columns)} variables")
    
    # Persistent Homology Causal
    print("\n🕸️  Testing PersistentHomologyCausal...")
    persistent_model = PersistentHomologyCausal(
        max_dimension=2,
        lifetime_threshold=0.15,
        density_threshold=0.25,
        resolution=40
    )
    
    start_time = time.time()
    persistent_model.fit(data)
    persistent_result = persistent_model.predict(data)
    persistent_time = time.time() - start_time
    
    print(f"⏱️  Persistent homology discovery completed in {persistent_time:.3f} seconds")
    print(f"🔢 Global Betti numbers: {persistent_result.metadata['global_betti_numbers']}")
    print(f"🕳️  Persistent features: {persistent_result.metadata['n_persistent_features']}")
    print(f"📊 Discovered {np.sum(persistent_result.adjacency_matrix)} causal connections")
    
    # Algebraic Topology Causal
    print("\n🎭 Testing AlgebraicTopologyCausal...")
    algebraic_model = AlgebraicTopologyCausal(
        sheaf_dimension=2,
        cohomology_degree=1,
        fiber_bundle_rank=3,
        connection_strength=0.25
    )
    
    start_time = time.time()
    algebraic_model.fit(data)
    algebraic_result = algebraic_model.predict(data)
    algebraic_time = time.time() - start_time
    
    print(f"⏱️  Algebraic topology discovery completed in {algebraic_time:.3f} seconds")
    print(f"🎯 H⁰ dimension: {algebraic_result.metadata['global_cohomology']['H0_dimension']}")
    print(f"🔄 H¹ dimension: {algebraic_result.metadata['global_cohomology']['H1_dimension']}")
    print(f"📊 Discovered {np.sum(algebraic_result.adjacency_matrix)} causal connections")
    
    return {
        'persistent': persistent_result,
        'algebraic': algebraic_result,
        'timing': {'persistent': persistent_time, 'algebraic': algebraic_time}
    }


def visualize_results(quantum_results, neuromorphic_results, topological_results):
    """Visualize and compare results from different algorithm families."""
    print("\n📊 COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    # Collect all results
    all_results = {
        'Quantum Discovery': quantum_results['quantum'],
        'Quantum Entanglement': quantum_results['entanglement'],
        'Spiking Neural': neuromorphic_results['spiking'],
        'Reservoir Computing': neuromorphic_results['reservoir'],
        'Persistent Homology': topological_results['persistent'],
        'Algebraic Topology': topological_results['algebraic']
    }
    
    # Performance comparison
    print("\n🏆 ALGORITHM PERFORMANCE COMPARISON")
    print("-" * 50)
    
    for name, result in all_results.items():
        n_connections = np.sum(result.adjacency_matrix)
        avg_confidence = np.mean(result.confidence_scores[result.adjacency_matrix > 0]) if n_connections > 0 else 0
        
        print(f"{name:20s}: {n_connections:2d} connections, confidence: {avg_confidence:.3f}")
    
    # Timing comparison
    print("\n⏱️  EXECUTION TIME COMPARISON")
    print("-" * 40)
    
    all_timings = {**quantum_results['timing'], **neuromorphic_results['timing'], **topological_results['timing']}
    for method, time_taken in all_timings.items():
        print(f"{method:20s}: {time_taken:.3f} seconds")
    
    # Algorithm-specific insights
    print("\n🔬 ALGORITHM-SPECIFIC INSIGHTS")
    print("-" * 40)
    
    print(f"🌌 Quantum coherence achieved: {quantum_results['quantum'].metadata['quantum_coherence']:.3f}")
    print(f"🔗 Quantum entanglement strength: {quantum_results['quantum'].metadata['quantum_entanglement']:.3f}")
    print(f"🧠 Neuromorphic firing rate: {neuromorphic_results['spiking'].metadata['average_firing_rate_hz']:.1f} Hz")
    print(f"🌊 Reservoir state diversity: {neuromorphic_results['reservoir'].metadata['state_diversity']:.3f}")
    print(f"🔺 Topological Betti₀: {topological_results['persistent'].metadata['global_betti_numbers'][0]}")
    print(f"🎭 Sheaf cohomology H¹: {topological_results['algebraic'].metadata['global_cohomology']['H1_dimension']}")


def demonstrate_convergence_analysis():
    """Demonstrate convergence properties of next-gen algorithms."""
    print("\n📈 CONVERGENCE & SCALABILITY ANALYSIS")
    print("=" * 60)
    
    sample_sizes = [50, 100, 200, 300, 500]
    algorithm_performance = {}
    
    for n_samples in sample_sizes:
        print(f"\n🔍 Testing with {n_samples} samples...")
        
        data = generate_complex_causal_data(n_samples=n_samples, noise_level=0.1)
        
        # Test quantum algorithm
        quantum_model = QuantumCausalDiscovery(coherence_threshold=0.5)
        start_time = time.time()
        quantum_model.fit(data)
        quantum_result = quantum_model.predict(data)
        quantum_time = time.time() - start_time
        
        # Test neuromorphic algorithm  
        reservoir_model = ReservoirComputingCausal(reservoir_size=50)
        start_time = time.time()
        reservoir_model.fit(data)
        reservoir_result = reservoir_model.predict(data)
        reservoir_time = time.time() - start_time
        
        algorithm_performance[n_samples] = {
            'quantum_connections': np.sum(quantum_result.adjacency_matrix),
            'quantum_time': quantum_time,
            'quantum_coherence': quantum_result.metadata['quantum_coherence'],
            'reservoir_connections': np.sum(reservoir_result.adjacency_matrix),
            'reservoir_time': reservoir_time,
            'reservoir_diversity': reservoir_result.metadata['state_diversity']
        }
    
    # Display convergence results
    print("\n📊 SCALABILITY RESULTS")
    print("-" * 50)
    print("Samples | Quantum Conn | Q-Time | Reservoir Conn | R-Time")
    print("-" * 50)
    
    for n_samples, perf in algorithm_performance.items():
        print(f"{n_samples:7d} | {perf['quantum_connections']:11d} | "
              f"{perf['quantum_time']:6.3f} | {perf['reservoir_connections']:13d} | "
              f"{perf['reservoir_time']:6.3f}")


def main():
    """Main demonstration function."""
    print("🚀 NEXT-GENERATION CAUSAL DISCOVERY ALGORITHMS")
    print("=" * 70)
    print("Demonstrating quantum, neuromorphic, and topological approaches")
    print("=" * 70)
    
    try:
        # Run demonstrations
        quantum_results = demonstrate_quantum_causal_discovery()
        neuromorphic_results = demonstrate_neuromorphic_causal_discovery()
        topological_results = demonstrate_topological_causal_discovery()
        
        # Visualize and compare
        visualize_results(quantum_results, neuromorphic_results, topological_results)
        
        # Convergence analysis
        demonstrate_convergence_analysis()
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("🎯 Next-generation algorithms show promising capabilities for:")
        print("   • Non-linear causal relationships")
        print("   • High-dimensional data analysis") 
        print("   • Novel physical insights")
        print("   • Complex system dynamics")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())