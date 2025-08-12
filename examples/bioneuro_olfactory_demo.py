#!/usr/bin/env python3
"""
Demonstration of bioneuro-olfactory fusion research capabilities.

This example showcases the specialized algorithms and data processing
for olfactory neural signal analysis and causal discovery.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import specialized bioneuro modules
from algorithms.bioneuro_olfactory import (
    OlfactoryNeuralCausalModel, 
    MultiModalOlfactoryCausalModel,
    OlfactoryNeuralSignal
)
from utils.bioneuro_data_processing import (
    BioneuroDataProcessor,
    OlfactoryFeatureExtractor, 
    OlfactoryDataProcessingConfig
)
from utils.logging_config import get_logger
from utils.validation import DataValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def generate_synthetic_olfactory_data(n_samples: int = 1000, 
                                    n_receptors: int = 10,
                                    n_neurons: int = 8,
                                    noise_level: float = 0.1,
                                    random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic olfactory neural data for demonstration.
    
    Args:
        n_samples: Number of temporal samples
        n_receptors: Number of olfactory receptors
        n_neurons: Number of neural units
        noise_level: Amount of noise to add
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic olfactory neural data
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_receptors} receptors, {n_neurons} neurons")
    
    # Time vector
    dt = 0.001  # 1ms resolution
    time = np.arange(n_samples) * dt
    
    # Generate odor stimulus (multiple odor components)
    n_odors = 3
    odor_concentrations = np.zeros((n_samples, n_odors))
    
    # Odor 1: Step stimulus
    odor_concentrations[200:600, 0] = 1.0
    
    # Odor 2: Gaussian pulse
    odor_concentrations[:, 1] = np.exp(-((time - 0.4)**2) / (2 * 0.05**2))
    
    # Odor 3: Oscillatory stimulus  
    odor_concentrations[:, 2] = 0.5 * (1 + np.sin(2 * np.pi * 10 * time)) * (time > 0.3) * (time < 0.8)
    
    # Generate receptor responses (with selectivity patterns)
    receptor_responses = np.zeros((n_samples, n_receptors))
    receptor_selectivity = np.random.uniform(0.1, 1.0, (n_receptors, n_odors))
    
    for i in range(n_receptors):
        for j in range(n_odors):
            # Convolve odor with receptor response kernel
            response_kernel = np.exp(-time / 0.05)  # 50ms decay
            receptor_response = np.convolve(odor_concentrations[:, j], response_kernel, mode='same')
            receptor_responses[:, i] += receptor_selectivity[i, j] * receptor_response
    
    # Add receptor adaptation
    for i in range(n_receptors):
        adaptation_tau = np.random.uniform(0.1, 0.3)  # 100-300ms adaptation
        adaptation = np.exp(-time / adaptation_tau)
        receptor_responses[:, i] *= adaptation
    
    # Generate neural firing rates (driven by receptor responses)
    neural_firing_rates = np.zeros((n_samples, n_neurons))
    neural_connectivity = np.random.uniform(0.0, 2.0, (n_neurons, n_receptors))
    
    for i in range(n_neurons):
        # Weighted sum of receptor inputs
        neural_input = np.sum(neural_connectivity[i, :].reshape(-1, 1) * receptor_responses.T, axis=0)
        
        # Add nonlinearity (sigmoid activation)
        neural_firing_rates[:, i] = 50 * (1 / (1 + np.exp(-neural_input + 0.5)))  # Max 50 Hz
        
        # Add temporal dynamics (integration)
        if i % 2 == 0:  # Every other neuron has different dynamics
            neural_firing_rates[:, i] = np.convolve(
                neural_firing_rates[:, i], 
                np.exp(-time[:100] / 0.02), 
                mode='same'
            )
    
    # Add behavioral response (simplified)
    behavioral_response = np.zeros(n_samples)
    # Behavioral response is delayed integration of neural activity
    neural_sum = np.sum(neural_firing_rates, axis=1)
    for t in range(50, n_samples):  # 50ms delay
        behavioral_response[t] = np.mean(neural_sum[t-50:t])
    behavioral_response = behavioral_response / np.max(behavioral_response)  # Normalize
    
    # Add noise to all signals
    receptor_responses += noise_level * np.random.randn(*receptor_responses.shape)
    neural_firing_rates += noise_level * np.random.randn(*neural_firing_rates.shape)
    behavioral_response += noise_level * np.random.randn(n_samples)
    
    # Create DataFrame
    data = pd.DataFrame()
    
    # Add time
    data['time'] = time
    
    # Add odor concentrations
    for i in range(n_odors):
        data[f'odor_concentration_{i}'] = odor_concentrations[:, i]
    
    # Add receptor responses
    for i in range(n_receptors):
        data[f'receptor_response_{i}'] = receptor_responses[:, i]
    
    # Add neural firing rates
    for i in range(n_neurons):
        data[f'neural_firing_rate_{i}'] = neural_firing_rates[:, i]
    
    # Add behavioral response
    data['behavioral_response'] = behavioral_response
    
    logger.info(f"Generated synthetic data with shape: {data.shape}")
    return data


def demonstrate_bioneuro_processing():
    """Demonstrate specialized bioneuro data processing."""
    print("\\n" + "="*80)
    print("BIONEURO-OLFACTORY DATA PROCESSING DEMONSTRATION")
    print("="*80)
    
    # Generate synthetic data
    data = generate_synthetic_olfactory_data(n_samples=500, n_receptors=6, n_neurons=4)
    print(f"\\n1. Generated synthetic olfactory data: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Configure data processor
    config = OlfactoryDataProcessingConfig(
        sampling_rate_hz=1000.0,
        filter_low_cutoff=1.0,
        filter_high_cutoff=100.0,
        normalization_method="z_score",
        artifact_removal=True,
        baseline_correction=True
    )
    
    # Initialize processor
    processor = BioneuroDataProcessor(config)
    
    # Process olfactory signals
    print("\\n2. Processing olfactory signals...")
    processed_data = processor.process_olfactory_signals(data)
    print(f"   Processed data shape: {processed_data.shape}")
    
    # Process neural activity
    print("\\n3. Processing neural activity...")
    neural_processed = processor.process_neural_activity(processed_data)
    print(f"   Neural processed shape: {neural_processed.shape}")
    
    # Process odor stimuli
    print("\\n4. Processing odor stimuli...")
    final_processed = processor.process_odor_stimuli(neural_processed)
    print(f"   Final processed shape: {final_processed.shape}")
    
    # Feature extraction
    print("\\n5. Extracting olfactory features...")
    feature_extractor = OlfactoryFeatureExtractor(sampling_rate=1000.0)
    features = feature_extractor.extract_olfactory_features(final_processed)
    print(f"   Feature data shape: {features.shape}")
    
    # Show some extracted features
    feature_cols = [col for col in features.columns if any(term in col for term in 
                   ['magnitude', 'latency', 'adaptation', 'entropy', 'synchronization'])]
    print(f"   Extracted features: {feature_cols[:10]}...")  # Show first 10
    
    return final_processed, features


def demonstrate_olfactory_causal_discovery():
    """Demonstrate olfactory-specific causal discovery."""
    print("\\n" + "="*80)
    print("OLFACTORY NEURAL CAUSAL DISCOVERY DEMONSTRATION") 
    print("="*80)
    
    # Generate data
    data = generate_synthetic_olfactory_data(n_samples=300, n_receptors=4, n_neurons=3)
    
    # Initialize olfactory causal model
    print("\\n1. Initializing OlfactoryNeuralCausalModel...")
    model = OlfactoryNeuralCausalModel(
        receptor_sensitivity_threshold=0.15,
        neural_firing_threshold=10.0,
        temporal_window_ms=50,
        cross_modal_integration=True,
        bootstrap_samples=100,
        confidence_level=0.95
    )
    
    # Fit and discover
    print("\\n2. Fitting model and discovering causal relationships...")
    result = model.fit_discover(data)
    
    # Display results
    print(f"\\n3. Causal Discovery Results:")
    print(f"   Method used: {result.method_used}")
    print(f"   Adjacency matrix shape: {result.adjacency_matrix.shape}")
    print(f"   Number of causal edges: {result.metadata['n_causal_edges']}")
    print(f"   Network density: {result.metadata.get('n_causal_edges', 0) / result.adjacency_matrix.size:.3f}")
    
    # Neural pathway analysis
    print(f"\\n4. Neural Pathway Analysis:")
    for pathway, strength in result.neural_pathways.items():
        print(f"   {pathway}: {strength:.3f}")
    
    # Sensory integration analysis
    print(f"\\n5. Sensory Integration Analysis:")
    for metric, value in result.sensory_integration_map.items():
        print(f"   {metric}: {value:.3f}")
    
    # Statistical significance
    print(f"\\n6. Statistical Significance:")
    for test, p_value in result.statistical_significance.items():
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"   {test}: p={p_value:.3f} ({significance})")
    
    # Confidence intervals
    print(f"\\n7. Confidence Intervals (95%):")
    for metric, (lower, upper) in result.confidence_intervals.items():
        print(f"   {metric}: [{lower:.3f}, {upper:.3f}]")
    
    return result


def demonstrate_multimodal_analysis():
    """Demonstrate multi-modal olfactory causal analysis."""
    print("\\n" + "="*80)
    print("MULTI-MODAL OLFACTORY CAUSAL ANALYSIS")
    print("="*80)
    
    # Generate data with behavioral component
    data = generate_synthetic_olfactory_data(n_samples=250, n_receptors=3, n_neurons=2)
    
    # Initialize multi-modal model
    print("\\n1. Initializing MultiModalOlfactoryCausalModel...")
    mm_model = MultiModalOlfactoryCausalModel(
        receptor_sensitivity_threshold=0.1,
        neural_firing_threshold=5.0,
        behavioral_threshold=0.15,
        multi_modal_fusion="late_fusion",
        attention_mechanism=True,
        cross_modal_integration=True
    )
    
    # Perform multi-modal discovery
    print("\\n2. Performing multi-modal causal discovery...")
    mm_result = mm_model.fit_discover(data)
    
    # Display enhanced results
    print(f"\\n3. Multi-Modal Results:")
    print(f"   Causal edges discovered: {mm_result.metadata['n_causal_edges']}")
    print(f"   Cross-modal integration: {mm_result.metadata['cross_modal_enabled']}")
    
    if 'behavioral_mean' in mm_result.metadata:
        print(f"   Behavioral response mean: {mm_result.metadata['behavioral_mean']:.3f}")
        print(f"   Behavioral significant: {mm_result.metadata['behavioral_significant']}")
    
    print(f"\\n4. Olfactory Correlations:")
    print(f"   Correlation matrix shape: {mm_result.olfactory_correlations.shape}")
    if mm_result.olfactory_correlations.size > 1:
        mean_corr = np.mean(mm_result.olfactory_correlations[np.triu_indices_from(mm_result.olfactory_correlations, k=1)])
        print(f"   Mean correlation: {mean_corr:.3f}")
    
    return mm_result


def create_visualization(data: pd.DataFrame, result, save_path: str = None):
    """Create visualization of olfactory causal discovery results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bioneuro-Olfactory Causal Discovery Results', fontsize=16)
        
        # Plot 1: Raw signals
        ax1 = axes[0, 0]
        receptor_cols = [col for col in data.columns if 'receptor' in col]
        for i, col in enumerate(receptor_cols[:3]):  # Plot first 3 receptors
            ax1.plot(data['time'][:200], data[col][:200], label=f'Receptor {i+1}')
        ax1.set_title('Olfactory Receptor Responses')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Response Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Neural firing
        ax2 = axes[0, 1]
        neural_cols = [col for col in data.columns if 'neural' in col]
        for i, col in enumerate(neural_cols[:3]):  # Plot first 3 neurons
            ax2.plot(data['time'][:200], data[col][:200], label=f'Neuron {i+1}')
        ax2.set_title('Neural Firing Rates')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Firing Rate (Hz)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Causal adjacency matrix
        ax3 = axes[1, 0]
        im = ax3.imshow(result.adjacency_matrix, cmap='Blues', aspect='auto')
        ax3.set_title('Causal Adjacency Matrix')
        ax3.set_xlabel('Target Variable')
        ax3.set_ylabel('Source Variable')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Neural pathways
        ax4 = axes[1, 1]
        pathways = list(result.neural_pathways.keys())
        strengths = list(result.neural_pathways.values())
        bars = ax4.bar(range(len(pathways)), strengths, color='skyblue')
        ax4.set_title('Neural Pathway Strengths')
        ax4.set_xlabel('Pathway Type')
        ax4.set_ylabel('Strength')
        ax4.set_xticks(range(len(pathways)))
        ax4.set_xticklabels([p.replace('_', '\\n') for p in pathways], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, strength in zip(bars, strengths):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{strength:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\\n   Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Visualization failed: {str(e)}")
        print("   (Visualization skipped - matplotlib may not be available)")


def run_comprehensive_demo():
    """Run comprehensive bioneuro-olfactory demonstration."""
    print("\\n" + "="*80)
    print("COMPREHENSIVE BIONEURO-OLFACTORY FUSION DEMONSTRATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Data processing demonstration
        processed_data, features = demonstrate_bioneuro_processing()
        
        # Basic causal discovery
        causal_result = demonstrate_olfactory_causal_discovery()
        
        # Multi-modal analysis
        mm_result = demonstrate_multimodal_analysis()
        
        # Create visualization
        print("\\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        # Use the processed data and causal result for visualization
        data_for_viz = generate_synthetic_olfactory_data(n_samples=200, n_receptors=3, n_neurons=2)
        create_visualization(data_for_viz, causal_result, 'bioneuro_results.png')
        
        # Summary report
        print("\\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        
        print(f"\\n✅ Data Processing: Successfully processed {processed_data.shape[0]} samples")
        print(f"✅ Feature Extraction: Extracted {features.shape[1]} features")
        print(f"✅ Causal Discovery: Found {causal_result.metadata['n_causal_edges']} causal relationships")
        print(f"✅ Multi-modal Analysis: Integrated {len(mm_result.neural_pathways)} neural pathways")
        
        # Research insights
        print(f"\\n🧠 Research Insights:")
        print(f"   • Receptor-neural pathway strength: {causal_result.neural_pathways.get('receptor_to_neural', 0):.3f}")
        print(f"   • Cross-modal coherence: {causal_result.sensory_integration_map.get('cross_modal_coherence', 0):.3f}")
        print(f"   • Temporal synchrony: {causal_result.sensory_integration_map.get('temporal_synchrony', 0):.3f}")
        
        # Statistical validation
        statistical_power = sum(1 for p in causal_result.statistical_significance.values() if p < 0.05)
        print(f"   • Statistical tests significant: {statistical_power}/{len(causal_result.statistical_significance)}")
        
        print(f"\\n🎯 Research Applications:")
        print(f"   • Olfactory receptor characterization")
        print(f"   • Neural encoding mechanisms")
        print(f"   • Cross-modal sensory integration")
        print(f"   • Biomarker discovery for neurological conditions")
        print(f"   • Brain-computer interface development")
        
        print(f"\\n📊 Next Steps:")
        print(f"   • Apply to real experimental data")
        print(f"   • Validate with ground-truth causal networks")
        print(f"   • Extend to other sensory modalities")
        print(f"   • Develop real-time analysis capabilities")
        
        return {
            'processed_data': processed_data,
            'features': features,
            'causal_result': causal_result,
            'multimodal_result': mm_result
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\\n❌ Error in demonstration: {str(e)}")
        return None


if __name__ == "__main__":
    print("Starting Bioneuro-Olfactory Fusion Research Demonstration...")
    
    # Run the comprehensive demonstration
    results = run_comprehensive_demo()
    
    if results:
        print("\\n✅ Demonstration completed successfully!")
        print("\\nTo run this demo:")
        print("  python examples/bioneuro_olfactory_demo.py")
    else:
        print("\\n❌ Demonstration failed. Check logs for details.")
    
    print("\\n" + "="*80)