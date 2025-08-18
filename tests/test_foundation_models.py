"""
Comprehensive Test Suite for Foundation Models
============================================

Advanced testing framework for breakthrough foundation model algorithms
including multi-modal validation, performance benchmarking, and robustness testing.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import warnings
import time
from typing import Dict, List, Any, Optional
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.algorithms.foundation_causal import (
        FoundationCausalModel, 
        MetaLearningCausalDiscovery,
        MultiModalCausalConfig
    )
    from src.algorithms.self_supervised_causal import (
        SelfSupervisedCausalModel,
        SelfSupervisedCausalConfig
    )
    from src.utils.foundation_validation import (
        MultiModalDataValidator,
        FoundationModelValidator,
        MultiModalValidationConfig
    )
    from src.utils.foundation_monitoring import (
        FoundationModelMonitor,
        FoundationModelHealthChecker
    )
    from src.utils.foundation_optimization import (
        FoundationModelOptimizer,
        FoundationOptimizationConfig
    )
except ImportError as e:
    pytest.skip(f"Foundation model imports failed: {e}", allow_module_level=True)


class TestFoundationCausalModel:
    """Test suite for Foundation Causal Model."""
    
    @pytest.fixture
    def sample_multimodal_data(self):
        """Generate sample multi-modal data for testing."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_samples = 100
        
        # Tabular data
        tabular_data = np.random.randn(n_samples, 6)
        
        # Vision data (simulated features)
        vision_data = np.random.randn(n_samples, 768)
        
        # Text data (simulated embeddings)
        text_data = np.random.randn(n_samples, 768)
        
        # True causal structure
        true_adjacency = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])
        
        return {
            'tabular': tabular_data,
            'vision': vision_data,
            'text': text_data,
            'true_adjacency': true_adjacency
        }
    
    @pytest.fixture
    def foundation_config(self):
        """Create foundation model configuration for testing."""
        return MultiModalCausalConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            batch_size=16,
            max_epochs=5,  # Short for testing
            learning_rate=1e-3
        )
    
    def test_foundation_model_initialization(self, foundation_config):
        """Test foundation model initialization."""
        model = FoundationCausalModel(
            config=foundation_config,
            num_variables=6
        )
        
        assert model.config.hidden_dim == 64
        assert model.config.num_heads == 4
        assert model.num_variables == 6
        assert hasattr(model, 'multimodal_encoder')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'causal_structure_learner')
    
    def test_foundation_model_forward_pass(self, foundation_config, sample_multimodal_data):
        """Test foundation model forward pass."""
        model = FoundationCausalModel(
            config=foundation_config,
            num_variables=6
        )
        
        # Convert to tensors
        vision_tensor = torch.FloatTensor(sample_multimodal_data['vision'][:10])
        text_tensor = torch.FloatTensor(sample_multimodal_data['text'][:10])
        tabular_tensor = torch.FloatTensor(sample_multimodal_data['tabular'][:10])
        
        # Forward pass
        with torch.no_grad():
            outputs = model.forward(vision_tensor, text_tensor, tabular_tensor)
        
        assert 'adjacency_matrix' in outputs
        assert 'mechanisms' in outputs
        assert 'features' in outputs
        assert 'contrastive_features' in outputs
        
        # Check output shapes
        assert outputs['adjacency_matrix'].shape == (10, 6, 6)
        assert outputs['features'].shape == (10, 64)
    
    def test_foundation_model_training(self, foundation_config, sample_multimodal_data):
        """Test foundation model training process."""
        model = FoundationCausalModel(
            config=foundation_config,
            num_variables=6
        )
        
        # Test fitting
        tabular_data = sample_multimodal_data['tabular']
        vision_data = sample_multimodal_data['vision']
        text_data = sample_multimodal_data['text']
        
        # Should not raise exceptions
        fitted_model = model.fit(
            tabular_data, 
            vision_data=vision_data, 
            text_data=text_data
        )
        
        assert fitted_model.is_trained
        assert len(fitted_model.training_history) == foundation_config.max_epochs
    
    def test_foundation_model_discovery(self, foundation_config, sample_multimodal_data):
        """Test causal discovery with foundation model."""
        model = FoundationCausalModel(
            config=foundation_config,
            num_variables=6
        )
        
        # Fit and discover
        tabular_data = sample_multimodal_data['tabular']
        vision_data = sample_multimodal_data['vision']
        text_data = sample_multimodal_data['text']
        
        model.fit(tabular_data, vision_data=vision_data, text_data=text_data)
        result = model.discover(tabular_data, vision_data=vision_data, text_data=text_data)
        
        assert result.adjacency_matrix.shape == (6, 6)
        assert result.confidence_matrix.shape == (6, 6)
        assert result.metadata['method_used'] == 'Foundation Causal Model'
        assert result.metadata['foundation_model'] is True
        assert result.metadata['meta_learning_enabled'] is True
    
    def test_dag_constraint_loss(self, foundation_config):
        """Test DAG constraint loss computation."""
        model = FoundationCausalModel(
            config=foundation_config,
            num_variables=4
        )
        
        # Test with valid DAG (upper triangular)
        valid_dag = torch.tensor([[[0.0, 0.5, 0.3, 0.0],
                                  [0.0, 0.0, 0.4, 0.2],
                                  [0.0, 0.0, 0.0, 0.6],
                                  [0.0, 0.0, 0.0, 0.0]]])
        
        dag_loss = model.compute_dag_constraint_loss(valid_dag)
        assert isinstance(dag_loss, torch.Tensor)
        assert dag_loss.item() >= 0  # DAG loss should be non-negative
    
    def test_meta_learning_model(self, foundation_config):
        """Test meta-learning causal discovery model."""
        meta_config = foundation_config
        meta_config.meta_learning_rate = 1e-3
        meta_config.inner_steps = 2
        
        meta_model = MetaLearningCausalDiscovery(
            config=meta_config,
            num_variables=4
        )
        
        assert hasattr(meta_model, 'meta_learner')
        assert hasattr(meta_model, 'meta_training_tasks')


class TestSelfSupervisedCausalModel:
    """Test suite for Self-Supervised Causal Model."""
    
    @pytest.fixture
    def ssl_config(self):
        """Create self-supervised configuration for testing."""
        return SelfSupervisedCausalConfig(
            representation_dim=64,
            batch_size=16,
            max_epochs=5,  # Short for testing
            learning_rate=1e-3
        )
    
    @pytest.fixture
    def sample_tabular_data(self):
        """Generate sample tabular data."""
        np.random.seed(42)
        return np.random.randn(100, 5)
    
    def test_ssl_model_initialization(self, ssl_config):
        """Test self-supervised model initialization."""
        model = SelfSupervisedCausalModel(
            config=ssl_config,
            num_variables=5
        )
        
        assert model.config.representation_dim == 64
        assert model.num_variables == 5
        assert hasattr(model, 'augmentations')
        assert hasattr(model, 'invariance_loss_fn')
    
    def test_ssl_model_training(self, ssl_config, sample_tabular_data):
        """Test self-supervised model training."""
        model = SelfSupervisedCausalModel(
            config=ssl_config,
            num_variables=5
        )
        
        # Test fitting
        fitted_model = model.fit(sample_tabular_data)
        
        assert fitted_model.is_trained
        assert len(fitted_model.training_history) == ssl_config.max_epochs
        assert fitted_model.encoder is not None
        assert fitted_model.structure_predictor is not None
    
    def test_ssl_augmentations(self, ssl_config):
        """Test causal augmentations."""
        from src.algorithms.self_supervised_causal import CausalAugmentations
        
        augmentations = CausalAugmentations(ssl_config)
        
        # Test data
        data = torch.randn(20, 5)
        
        # Test noise addition
        noisy_data = augmentations.add_noise(data)
        assert noisy_data.shape == data.shape
        assert not torch.equal(noisy_data, data)
        
        # Test masking
        masked_data, mask = augmentations.random_masking(data)
        assert masked_data.shape == data.shape
        assert mask.shape == data.shape
        
        # Test intervention
        intervened_data = augmentations.simulate_intervention(data, variable_idx=0)
        assert intervened_data.shape == data.shape
        # First variable should be different
        assert not torch.equal(intervened_data[:, 0], data[:, 0])
        
        # Test positive pairs
        aug1, aug2 = augmentations.create_positive_pairs(data)
        assert aug1.shape == data.shape
        assert aug2.shape == data.shape


class TestMultiModalDataValidator:
    """Test suite for multi-modal data validation."""
    
    @pytest.fixture
    def validation_config(self):
        """Create validation configuration."""
        return MultiModalValidationConfig(
            min_samples=10,
            max_samples=1000,
            missing_data_threshold=0.2
        )
    
    @pytest.fixture
    def valid_multimodal_data(self):
        """Generate valid multi-modal data."""
        np.random.seed(42)
        return {
            'tabular': np.random.randn(50, 5),
            'vision': np.random.randn(50, 768),
            'text': np.random.randn(50, 768)
        }
    
    def test_validator_initialization(self, validation_config):
        """Test validator initialization."""
        validator = MultiModalDataValidator(validation_config)
        
        assert validator.config.min_samples == 10
        assert validator.config.max_samples == 1000
        assert hasattr(validator, 'base_validator')
        assert hasattr(validator, 'security_validator')
    
    def test_valid_data_validation(self, validation_config, valid_multimodal_data):
        """Test validation of valid multi-modal data."""
        validator = MultiModalDataValidator(validation_config)
        
        result = validator.validate_multimodal_data(
            valid_multimodal_data['tabular'],
            valid_multimodal_data['vision'],
            valid_multimodal_data['text']
        )
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'tabular' in result['statistics']
        assert 'vision' in result['statistics']
        assert 'text' in result['statistics']
        assert 'consistency' in result['statistics']
    
    def test_invalid_data_validation(self, validation_config):
        """Test validation of invalid data."""
        validator = MultiModalDataValidator(validation_config)
        
        # Too few samples
        small_data = np.random.randn(5, 3)
        result = validator.validate_multimodal_data(small_data)
        
        assert result['valid'] is False
        assert any('Too few samples' in error for error in result['errors'])
    
    def test_mismatched_samples_validation(self, validation_config):
        """Test validation with mismatched sample counts."""
        validator = MultiModalDataValidator(validation_config)
        
        tabular_data = np.random.randn(50, 5)
        vision_data = np.random.randn(40, 768)  # Different sample count
        
        result = validator.validate_multimodal_data(tabular_data, vision_data)
        
        assert result['valid'] is False
        assert any('Sample mismatch' in error for error in result['errors'])


class TestFoundationModelValidator:
    """Test suite for foundation model validation."""
    
    def test_model_architecture_validation(self):
        """Test model architecture validation."""
        validator = FoundationModelValidator()
        
        # Create simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
        
        result = validator.validate_model_architecture(model)
        
        assert result['valid'] is True
        assert 'total_parameters' in result
        assert 'trainable_parameters' in result
        assert 'estimated_memory_gb' in result
        assert result['total_parameters'] > 0
    
    def test_training_stability_validation(self):
        """Test training stability validation."""
        validator = FoundationModelValidator()
        
        # Stable training (decreasing loss)
        stable_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.31, 0.30, 0.30]
        result = validator.validate_training_stability(stable_losses)
        
        assert result['valid'] is True
        assert 'coefficient_of_variation' in result
        assert 'loss_reduction' in result
        assert result['loss_reduction'] > 0  # Loss should decrease
        
        # Unstable training (fluctuating loss)
        unstable_losses = [1.0, 0.5, 1.2, 0.3, 1.5, 0.2, 1.8, 0.1, 2.0, 0.05]
        result = validator.validate_training_stability(unstable_losses)
        
        assert len(result['warnings']) > 0  # Should have warnings


class TestFoundationModelMonitor:
    """Test suite for foundation model monitoring."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = FoundationModelMonitor(
            model_name="test_foundation_model",
            metrics_window_size=50
        )
        
        assert monitor.model_name == "test_foundation_model"
        assert monitor.metrics_window_size == 50
        assert hasattr(monitor, 'metrics_history')
        assert hasattr(monitor, 'alert_thresholds')
    
    def test_inference_monitoring_context(self):
        """Test inference monitoring context manager."""
        monitor = FoundationModelMonitor()
        
        with monitor.start_inference_monitoring() as context:
            assert context is not None
            context.set_batch_size(32)
            context.set_input_dimensions({'tabular': [32, 10], 'vision': [32, 768]})
            
            # Simulate processing time
            time.sleep(0.01)
        
        # Check that metrics were recorded
        assert len(monitor.metrics_history) == 1
        recorded_metrics = monitor.metrics_history[0]
        assert recorded_metrics.batch_size == 32
        assert recorded_metrics.inference_time > 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = FoundationModelMonitor()
        
        # Initially no data
        summary = monitor.get_performance_summary()
        assert summary['status'] == 'no_data'
        
        # Add some mock metrics
        from src.utils.foundation_monitoring import FoundationModelMetrics
        
        for i in range(5):
            metrics = FoundationModelMetrics(
                inference_time=1.0 + i * 0.1,
                memory_usage_gb=2.0,
                causal_discovery_accuracy=0.8 + i * 0.02
            )
            monitor.record_metrics(metrics)
        
        summary = monitor.get_performance_summary()
        assert summary['measurements_count'] == 5
        assert 'recent_performance' in summary
        assert 'avg_inference_time' in summary['recent_performance']


class TestFoundationModelOptimizer:
    """Test suite for foundation model optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = FoundationOptimizationConfig(
            enable_pruning=True,
            enable_quantization=True,
            memory_efficient_attention=True
        )
        
        optimizer = FoundationModelOptimizer(config)
        
        assert optimizer.config.enable_pruning is True
        assert optimizer.config.enable_quantization is True
        assert hasattr(optimizer, 'pruner')
        assert hasattr(optimizer, 'adaptive_optimizer')
        assert hasattr(optimizer, 'distributed_processor')
    
    def test_adaptive_inference_optimization(self):
        """Test adaptive inference optimization."""
        config = FoundationOptimizationConfig(adaptive_batch_size=True)
        optimizer = FoundationModelOptimizer(config)
        
        # Test input analysis
        input_data = {
            'tabular': torch.randn(32, 10),
            'vision': torch.randn(32, 768),
            'text': torch.randn(32, 768)
        }
        
        optimizations = optimizer.adaptive_optimizer.optimize_for_input(None, input_data)
        
        assert 'batch_size' in optimizations
        assert 'attention_mechanism' in optimizations
        assert 'memory_strategy' in optimizations
        assert isinstance(optimizations['batch_size'], int)


class TestPerformanceBenchmarks:
    """Performance benchmark tests for foundation models."""
    
    @pytest.mark.slow
    def test_foundation_model_performance(self):
        """Benchmark foundation model performance."""
        config = MultiModalCausalConfig(
            hidden_dim=128,
            num_heads=8,
            num_layers=4,
            batch_size=16,
            max_epochs=10
        )
        
        model = FoundationCausalModel(config=config, num_variables=6)
        
        # Generate benchmark data
        n_samples = 200
        tabular_data = np.random.randn(n_samples, 6)
        vision_data = np.random.randn(n_samples, 768)
        text_data = np.random.randn(n_samples, 768)
        
        # Benchmark training time
        start_time = time.time()
        model.fit(tabular_data, vision_data=vision_data, text_data=text_data)
        training_time = time.time() - start_time
        
        # Benchmark inference time
        start_time = time.time()
        result = model.discover(tabular_data, vision_data=vision_data, text_data=text_data)
        inference_time = time.time() - start_time
        
        # Performance assertions
        assert training_time < 60  # Should train in under 1 minute
        assert inference_time < 5   # Should infer in under 5 seconds
        assert result.adjacency_matrix.shape == (6, 6)
        
        print(f"Foundation Model Performance:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Memory usage: ~{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "CPU mode")


class TestIntegrationTests:
    """Integration tests for complete foundation model pipeline."""
    
    def test_end_to_end_multimodal_pipeline(self):
        """Test complete end-to-end multi-modal causal discovery."""
        # Data generation
        np.random.seed(42)
        n_samples = 80
        
        # Create realistic multi-modal data
        tabular_data = np.random.randn(n_samples, 4)
        vision_data = np.random.randn(n_samples, 512)
        text_data = np.random.randn(n_samples, 512)
        
        # Configuration
        config = MultiModalCausalConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            batch_size=16,
            max_epochs=3
        )
        
        # Model pipeline
        model = FoundationCausalModel(config=config, num_variables=4)
        
        # Validation
        validator = MultiModalDataValidator()
        validation_result = validator.validate_multimodal_data(
            tabular_data, vision_data, text_data
        )
        assert validation_result['valid'] is True
        
        # Training and discovery
        model.fit(tabular_data, vision_data=vision_data, text_data=text_data)
        result = model.discover(tabular_data, vision_data=vision_data, text_data=text_data)
        
        # Verification
        assert result.adjacency_matrix.shape == (4, 4)
        assert result.metadata['foundation_model'] is True
        assert result.metadata['multimodal_fusion'] in ['cross_attention', 'concat', 'gated_fusion']
        
    def test_optimization_pipeline(self):
        """Test model optimization pipeline."""
        # Create model
        config = MultiModalCausalConfig(hidden_dim=64, num_heads=4, num_layers=2)
        model = FoundationCausalModel(config=config, num_variables=4)
        
        # Optimization config
        opt_config = FoundationOptimizationConfig(
            enable_pruning=True,
            enable_quantization=False,  # Disable for testing
            memory_efficient_attention=True
        )
        
        # Optimizer
        optimizer = FoundationModelOptimizer(opt_config)
        
        # Apply optimizations
        optimized_model = optimizer.optimize_model(model)
        
        # Verify optimization was applied
        assert optimized_model is not None
        
        # Test optimized inference
        input_data = {
            'tabular': torch.randn(16, 4),
            'vision': torch.randn(16, 512),
            'text': torch.randn(16, 512)
        }
        
        opt_result = optimizer.optimize_inference(optimized_model, input_data)
        assert 'optimized_model' in opt_result
        assert 'optimizations_applied' in opt_result


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])