"""
Enhanced Validation for Foundation Models
========================================

Comprehensive validation, error handling, and robustness testing
specifically designed for breakthrough foundation model algorithms.

Features:
- Multi-modal data validation
- Model architecture verification
- Training stability monitoring
- Performance degradation detection
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

try:
    from .validation import DataValidator
    from .error_handling import safe_execution, robust_execution
    from .security import SecurityValidator
except ImportError:
    from validation import DataValidator
    from error_handling import safe_execution, robust_execution
    from security import SecurityValidator


@dataclass
class MultiModalValidationConfig:
    """Configuration for multi-modal validation."""
    min_samples: int = 50
    max_samples: int = 100000
    min_features: int = 1
    max_features: int = 10000
    vision_dim_range: Tuple[int, int] = (512, 2048)
    text_dim_range: Tuple[int, int] = (256, 1024)
    tabular_dim_range: Tuple[int, int] = (2, 100)
    missing_data_threshold: float = 0.3
    outlier_threshold: float = 3.0
    correlation_threshold: float = 0.95


class MultiModalDataValidator:
    """Validator for multi-modal causal discovery data."""
    
    def __init__(self, config: Optional[MultiModalValidationConfig] = None):
        self.config = config or MultiModalValidationConfig()
        self.base_validator = DataValidator()
        self.security_validator = SecurityValidator()
        self.logger = logging.getLogger(__name__)
        
    def validate_multimodal_data(self, 
                                tabular_data: Union[np.ndarray, pd.DataFrame],
                                vision_data: Optional[np.ndarray] = None,
                                text_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive multi-modal data validation."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Validate tabular data
            tabular_result = self._validate_tabular_data(tabular_data)
            validation_results['statistics']['tabular'] = tabular_result
            
            if not tabular_result['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(tabular_result['errors'])
            
            # Validate vision data if provided
            if vision_data is not None:
                vision_result = self._validate_vision_data(vision_data, tabular_data)
                validation_results['statistics']['vision'] = vision_result
                
                if not vision_result['valid']:
                    validation_results['warnings'].extend(vision_result['warnings'])
            
            # Validate text data if provided  
            if text_data is not None:
                text_result = self._validate_text_data(text_data, tabular_data)
                validation_results['statistics']['text'] = text_result
                
                if not text_result['valid']:
                    validation_results['warnings'].extend(text_result['warnings'])
            
            # Cross-modal consistency checks
            consistency_result = self._validate_cross_modal_consistency(
                tabular_data, vision_data, text_data
            )
            validation_results['statistics']['consistency'] = consistency_result
            
            # Security validation
            security_result = self._validate_security(tabular_data, vision_data, text_data)
            validation_results['statistics']['security'] = security_result
            
            if not security_result['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(security_result['errors'])
            
            # Generate recommendations
            self._generate_recommendations(validation_results)
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            self.logger.error(f"Multi-modal validation error: {e}")
        
        return validation_results
    
    def _validate_tabular_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Validate tabular data component."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Convert to numpy for analysis
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data
            
            n_samples, n_features = data_array.shape
            
            # Sample size validation
            if n_samples < self.config.min_samples:
                result['errors'].append(f"Too few samples: {n_samples} < {self.config.min_samples}")
                result['valid'] = False
            elif n_samples > self.config.max_samples:
                result['warnings'].append(f"Large dataset: {n_samples} samples may slow training")
            
            # Feature count validation
            if n_features < self.config.tabular_dim_range[0]:
                result['errors'].append(f"Too few features: {n_features} < {self.config.tabular_dim_range[0]}")
                result['valid'] = False
            elif n_features > self.config.tabular_dim_range[1]:
                result['warnings'].append(f"High dimensionality: {n_features} features")
            
            # Missing data check
            missing_ratio = np.isnan(data_array).sum() / data_array.size
            if missing_ratio > self.config.missing_data_threshold:
                result['errors'].append(f"Too much missing data: {missing_ratio:.1%}")
                result['valid'] = False
            elif missing_ratio > 0.1:
                result['warnings'].append(f"Moderate missing data: {missing_ratio:.1%}")
            
            # Outlier detection
            z_scores = np.abs((data_array - np.nanmean(data_array, axis=0)) / np.nanstd(data_array, axis=0))
            outlier_ratio = np.sum(z_scores > self.config.outlier_threshold) / data_array.size
            if outlier_ratio > 0.05:
                result['warnings'].append(f"High outlier ratio: {outlier_ratio:.1%}")
            
            # Multicollinearity check
            if n_features > 1:
                corr_matrix = np.corrcoef(data_array, rowvar=False)
                high_corr = np.sum(np.abs(corr_matrix) > self.config.correlation_threshold) - n_features
                if high_corr > 0:
                    result['warnings'].append(f"High correlation detected: {high_corr} pairs")
            
            result['shape'] = data_array.shape
            result['missing_ratio'] = missing_ratio
            result['outlier_ratio'] = outlier_ratio
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Tabular validation failed: {str(e)}")
        
        return result
    
    def _validate_vision_data(self, vision_data: np.ndarray, 
                             tabular_data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Validate vision feature data."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if len(vision_data.shape) != 2:
                result['errors'].append(f"Vision data must be 2D, got {len(vision_data.shape)}D")
                result['valid'] = False
                return result
            
            n_samples, vision_dim = vision_data.shape
            tabular_samples = len(tabular_data)
            
            # Sample consistency
            if n_samples != tabular_samples:
                result['errors'].append(f"Sample mismatch: vision {n_samples} vs tabular {tabular_samples}")
                result['valid'] = False
            
            # Dimension validation
            if not (self.config.vision_dim_range[0] <= vision_dim <= self.config.vision_dim_range[1]):
                result['warnings'].append(f"Unusual vision dimension: {vision_dim}")
            
            # Feature quality checks
            zero_features = np.sum(np.all(vision_data == 0, axis=0))
            if zero_features > vision_dim * 0.1:
                result['warnings'].append(f"Many zero features: {zero_features}/{vision_dim}")
            
            # Magnitude checks
            mean_magnitude = np.mean(np.abs(vision_data))
            if mean_magnitude < 1e-6:
                result['warnings'].append("Vision features have very small magnitude")
            elif mean_magnitude > 100:
                result['warnings'].append("Vision features have very large magnitude")
            
            result['shape'] = vision_data.shape
            result['mean_magnitude'] = mean_magnitude
            result['zero_features'] = zero_features
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Vision validation failed: {str(e)}")
        
        return result
    
    def _validate_text_data(self, text_data: np.ndarray,
                           tabular_data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Validate text embedding data."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if len(text_data.shape) != 2:
                result['errors'].append(f"Text data must be 2D, got {len(text_data.shape)}D")
                result['valid'] = False
                return result
            
            n_samples, text_dim = text_data.shape
            tabular_samples = len(tabular_data)
            
            # Sample consistency
            if n_samples != tabular_samples:
                result['errors'].append(f"Sample mismatch: text {n_samples} vs tabular {tabular_samples}")
                result['valid'] = False
            
            # Dimension validation
            if not (self.config.text_dim_range[0] <= text_dim <= self.config.text_dim_range[1]):
                result['warnings'].append(f"Unusual text dimension: {text_dim}")
            
            # Embedding quality checks
            norm_check = np.linalg.norm(text_data, axis=1)
            zero_norm_count = np.sum(norm_check < 1e-8)
            if zero_norm_count > 0:
                result['warnings'].append(f"Zero-norm embeddings: {zero_norm_count}")
            
            # Check for normalized embeddings
            mean_norm = np.mean(norm_check)
            if 0.9 <= mean_norm <= 1.1:
                result['info'] = "Text embeddings appear normalized"
            elif mean_norm < 0.1:
                result['warnings'].append("Text embeddings have very small norms")
            
            result['shape'] = text_data.shape
            result['mean_norm'] = mean_norm
            result['zero_norm_count'] = zero_norm_count
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Text validation failed: {str(e)}")
        
        return result
    
    def _validate_cross_modal_consistency(self, 
                                        tabular_data: Union[np.ndarray, pd.DataFrame],
                                        vision_data: Optional[np.ndarray] = None,
                                        text_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate consistency across modalities."""
        result = {'valid': True, 'warnings': []}
        
        try:
            if isinstance(tabular_data, pd.DataFrame):
                tabular_array = tabular_data.values
            else:
                tabular_array = tabular_data
            
            n_samples = len(tabular_array)
            
            # Cross-modal correlation analysis
            correlations = {}
            
            if vision_data is not None and text_data is not None:
                # Vision-Text correlation
                vision_mean = np.mean(vision_data, axis=1)
                text_mean = np.mean(text_data, axis=1)
                vt_corr = np.corrcoef(vision_mean, text_mean)[0, 1]
                correlations['vision_text'] = vt_corr
                
                if abs(vt_corr) > 0.9:
                    result['warnings'].append("Very high vision-text correlation - possible data leakage")
            
            if vision_data is not None:
                # Vision-Tabular correlation
                vision_mean = np.mean(vision_data, axis=1)
                tabular_mean = np.mean(tabular_array, axis=1)
                vt_corr = np.corrcoef(vision_mean, tabular_mean)[0, 1]
                correlations['vision_tabular'] = vt_corr
            
            if text_data is not None:
                # Text-Tabular correlation
                text_mean = np.mean(text_data, axis=1)
                tabular_mean = np.mean(tabular_array, axis=1)
                tt_corr = np.corrcoef(text_mean, tabular_mean)[0, 1]
                correlations['text_tabular'] = tt_corr
            
            result['correlations'] = correlations
            
        except Exception as e:
            result['warnings'].append(f"Cross-modal validation failed: {str(e)}")
        
        return result
    
    def _validate_security(self, 
                          tabular_data: Union[np.ndarray, pd.DataFrame],
                          vision_data: Optional[np.ndarray] = None,
                          text_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Security validation for multi-modal data."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Use existing security validator for tabular data
            tabular_security = self.security_validator.validate_data_security(tabular_data)
            
            if not tabular_security['is_safe']:
                result['valid'] = False
                result['errors'].extend(tabular_security['security_issues'])
            
            # Additional checks for multi-modal data
            total_memory = 0
            if isinstance(tabular_data, np.ndarray):
                total_memory += tabular_data.nbytes
            
            if vision_data is not None:
                total_memory += vision_data.nbytes
                
            if text_data is not None:
                total_memory += text_data.nbytes
            
            # Memory safety check (1GB limit)
            if total_memory > 1e9:
                result['warnings'].append(f"Large memory usage: {total_memory/1e9:.1f}GB")
            
            result['memory_usage'] = total_memory
            result['tabular_security'] = tabular_security
            
        except Exception as e:
            result['errors'].append(f"Security validation failed: {str(e)}")
            result['valid'] = False
        
        return result
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]):
        """Generate optimization recommendations based on validation results."""
        recommendations = []
        
        try:
            stats = validation_results['statistics']
            
            # Data quality recommendations
            if 'tabular' in stats:
                tabular_stats = stats['tabular']
                if tabular_stats.get('missing_ratio', 0) > 0.05:
                    recommendations.append("Consider data imputation for missing values")
                
                if tabular_stats.get('outlier_ratio', 0) > 0.02:
                    recommendations.append("Consider outlier detection and removal")
            
            # Multi-modal recommendations
            if 'vision' in stats and 'text' in stats:
                recommendations.append("Multi-modal fusion will benefit from attention mechanisms")
            
            if 'consistency' in stats:
                correlations = stats['consistency'].get('correlations', {})
                if any(abs(corr) > 0.8 for corr in correlations.values()):
                    recommendations.append("High cross-modal correlation - consider regularization")
            
            # Performance recommendations
            if 'vision' in stats:
                vision_dim = stats['vision'].get('shape', [0, 0])[1]
                if vision_dim > 1000:
                    recommendations.append("Consider dimensionality reduction for vision features")
            
            if 'text' in stats:
                text_dim = stats['text'].get('shape', [0, 0])[1]
                if text_dim > 800:
                    recommendations.append("Consider PCA for text embeddings")
            
            validation_results['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")


class FoundationModelValidator:
    """Validator for foundation model architectures and training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @safe_execution("model_architecture_validation")
    def validate_model_architecture(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Validate foundation model architecture."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            result['total_parameters'] = total_params
            result['trainable_parameters'] = trainable_params
            
            # Architecture checks
            if total_params > 1e9:  # > 1B parameters
                result['warnings'].append(f"Very large model: {total_params/1e6:.1f}M parameters")
            elif total_params < 1e6:  # < 1M parameters
                result['warnings'].append(f"Small model: {total_params/1e3:.1f}K parameters")
            
            # Check for gradient flow
            has_gradients = any(p.requires_grad for p in model.parameters())
            if not has_gradients:
                result['errors'].append("No trainable parameters found")
                result['valid'] = False
            
            # Memory estimation
            param_memory = total_params * 4 / 1e9  # 4 bytes per float32, in GB
            if param_memory > 8:  # > 8GB
                result['warnings'].append(f"High memory usage: {param_memory:.1f}GB")
            
            result['estimated_memory_gb'] = param_memory
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Architecture validation failed: {str(e)}")
            self.logger.error(f"Model architecture validation error: {e}")
        
        return result
    
    @robust_execution(max_retries=3)
    def validate_training_stability(self, loss_history: List[float]) -> Dict[str, Any]:
        """Validate training stability from loss history."""
        result = {'valid': True, 'warnings': []}
        
        try:
            if len(loss_history) < 10:
                result['warnings'].append("Short training history - stability unclear")
                return result
            
            losses = np.array(loss_history)
            
            # Check for convergence
            recent_losses = losses[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            cv = loss_std / (loss_mean + 1e-8)  # Coefficient of variation
            
            if cv > 0.1:
                result['warnings'].append(f"Training unstable: CV={cv:.3f}")
            
            # Check for divergence
            if len(losses) > 20:
                early_mean = np.mean(losses[:10])
                late_mean = np.mean(losses[-10:])
                
                if late_mean > early_mean * 2:
                    result['warnings'].append("Possible training divergence")
            
            # Check for plateau
            if len(losses) > 50:
                recent_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                if abs(recent_trend) < 1e-6:
                    result['warnings'].append("Training appears to have plateaued")
            
            result['coefficient_of_variation'] = cv
            result['final_loss'] = losses[-1]
            result['loss_reduction'] = (losses[0] - losses[-1]) / losses[0] if losses[0] != 0 else 0
            
        except Exception as e:
            result['warnings'].append(f"Training stability validation failed: {str(e)}")
            self.logger.error(f"Training stability validation error: {e}")
        
        return result


# Export classes
__all__ = [
    'MultiModalDataValidator',
    'FoundationModelValidator', 
    'MultiModalValidationConfig'
]