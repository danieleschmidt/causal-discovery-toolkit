"""Evaluation metrics for causal discovery."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import precision_score, recall_score, f1_score


class CausalMetrics:
    """Metrics for evaluating causal discovery performance."""
    
    @staticmethod
    def structural_hamming_distance(true_adj: np.ndarray, 
                                  pred_adj: np.ndarray) -> int:
        """Compute Structural Hamming Distance between adjacency matrices.
        
        Args:
            true_adj: True adjacency matrix
            pred_adj: Predicted adjacency matrix
            
        Returns:
            Structural Hamming Distance (number of edge differences)
        """
        if true_adj.shape != pred_adj.shape:
            raise ValueError("Adjacency matrices must have same shape")
            
        return np.sum(true_adj != pred_adj)
    
    @staticmethod
    def precision_recall_f1(true_adj: np.ndarray, 
                           pred_adj: np.ndarray) -> Dict[str, float]:
        """Compute precision, recall, and F1 score for edge prediction.
        
        Args:
            true_adj: True adjacency matrix  
            pred_adj: Predicted adjacency matrix
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if true_adj.shape != pred_adj.shape:
            raise ValueError("Adjacency matrices must have same shape")
            
        # Flatten matrices and treat as binary classification
        true_flat = true_adj.flatten()
        pred_flat = pred_adj.flatten()
        
        # Handle case where no edges exist
        if np.sum(true_flat) == 0:
            precision = 1.0 if np.sum(pred_flat) == 0 else 0.0
            recall = 1.0
            f1 = 1.0 if precision == 1.0 else 0.0
        else:
            precision = precision_score(true_flat, pred_flat, zero_division=0)
            recall = recall_score(true_flat, pred_flat, zero_division=0)
            f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def true_positive_rate(true_adj: np.ndarray, 
                          pred_adj: np.ndarray) -> float:
        """Compute True Positive Rate (sensitivity) for edge detection.
        
        Args:
            true_adj: True adjacency matrix
            pred_adj: Predicted adjacency matrix
            
        Returns:
            True Positive Rate
        """
        true_edges = np.sum(true_adj)
        if true_edges == 0:
            return 1.0  # No edges to detect
            
        true_positives = np.sum((true_adj == 1) & (pred_adj == 1))
        return true_positives / true_edges
    
    @staticmethod 
    def false_positive_rate(true_adj: np.ndarray,
                           pred_adj: np.ndarray) -> float:
        """Compute False Positive Rate for edge detection.
        
        Args:
            true_adj: True adjacency matrix
            pred_adj: Predicted adjacency matrix
            
        Returns:
            False Positive Rate
        """
        true_non_edges = np.sum(true_adj == 0)
        if true_non_edges == 0:
            return 0.0  # No non-edges
            
        false_positives = np.sum((true_adj == 0) & (pred_adj == 1))
        return false_positives / true_non_edges
    
    @staticmethod
    def evaluate_discovery(true_adj: np.ndarray,
                          pred_adj: np.ndarray,
                          confidence_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of causal discovery results.
        
        Args:
            true_adj: True adjacency matrix
            pred_adj: Predicted adjacency matrix  
            confidence_scores: Optional confidence scores for predictions
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Basic metrics
        shd = CausalMetrics.structural_hamming_distance(true_adj, pred_adj)
        prf_metrics = CausalMetrics.precision_recall_f1(true_adj, pred_adj)
        tpr = CausalMetrics.true_positive_rate(true_adj, pred_adj)
        fpr = CausalMetrics.false_positive_rate(true_adj, pred_adj)
        
        # Edge statistics
        n_true_edges = np.sum(true_adj)
        n_pred_edges = np.sum(pred_adj)
        n_total_possible = true_adj.size - np.trace(np.ones_like(true_adj))
        
        results = {
            'structural_hamming_distance': shd,
            'precision': prf_metrics['precision'],
            'recall': prf_metrics['recall'],
            'f1_score': prf_metrics['f1_score'],
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'n_true_edges': n_true_edges,
            'n_predicted_edges': n_pred_edges,
            'n_total_possible_edges': n_total_possible,
            'edge_density_true': n_true_edges / n_total_possible,
            'edge_density_pred': n_pred_edges / n_total_possible
        }
        
        # Add confidence-based metrics if available
        if confidence_scores is not None:
            results['mean_confidence'] = np.mean(confidence_scores)
            results['std_confidence'] = np.std(confidence_scores)
            
            # Average confidence for true/false positives
            true_pos_mask = (true_adj == 1) & (pred_adj == 1)
            false_pos_mask = (true_adj == 0) & (pred_adj == 1)
            
            if np.any(true_pos_mask):
                results['confidence_true_positives'] = np.mean(confidence_scores[true_pos_mask])
            if np.any(false_pos_mask):
                results['confidence_false_positives'] = np.mean(confidence_scores[false_pos_mask])
        
        return results