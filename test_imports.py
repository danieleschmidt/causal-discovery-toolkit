#!/usr/bin/env python3
"""Test script to verify causal discovery toolkit functionality."""

import sys
import os

# Add current directory to path for package import
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test direct imports from modules
    import causal_discovery_toolkit
    from causal_discovery_toolkit import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
    from causal_discovery_toolkit import DataProcessor, CausalMetrics
    
    print("✅ All core modules imported successfully")
    
    # Test basic functionality
    import pandas as pd
    import numpy as np
    
    # Create synthetic data
    np.random.seed(42)
    data = pd.DataFrame({
        'X1': np.random.randn(100),
        'X2': np.random.randn(100),
        'X3': np.random.randn(100)
    })
    
    # Add some causal relationships
    data['X2'] = 0.5 * data['X1'] + 0.3 * np.random.randn(100)
    data['X3'] = 0.7 * data['X2'] + 0.2 * np.random.randn(100)
    
    print("✅ Test data created")
    
    # Test DataProcessor
    processor = DataProcessor()
    cleaned_data = processor.clean_data(data)
    standardized_data = processor.standardize(cleaned_data)
    
    print("✅ DataProcessor working")
    
    # Test SimpleLinearCausalModel
    model = SimpleLinearCausalModel(threshold=0.3)
    model.fit(standardized_data)
    result = model.discover()
    
    print("✅ SimpleLinearCausalModel working")
    print(f"  Discovered {result.metadata['n_edges']} causal relationships")
    
    # Test metrics
    true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # X1->X2->X3
    pred_adj = result.adjacency_matrix
    
    metrics = CausalMetrics.evaluate_discovery(true_adj, pred_adj, result.confidence_scores)
    print("✅ CausalMetrics working")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    
    print("\n🎉 ALL TESTS PASSED - Generation 1 Complete!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    sys.exit(1)