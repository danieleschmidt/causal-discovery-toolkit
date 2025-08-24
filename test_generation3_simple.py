#!/usr/bin/env python3
"""Simple Generation 3 scalability test"""

import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
import time


def test_simple_scalability():
    """Test basic scalability functionality"""
    print("🔍 Testing basic scalability...")
    
    from algorithms.scalable_causal import ScalableCausalDiscoveryModel
    from utils.data_processing import DataProcessor
    
    processor = DataProcessor()
    data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
    
    model = ScalableCausalDiscoveryModel(
        enable_parallelization=True,
        enable_caching=True,
        enable_auto_scaling=False,
        max_workers=2,
        user_id="scalability_test"
    )
    
    # Test discovery
    result = model.fit_discover(data)
    
    print(f"✅ Scalable model works")
    print(f"   - Quality score: {result.quality_score:.3f}")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Method: {result.method_used}")
    print(f"   - Edges: {result.metadata['n_edges']}")
    
    # Get scalability report
    report = model.get_scalability_report()
    if report:
        print(f"   - Operations: {report.get('operations_count', 0)}")
        print(f"   - Cache enabled: {report.get('caching_enabled', False)}")
        print(f"   - Max workers: {report.get('max_workers', 0)}")
    
    return True


if __name__ == "__main__":
    print("⚡ GENERATION 3 SIMPLE SCALABILITY TEST")
    print("=" * 45)
    
    try:
        test_simple_scalability()
        print()
        print("🎉 GENERATION 3 BASIC SCALABILITY WORKS!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)