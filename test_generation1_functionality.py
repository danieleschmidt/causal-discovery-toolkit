#!/usr/bin/env python3
"""Generation 1 functionality test - Make it Work"""

import sys
import os
sys.path.append('src')

def test_basic_imports():
    """Test that core imports work"""
    print("🔍 Testing basic imports...")
    from algorithms.base import CausalDiscoveryModel, SimpleLinearCausalModel, CausalResult
    from utils.data_processing import DataProcessor
    print("✅ Basic imports successful")

def test_data_processing():
    """Test data processing functionality"""
    print("🔍 Testing data processing...")
    from utils.data_processing import DataProcessor
    import pandas as pd
    
    processor = DataProcessor()
    
    # Generate synthetic data
    data = processor.generate_synthetic_data(n_samples=100, n_variables=3, random_state=42)
    print(f"✅ Generated synthetic data: {data.shape}")
    
    # Clean data
    cleaned = processor.clean_data(data)
    print(f"✅ Cleaned data: {cleaned.shape}")
    
    # Standardize data
    standardized = processor.standardize(cleaned)
    print(f"✅ Standardized data: {standardized.shape}")
    
    # Validate data
    is_valid, issues = processor.validate_data(standardized)
    print(f"✅ Data validation: {is_valid}, issues: {len(issues)}")
    
    return data

def test_causal_discovery():
    """Test causal discovery functionality"""
    print("🔍 Testing causal discovery...")
    from algorithms.base import SimpleLinearCausalModel
    from utils.data_processing import DataProcessor
    
    # Generate test data
    processor = DataProcessor()
    data = processor.generate_synthetic_data(n_samples=200, n_variables=4, random_state=42)
    
    # Create and fit model
    model = SimpleLinearCausalModel(threshold=0.3)
    result = model.fit_discover(data)
    
    print(f"✅ Causal discovery completed")
    print(f"   - Adjacency matrix shape: {result.adjacency_matrix.shape}")
    print(f"   - Method used: {result.method_used}")
    print(f"   - Number of edges: {result.metadata['n_edges']}")
    print(f"   - Variables: {result.metadata['variable_names']}")
    
    return result

def test_end_to_end():
    """Test complete end-to-end workflow"""
    print("🔍 Testing end-to-end workflow...")
    from algorithms.base import SimpleLinearCausalModel
    from utils.data_processing import DataProcessor
    import pandas as pd
    import numpy as np
    
    # 1. Create realistic synthetic data with known causal structure
    np.random.seed(42)
    n_samples = 500
    
    # Create data with known causal relationships: A -> B -> C, A -> D
    A = np.random.randn(n_samples)
    B = 0.8 * A + 0.2 * np.random.randn(n_samples)
    C = 0.6 * B + 0.3 * np.random.randn(n_samples) 
    D = 0.5 * A + 0.4 * np.random.randn(n_samples)
    
    data = pd.DataFrame({
        'A': A,
        'B': B, 
        'C': C,
        'D': D
    })
    
    # 2. Process data
    processor = DataProcessor()
    is_valid, issues = processor.validate_data(data)
    assert is_valid, f"Data validation failed: {issues}"
    
    cleaned_data = processor.clean_data(data)
    standardized_data = processor.standardize(cleaned_data)
    
    # 3. Discover causal relationships
    model = SimpleLinearCausalModel(threshold=0.4)
    result = model.fit_discover(standardized_data)
    
    # 4. Validate results
    assert result.adjacency_matrix.shape == (4, 4), "Incorrect adjacency matrix shape"
    assert result.method_used == "SimpleLinearCausal", "Incorrect method name"
    assert len(result.metadata['variable_names']) == 4, "Incorrect number of variables"
    
    print("✅ End-to-end workflow successful")
    print(f"   - Data shape: {data.shape}")
    print(f"   - Detected {result.metadata['n_edges']} causal edges")
    
    # Print adjacency matrix for verification
    import numpy as np
    print(f"   - Adjacency matrix:")
    for i, var1 in enumerate(result.metadata['variable_names']):
        for j, var2 in enumerate(result.metadata['variable_names']):
            if result.adjacency_matrix[i,j] == 1:
                confidence = result.confidence_scores[i,j]
                print(f"     {var1} -> {var2} (confidence: {confidence:.3f})")
    
    return True

if __name__ == "__main__":
    print("🚀 GENERATION 1 FUNCTIONALITY TEST - Make it Work")
    print("=" * 60)
    
    try:
        # Test core functionality
        test_basic_imports()
        print()
        
        test_data_processing()
        print()
        
        test_causal_discovery()
        print()
        
        test_end_to_end()
        print()
        
        print("🎉 ALL GENERATION 1 TESTS PASSED!")
        print("✅ Basic functionality is working correctly")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)