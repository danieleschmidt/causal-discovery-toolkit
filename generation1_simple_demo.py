#!/usr/bin/env python3
"""
Generation 1: Simple Causal Discovery Demo
TERRAGON AUTONOMOUS SDLC - Make It Work
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Simple demonstration of causal discovery functionality"""
    print("🚀 Generation 1: Simple Causal Discovery Demo")
    print("=" * 55)
    
    start_time = time.time()
    
    try:
        # Import core components
        print("📦 Loading core modules...")
        from algorithms.base import SimpleLinearCausalModel, CausalResult
        from utils.data_processing import DataProcessor
        print("✅ Core imports successful")
        
        # Initialize data processor
        print("\n🔧 Initializing data processor...")
        processor = DataProcessor()
        print("✅ Data processor ready")
        
        # Generate synthetic test data
        print("\n📊 Generating synthetic data...")
        data = processor.generate_synthetic_data(
            n_samples=100, 
            n_variables=4,
            noise_level=0.2,
            random_state=42
        )
        print(f"✅ Generated data: {data.shape[0]} samples, {data.shape[1]} variables")
        print(f"   Variables: {list(data.columns)}")
        print(f"   Sample statistics:")
        for col in data.columns:
            print(f"     {col}: mean={data[col].mean():.3f}, std={data[col].std():.3f}")
        
        # Clean and standardize data
        print("\n🧹 Preprocessing data...")
        cleaned_data = processor.clean_data(data, drop_na=True)
        standardized_data = processor.standardize(cleaned_data)
        print(f"✅ Data cleaned and standardized")
        print(f"   Final shape: {standardized_data.shape}")
        
        # Initialize causal discovery model
        print("\n🧠 Initializing causal discovery model...")
        model = SimpleLinearCausalModel(threshold=0.3)
        print("✅ Model initialized with correlation threshold = 0.3")
        
        # Fit the model
        print("\n🎯 Fitting model to data...")
        model.fit(standardized_data)
        print("✅ Model fitted successfully")
        
        # Discover causal relationships
        print("\n🔍 Discovering causal relationships...")
        result = model.discover()
        
        # Display results
        print("\n📈 Results Summary:")
        print(f"   Method used: {result.method_used}")
        print(f"   Variables analyzed: {result.metadata['n_variables']}")
        print(f"   Causal edges found: {result.metadata['n_edges']}")
        print(f"   Threshold used: {result.metadata['threshold']}")
        
        # Show adjacency matrix
        print("\n🔗 Adjacency Matrix:")
        var_names = result.metadata['variable_names']
        adj_matrix = result.adjacency_matrix
        
        print("     ", end="")
        for name in var_names:
            print(f"{name:>8}", end="")
        print()
        
        for i, name in enumerate(var_names):
            print(f"{name:>5}", end="")
            for j in range(len(var_names)):
                print(f"{adj_matrix[i,j]:>8}", end="")
            print()
        
        # Show strongest relationships
        print("\n💪 Strongest Relationships (confidence > 0.5):")
        confidence_matrix = result.confidence_scores
        strong_relationships = []
        
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i != j and confidence_matrix[i,j] > 0.5:
                    strong_relationships.append({
                        'from': var_names[i],
                        'to': var_names[j], 
                        'confidence': confidence_matrix[i,j]
                    })
        
        if strong_relationships:
            for rel in sorted(strong_relationships, key=lambda x: x['confidence'], reverse=True):
                print(f"   {rel['from']} → {rel['to']}: {rel['confidence']:.3f}")
        else:
            print("   No relationships above threshold found")
        
        execution_time = time.time() - start_time
        
        print(f"\n🎉 Generation 1 Demo completed successfully!")
        print(f"   Total execution time: {execution_time:.3f} seconds")
        print(f"   Status: WORKING - Basic functionality demonstrated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)