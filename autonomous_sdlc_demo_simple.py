#!/usr/bin/env python3
"""
Simple Autonomous SDLC Demo - Generation 1
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Simple demo of causal discovery toolkit"""
    print("🚀 Autonomous SDLC - Simple Demo")
    print("=" * 50)
    
    try:
        # Import core components
        from algorithms.base import SimpleLinearCausalModel
        from utils.data_processing import DataProcessor
        
        print("✅ Core imports successful")
        
        # Create simple synthetic data
        processor = DataProcessor()
        data = processor.generate_synthetic_data(
            n_samples=100, 
            n_variables=5,
            noise_level=0.2
        )
        
        print(f"✅ Generated synthetic data: {data.shape}")
        
        # Run simple causal discovery
        model = SimpleLinearCausalModel(threshold=0.3)
        result = model.fit_discover(data)
        
        print(f"✅ Causal discovery completed")
        print(f"   Method: {result.method_used}")
        print(f"   Edges found: {result.adjacency_matrix.sum()}")
        print(f"   Threshold: {result.metadata.get('threshold', 'N/A')}")
        
        print("\n🎉 Simple demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
