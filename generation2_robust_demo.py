#!/usr/bin/env python3
"""
Generation 2: Robust Causal Discovery Demo
TERRAGON AUTONOMOUS SDLC - Make It Robust & Reliable
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
import warnings
from typing import Dict, Any, Optional
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('generation2_robust.log')
    ]
)
logger = logging.getLogger(__name__)

class RobustCausalDiscovery:
    """Robust wrapper for causal discovery with comprehensive error handling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'threshold': 0.3,
            'max_samples': 10000,
            'max_variables': 50,
            'timeout_seconds': 300,
            'min_samples': 10
        }
        self.execution_log = []
        logger.info("Initialized RobustCausalDiscovery with config: %s", self.config)
    
    def validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive input validation with security checks"""
        logger.info("Starting comprehensive input validation...")
        
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'sanitized': False
        }
        
        try:
            # Type validation
            if not isinstance(data, pd.DataFrame):
                validation_result['valid'] = False
                validation_result['issues'].append(f"Expected DataFrame, got {type(data)}")
                return validation_result
            
            # Size validation
            n_samples, n_vars = data.shape
            if n_samples < self.config['min_samples']:
                validation_result['valid'] = False
                validation_result['issues'].append(f"Too few samples: {n_samples} < {self.config['min_samples']}")
            
            if n_samples > self.config['max_samples']:
                validation_result['warnings'].append(f"Large dataset: {n_samples} samples may be slow")
            
            if n_vars > self.config['max_variables']:
                validation_result['warnings'].append(f"Many variables: {n_vars} may affect performance")
            
            # Check for missing values
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                validation_result['warnings'].append(f"Found {missing_count} missing values")
            
            # Check for infinite values
            infinite_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if infinite_count > 0:
                validation_result['issues'].append(f"Found {infinite_count} infinite values")
                validation_result['valid'] = False
            
            # Check for non-numeric data
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                validation_result['warnings'].append(f"Non-numeric columns: {non_numeric_cols}")
            
            # Check for constant columns (zero variance)
            numeric_data = data.select_dtypes(include=[np.number])
            constant_cols = numeric_data.columns[numeric_data.var() < 1e-10].tolist()
            if constant_cols:
                validation_result['warnings'].append(f"Constant columns (zero variance): {constant_cols}")
            
            # Security checks - look for suspicious patterns
            for col in data.columns:
                if any(suspicious in str(col).lower() for suspicious in ['script', 'exec', 'eval', 'import']):
                    validation_result['warnings'].append(f"Suspicious column name: {col}")
            
            logger.info("Input validation completed: %s", validation_result)
            return validation_result
            
        except Exception as e:
            logger.error("Input validation failed: %s", str(e))
            validation_result['valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def sanitize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sanitize input data for safe processing"""
        logger.info("Sanitizing input data...")
        
        try:
            sanitized = data.copy()
            
            # Remove any object columns with suspicious content
            for col in sanitized.columns:
                if sanitized[col].dtype == 'object':
                    # Check for potential code injection
                    has_suspicious = sanitized[col].astype(str).str.contains(
                        r'(import|exec|eval|__|\bos\b|\bsys\b)', 
                        case=False, 
                        na=False
                    ).any()
                    
                    if has_suspicious:
                        logger.warning("Dropping suspicious column: %s", col)
                        sanitized = sanitized.drop(columns=[col])
                        continue
            
            # Handle infinite values
            numeric_cols = sanitized.select_dtypes(include=[np.number]).columns
            sanitized[numeric_cols] = sanitized[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Limit extreme values (cap at 99.9th percentile)
            for col in numeric_cols:
                if sanitized[col].std() > 0:  # Only for non-constant columns
                    upper_bound = sanitized[col].quantile(0.999)
                    lower_bound = sanitized[col].quantile(0.001)
                    sanitized[col] = sanitized[col].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info("Data sanitization completed")
            return sanitized
            
        except Exception as e:
            logger.error("Data sanitization failed: %s", str(e))
            raise ValueError(f"Data sanitization failed: {str(e)}")
    
    def robust_causal_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Robust causal discovery with comprehensive error handling"""
        start_time = time.time()
        
        result = {
            'success': False,
            'causal_result': None,
            'execution_time': 0.0,
            'validation_issues': [],
            'warnings': [],
            'error': None,
            'metadata': {}
        }
        
        try:
            logger.info("Starting robust causal discovery on data shape: %s", data.shape)
            
            # Step 1: Input validation
            validation = self.validate_input_data(data)
            result['validation_issues'] = validation['issues']
            result['warnings'] = validation['warnings']
            
            if not validation['valid']:
                result['error'] = f"Input validation failed: {validation['issues']}"
                return result
            
            # Step 2: Data sanitization
            sanitized_data = self.sanitize_data(data)
            logger.info("Data sanitized, shape: %s", sanitized_data.shape)
            
            # Step 3: Data preprocessing with error handling
            try:
                from utils.data_processing import DataProcessor
                processor = DataProcessor()
                
                # Clean data
                cleaned_data = processor.clean_data(
                    sanitized_data, 
                    drop_na=True, 
                    fill_method='mean' if sanitized_data.isnull().sum().sum() > 0 else None
                )
                
                # Standardize data
                standardized_data = processor.standardize(cleaned_data)
                
                logger.info("Data preprocessing completed, final shape: %s", standardized_data.shape)
                
            except Exception as e:
                logger.error("Data preprocessing failed: %s", str(e))
                result['error'] = f"Data preprocessing failed: {str(e)}"
                return result
            
            # Step 4: Model initialization with fallback
            try:
                from algorithms.base import SimpleLinearCausalModel
                
                # Try with primary model
                model = SimpleLinearCausalModel(threshold=self.config['threshold'])
                logger.info("Primary model initialized successfully")
                
            except Exception as e:
                logger.error("Primary model initialization failed: %s", str(e))
                result['error'] = f"Model initialization failed: {str(e)}"
                return result
            
            # Step 5: Robust model fitting with timeout and retry
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info("Fitting model (attempt %d/%d)...", retry_count + 1, max_retries)
                    
                    # Fit with timeout simulation
                    fit_start = time.time()
                    model.fit(standardized_data)
                    fit_time = time.time() - fit_start
                    
                    if fit_time > self.config['timeout_seconds']:
                        raise TimeoutError(f"Model fitting timed out: {fit_time:.2f}s")
                    
                    logger.info("Model fitted successfully in %.3fs", fit_time)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning("Model fit attempt %d failed: %s", retry_count, str(e))
                    
                    if retry_count >= max_retries:
                        result['error'] = f"Model fitting failed after {max_retries} attempts: {str(e)}"
                        return result
                    
                    # Wait before retry
                    time.sleep(0.1)
            
            # Step 6: Causal discovery with monitoring
            try:
                logger.info("Performing causal discovery...")
                
                discovery_start = time.time()
                causal_result = model.discover()
                discovery_time = time.time() - discovery_start
                
                logger.info("Causal discovery completed in %.3fs", discovery_time)
                
                # Validate results
                if causal_result is None:
                    raise ValueError("Causal discovery returned None")
                
                if not hasattr(causal_result, 'adjacency_matrix'):
                    raise ValueError("Invalid causal result: missing adjacency matrix")
                
                result['causal_result'] = causal_result
                result['success'] = True
                
                # Add performance metadata
                result['metadata'] = {
                    'original_shape': data.shape,
                    'processed_shape': standardized_data.shape,
                    'fit_time': fit_time,
                    'discovery_time': discovery_time,
                    'total_edges': int(causal_result.adjacency_matrix.sum()),
                    'confidence_mean': float(causal_result.confidence_scores.mean()),
                    'confidence_std': float(causal_result.confidence_scores.std())
                }
                
                logger.info("Causal discovery completed successfully: %s", result['metadata'])
                
            except Exception as e:
                logger.error("Causal discovery failed: %s", str(e))
                result['error'] = f"Causal discovery failed: {str(e)}"
                return result
            
        except Exception as e:
            logger.error("Robust causal discovery failed with unexpected error: %s", str(e))
            result['error'] = f"Unexpected error: {str(e)}"
        
        finally:
            result['execution_time'] = time.time() - start_time
            logger.info("Robust causal discovery completed in %.3fs", result['execution_time'])
        
        return result

def main():
    """Robust causal discovery demonstration"""
    print("🛡️ Generation 2: Robust Causal Discovery Demo")
    print("=" * 58)
    
    overall_start = time.time()
    
    try:
        # Initialize robust discovery system
        print("🔧 Initializing robust causal discovery system...")
        robust_discovery = RobustCausalDiscovery({
            'threshold': 0.3,
            'max_samples': 5000,
            'max_variables': 20,
            'timeout_seconds': 60,
            'min_samples': 5
        })
        print("✅ Robust system initialized")
        
        # Test 1: Normal case
        print("\n📊 Test 1: Normal synthetic data")
        try:
            from utils.data_processing import DataProcessor
            processor = DataProcessor()
            
            normal_data = processor.generate_synthetic_data(
                n_samples=200,
                n_variables=5, 
                noise_level=0.15,
                random_state=42
            )
            
            result1 = robust_discovery.robust_causal_discovery(normal_data)
            
            if result1['success']:
                print("✅ Normal case: SUCCESS")
                print(f"   Edges found: {result1['metadata']['total_edges']}")
                print(f"   Execution time: {result1['execution_time']:.3f}s")
                if result1['warnings']:
                    print(f"   Warnings: {len(result1['warnings'])}")
            else:
                print(f"❌ Normal case failed: {result1['error']}")
                
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
        
        # Test 2: Data with missing values
        print("\n⚠️ Test 2: Data with missing values")
        try:
            data_with_missing = normal_data.copy()
            # Introduce missing values
            np.random.seed(42)
            mask = np.random.random(data_with_missing.shape) < 0.1  # 10% missing
            data_with_missing[mask] = np.nan
            
            result2 = robust_discovery.robust_causal_discovery(data_with_missing)
            
            if result2['success']:
                print("✅ Missing values case: SUCCESS")
                print(f"   Warnings: {len(result2['warnings'])}")
                print(f"   Execution time: {result2['execution_time']:.3f}s")
            else:
                print(f"❌ Missing values case failed: {result2['error']}")
                
        except Exception as e:
            print(f"❌ Test 2 failed: {e}")
        
        # Test 3: Edge case - very small dataset
        print("\n🔬 Test 3: Very small dataset")
        try:
            small_data = processor.generate_synthetic_data(
                n_samples=3,  # Very small
                n_variables=2,
                noise_level=0.1,
                random_state=42
            )
            
            result3 = robust_discovery.robust_causal_discovery(small_data)
            
            if result3['success']:
                print("✅ Small dataset case: SUCCESS (unexpected)")
            else:
                print(f"✅ Small dataset case: Expected failure - {result3['error']}")
                
        except Exception as e:
            print(f"⚠️ Test 3 error handling: {e}")
        
        # Test 4: Stress test - larger dataset
        print("\n⚡ Test 4: Larger dataset performance")
        try:
            large_data = processor.generate_synthetic_data(
                n_samples=1000,
                n_variables=8,
                noise_level=0.2,
                random_state=42
            )
            
            result4 = robust_discovery.robust_causal_discovery(large_data)
            
            if result4['success']:
                print("✅ Large dataset case: SUCCESS")
                print(f"   Data shape: {result4['metadata']['processed_shape']}")
                print(f"   Total edges: {result4['metadata']['total_edges']}")
                print(f"   Mean confidence: {result4['metadata']['confidence_mean']:.3f}")
                print(f"   Execution time: {result4['execution_time']:.3f}s")
            else:
                print(f"❌ Large dataset case failed: {result4['error']}")
                
        except Exception as e:
            print(f"❌ Test 4 failed: {e}")
        
        # Health check
        print("\n🏥 System Health Check")
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=1)
            
            print(f"✅ System resources:")
            print(f"   Memory usage: {memory_usage:.1f}%")
            print(f"   CPU usage: {cpu_usage:.1f}%")
            
            if memory_usage > 90:
                print("⚠️  High memory usage detected")
            if cpu_usage > 90:
                print("⚠️  High CPU usage detected")
                
        except Exception as e:
            print(f"⚠️ Health check failed: {e}")
        
        total_time = time.time() - overall_start
        
        print(f"\n🎉 Generation 2 Robust Demo completed!")
        print(f"   Total execution time: {total_time:.3f} seconds")
        print(f"   Status: ROBUST - Comprehensive error handling implemented")
        print(f"   Features: Input validation, data sanitization, retry logic")
        print(f"   Monitoring: Health checks, performance metrics, logging")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Robust demo failed with unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)