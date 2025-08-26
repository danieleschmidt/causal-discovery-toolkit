#!/usr/bin/env python3
"""
Autonomous SDLC - Robust Demo (Generation 2)
Enhanced with comprehensive error handling, validation, and monitoring
"""

import sys
import os
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('robust_demo.log')
    ]
)
logger = logging.getLogger(__name__)


class RobustCausalDiscovery:
    """Robust causal discovery with comprehensive error handling and validation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.metrics: Dict[str, Any] = {}
        self.execution_history: List[Dict] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with robust settings"""
        return {
            "max_retries": 3,
            "timeout_seconds": 300,
            "min_samples": 10,
            "max_variables": 1000,
            "validation_enabled": True,
            "monitoring_enabled": True,
            "backup_enabled": True
        }
    
    def validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data validation with security checks"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Basic validation
            if data is None or data.empty:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Data is empty or None")
                return validation_results
            
            # Shape validation
            n_samples, n_features = data.shape
            validation_results["metrics"]["n_samples"] = n_samples
            validation_results["metrics"]["n_features"] = n_features
            
            if n_samples < self.config["min_samples"]:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Insufficient samples: {n_samples} < {self.config['min_samples']}")
            
            if n_features > self.config["max_variables"]:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Too many variables: {n_features} > {self.config['max_variables']}")
            
            # Data quality checks
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                validation_results["warnings"].append(f"Missing values detected: {missing_values}")
                validation_results["metrics"]["missing_values"] = missing_values
            
            # Check for infinite values
            infinite_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if infinite_values > 0:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Infinite values detected: {infinite_values}")
            
            # Check for constant columns
            constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
            if constant_columns:
                validation_results["warnings"].append(f"Constant columns detected: {constant_columns}")
                validation_results["metrics"]["constant_columns"] = len(constant_columns)
            
            # Security check: column names validation
            unsafe_chars = ['<', '>', '&', '"', "'", '\\', '/', '..']
            unsafe_columns = [col for col in data.columns if any(char in str(col) for char in unsafe_chars)]
            if unsafe_columns:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Unsafe column names detected: {unsafe_columns}")
            
            logger.info(f"Data validation completed: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
            return validation_results
    
    def sanitize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sanitize input data for security and robustness"""
        try:
            # Create copy to avoid modifying original
            clean_data = data.copy()
            
            # Handle missing values
            if clean_data.isnull().any().any():
                logger.warning("Filling missing values with column means")
                numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
                clean_data[numeric_columns] = clean_data[numeric_columns].fillna(
                    clean_data[numeric_columns].mean()
                )
                
                # For non-numeric columns, fill with mode
                non_numeric = clean_data.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric:
                    clean_data[col] = clean_data[col].fillna(clean_data[col].mode().iloc[0] if not clean_data[col].mode().empty else "unknown")
            
            # Cap extreme values (outlier handling)
            numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                q95 = clean_data[col].quantile(0.95)
                q05 = clean_data[col].quantile(0.05)
                clean_data[col] = np.clip(clean_data[col], q05, q95)
            
            # Sanitize column names
            clean_data.columns = [f"var_{i}" if not str(col).replace('_', '').replace('-', '').isalnum() 
                                else str(col) for i, col in enumerate(clean_data.columns)]
            
            logger.info("Data sanitization completed successfully")
            return clean_data
            
        except Exception as e:
            logger.error(f"Data sanitization error: {e}")
            raise ValueError(f"Failed to sanitize data: {e}")
    
    def run_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic and error recovery"""
        last_exception = None
        
        for attempt in range(self.config["max_retries"]):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record successful execution
                self.execution_history.append({
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "success": True,
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time()
                })
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                self.execution_history.append({
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time()
                })
                
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        logger.error(f"All {self.config['max_retries']} attempts failed for {func.__name__}")
        raise last_exception
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor system health and resource usage"""
        try:
            import psutil
            
            health_metrics = {
                "memory_usage_percent": psutil.virtual_memory().percent,
                "cpu_usage_percent": psutil.cpu_percent(interval=1),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "timestamp": time.time()
            }
            
            # Check for resource constraints
            warnings = []
            if health_metrics["memory_usage_percent"] > 90:
                warnings.append("High memory usage detected")
            if health_metrics["cpu_usage_percent"] > 95:
                warnings.append("High CPU usage detected")
            if health_metrics["disk_usage_percent"] > 95:
                warnings.append("Low disk space detected")
            
            health_metrics["warnings"] = warnings
            health_metrics["status"] = "warning" if warnings else "healthy"
            
            return health_metrics
            
        except ImportError:
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def create_backup(self, data: pd.DataFrame) -> str:
        """Create backup of input data"""
        try:
            if not self.config["backup_enabled"]:
                return "backup_disabled"
            
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            backup_file = backup_dir / f"data_backup_{timestamp}.csv"
            
            data.to_csv(backup_file, index=False)
            logger.info(f"Data backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.warning(f"Backup creation failed: {e}")
            return f"backup_failed: {e}"
    
    def robust_causal_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute causal discovery with robust error handling"""
        try:
            # Import core components with error handling
            try:
                from algorithms.base import SimpleLinearCausalModel
                from utils.data_processing import DataProcessor
            except ImportError as e:
                logger.error(f"Failed to import required modules: {e}")
                raise ImportError(f"Core modules unavailable: {e}")
            
            # Pre-execution health check
            health_status = self.monitor_system_health()
            if health_status["status"] == "warning":
                logger.warning(f"System health warnings: {health_status['warnings']}")
            
            # Data validation
            validation_results = self.validate_input_data(data)
            
            # Data sanitization (even if validation has issues - we can fix them)
            clean_data = self.sanitize_data(data)
            
            # Re-validate after sanitization
            if not validation_results["is_valid"]:
                logger.warning(f"Initial validation failed, attempting data sanitization: {validation_results['issues']}")
                post_sanitization_validation = self.validate_input_data(clean_data)
                if not post_sanitization_validation["is_valid"]:
                    raise ValueError(f"Data validation failed even after sanitization: {post_sanitization_validation['issues']}")
                else:
                    logger.info("Data successfully sanitized and validated")
                    validation_results = post_sanitization_validation
            
            # Create backup
            backup_path = self.create_backup(clean_data)
            
            # Execute causal discovery with retry logic
            def discovery_function():
                model = SimpleLinearCausalModel(threshold=0.3)
                return model.fit_discover(clean_data)
            
            result = self.run_with_retry(discovery_function)
            
            # Compile comprehensive results
            robust_results = {
                "causal_result": {
                    "method": result.method_used,
                    "edges_found": int(result.adjacency_matrix.sum()),
                    "adjacency_matrix": result.adjacency_matrix.tolist(),
                    "confidence_scores": result.confidence_scores.tolist(),
                    "metadata": result.metadata
                },
                "validation_results": validation_results,
                "health_status": health_status,
                "backup_path": backup_path,
                "execution_history": self.execution_history[-3:],  # Last 3 attempts
                "system_metrics": self.metrics,
                "status": "success"
            }
            
            logger.info("Robust causal discovery completed successfully")
            return robust_results
            
        except Exception as e:
            logger.error(f"Robust causal discovery failed: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "health_status": self.monitor_system_health(),
                "execution_history": self.execution_history,
                "timestamp": time.time()
            }


def main():
    """Main robust demo execution"""
    print("🛡️ Autonomous SDLC - Robust Demo (Generation 2)")
    print("=" * 60)
    
    try:
        # Initialize robust discovery system
        robust_config = {
            "max_retries": 3,
            "timeout_seconds": 300,
            "min_samples": 50,
            "max_variables": 100,
            "validation_enabled": True,
            "monitoring_enabled": True,
            "backup_enabled": True
        }
        
        discovery_system = RobustCausalDiscovery(robust_config)
        
        print("✅ Robust discovery system initialized")
        
        # Generate test data with some challenging properties
        from utils.data_processing import DataProcessor
        processor = DataProcessor()
        
        # Generate slightly challenging synthetic data
        data = processor.generate_synthetic_data(
            n_samples=200,
            n_variables=8,
            noise_level=0.3,
            random_state=42
        )
        
        # Introduce some data quality issues for robustness testing  
        data.iloc[10:15, 2] = np.nan  # Missing values
        data.iloc[20, 3] = np.inf     # Infinite value (will be handled by sanitization)
        
        print(f"✅ Generated challenging test data: {data.shape}")
        
        # Execute robust causal discovery
        print("\n🔍 Executing robust causal discovery...")
        results = discovery_system.robust_causal_discovery(data)
        
        # Display comprehensive results
        if results["status"] == "success":
            print("\n🎉 Robust causal discovery completed successfully!")
            
            causal_results = results["causal_result"]
            print(f"   Method: {causal_results['method']}")
            print(f"   Edges found: {causal_results['edges_found']}")
            print(f"   Data validation: {'✅ PASSED' if results['validation_results']['is_valid'] else '❌ FAILED'}")
            print(f"   System health: {results['health_status']['status'].upper()}")
            print(f"   Backup created: {results['backup_path']}")
            
            if results["validation_results"]["warnings"]:
                print(f"   Warnings handled: {len(results['validation_results']['warnings'])}")
            
            if results["health_status"]["warnings"]:
                print(f"   System warnings: {len(results['health_status']['warnings'])}")
            
            print(f"\n📊 Resource Usage:")
            health = results["health_status"]
            if "memory_usage_percent" in health:
                print(f"   Memory: {health['memory_usage_percent']:.1f}%")
                print(f"   CPU: {health['cpu_usage_percent']:.1f}%")
                print(f"   Available Memory: {health['available_memory_gb']:.2f} GB")
            
            return True
            
        else:
            print(f"\n❌ Robust causal discovery failed:")
            print(f"   Error: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Main demo execution failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Demo failed with unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)