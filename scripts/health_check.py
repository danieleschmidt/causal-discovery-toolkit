#!/usr/bin/env python3
"""
Production health check script for bioneuro-olfactory causal discovery toolkit.
Validates system health, dependencies, configuration, and performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import subprocess

try:
    import psutil
    import numpy as np
    import pandas as pd
    from utils.monitoring import PerformanceMonitor, HealthChecker
    from utils.validation import DataValidator
    from utils.logging_config import get_logger
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class SystemHealthChecker:
    """Comprehensive system health checker for production deployment."""
    
    def __init__(self):
        """Initialize health checker."""
        self.results = {}
        self.warnings = []
        self.errors = []
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive results."""
        print("🔍 Starting comprehensive health check...")
        print("=" * 60)
        
        # Core system checks
        self.check_system_resources()
        self.check_python_environment()
        self.check_dependencies()
        
        # Application-specific checks
        if DEPENDENCIES_AVAILABLE:
            self.check_import_functionality()
            self.check_core_functionality()
            self.check_configuration()
            self.check_performance()
        else:
            self.errors.append(f"Dependencies not available: {IMPORT_ERROR}")
        
        # File system checks
        self.check_file_permissions()
        self.check_disk_space()
        
        # Generate summary
        self.generate_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if not self.errors else "unhealthy",
            "results": self.results,
            "warnings": self.warnings,
            "errors": self.errors
        }
    
    def check_system_resources(self):
        """Check system resources (CPU, memory, disk)."""
        print("📊 Checking system resources...")
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_usage_pct = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            self.results["memory"] = {
                "usage_percent": memory_usage_pct,
                "available_gb": memory_available_gb,
                "total_gb": memory.total / (1024**3),
                "status": "healthy" if memory_usage_pct < 85 else "warning"
            }
            
            if memory_usage_pct > 85:
                self.warnings.append(f"High memory usage: {memory_usage_pct:.1f}%")
            elif memory_available_gb < 1.0:
                self.warnings.append(f"Low available memory: {memory_available_gb:.1f}GB")
            
            print(f"   ✅ Memory: {memory_usage_pct:.1f}% used ({memory_available_gb:.1f}GB available)")
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            self.results["cpu"] = {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "status": "healthy" if cpu_percent < 80 else "warning"
            }
            
            if cpu_percent > 80:
                self.warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            print(f"   ✅ CPU: {cpu_percent:.1f}% usage ({cpu_count} cores)")
            
        except Exception as e:
            self.errors.append(f"System resource check failed: {str(e)}")
            print(f"   ❌ System resources: {str(e)}")
    
    def check_python_environment(self):
        """Check Python environment and version."""
        print("🐍 Checking Python environment...")
        
        try:
            python_version = sys.version_info
            version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            self.results["python"] = {
                "version": version_str,
                "executable": sys.executable,
                "platform": sys.platform,
                "status": "healthy" if python_version >= (3, 8) else "error"
            }
            
            if python_version < (3, 8):
                self.errors.append(f"Python version too old: {version_str} (requires >= 3.8)")
                print(f"   ❌ Python: {version_str} (too old)")
            else:
                print(f"   ✅ Python: {version_str}")
            
            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            
            self.results["python"]["virtual_env"] = in_venv
            
            if not in_venv:
                self.warnings.append("Not running in virtual environment")
                print(f"   ⚠️  Virtual environment: Not detected")
            else:
                print(f"   ✅ Virtual environment: Active")
                
        except Exception as e:
            self.errors.append(f"Python environment check failed: {str(e)}")
            print(f"   ❌ Python environment: {str(e)}")
    
    def check_dependencies(self):
        """Check required dependencies."""
        print("📦 Checking dependencies...")
        
        required_packages = [
            ("numpy", "1.21.0"),
            ("pandas", "1.3.0"),
            ("scikit-learn", "1.0.0"),
            ("scipy", "1.7.0"),
            ("psutil", "5.0.0"),
        ]
        
        optional_packages = [
            ("matplotlib", "3.4.0"),
            ("torch", "2.0.0"),
            ("jax", "0.4.0"),
        ]
        
        dependency_results = {}
        
        for package, min_version in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                dependency_results[package] = {
                    "installed": True,
                    "version": version,
                    "required": True,
                    "status": "healthy"
                }
                print(f"   ✅ {package}: {version}")
            except ImportError:
                dependency_results[package] = {
                    "installed": False,
                    "version": None,
                    "required": True,
                    "status": "error"
                }
                self.errors.append(f"Required package missing: {package}")
                print(f"   ❌ {package}: Not installed")
        
        for package, min_version in optional_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                dependency_results[package] = {
                    "installed": True,
                    "version": version,
                    "required": False,
                    "status": "healthy"
                }
                print(f"   ✅ {package}: {version} (optional)")
            except (ImportError, OSError) as e:
                dependency_results[package] = {
                    "installed": False,
                    "version": None,
                    "required": False,
                    "status": "warning",
                    "error": str(e)
                }
                self.warnings.append(f"Optional package issue: {package} - {str(e)}")
                print(f"   ⚠️  {package}: Not available (optional)")
        
        self.results["dependencies"] = dependency_results
    
    def check_import_functionality(self):
        """Check that core modules can be imported."""
        print("🔧 Checking import functionality...")
        
        import_tests = [
            ("algorithms.base", "SimpleLinearCausalModel"),
            ("algorithms.bioneuro_olfactory", "OlfactoryNeuralCausalModel"),
            ("utils.bioneuro_data_processing", "BioneuroDataProcessor"),
            ("utils.monitoring", "PerformanceMonitor"),
            ("utils.validation", "DataValidator"),
        ]
        
        import_results = {}
        
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                
                # Try to instantiate
                if class_name == "SimpleLinearCausalModel":
                    instance = cls(threshold=0.1)
                elif class_name == "OlfactoryNeuralCausalModel":
                    instance = cls()
                elif class_name == "BioneuroDataProcessor":
                    instance = cls()
                elif class_name == "PerformanceMonitor":
                    instance = cls()
                elif class_name == "DataValidator":
                    instance = cls()
                
                import_results[f"{module_name}.{class_name}"] = {
                    "import_success": True,
                    "instantiation_success": True,
                    "status": "healthy"
                }
                print(f"   ✅ {module_name}.{class_name}")
                
            except ImportError as e:
                import_results[f"{module_name}.{class_name}"] = {
                    "import_success": False,
                    "instantiation_success": False,
                    "error": str(e),
                    "status": "error"
                }
                self.errors.append(f"Import failed: {module_name}.{class_name} - {str(e)}")
                print(f"   ❌ {module_name}.{class_name}: Import failed")
                
            except Exception as e:
                import_results[f"{module_name}.{class_name}"] = {
                    "import_success": True,
                    "instantiation_success": False,
                    "error": str(e),
                    "status": "warning"
                }
                self.warnings.append(f"Instantiation failed: {module_name}.{class_name} - {str(e)}")
                print(f"   ⚠️  {module_name}.{class_name}: Instantiation failed")
        
        self.results["imports"] = import_results
    
    def check_core_functionality(self):
        """Check core functionality with synthetic data."""
        print("⚙️  Checking core functionality...")
        
        try:
            from algorithms.base import SimpleLinearCausalModel
            
            # Generate synthetic data
            np.random.seed(42)
            data = pd.DataFrame({
                'x1': np.random.randn(50),
                'x2': np.random.randn(50),
                'x3': np.random.randn(50)
            })
            data['y'] = 2 * data['x1'] + 0.5 * data['x2'] + np.random.randn(50) * 0.1
            
            # Test basic model
            model = SimpleLinearCausalModel(threshold=0.3)
            result = model.fit_discover(data)
            
            functionality_results = {
                "basic_causal_discovery": {
                    "success": True,
                    "edges_found": result.metadata.get("n_edges", 0),
                    "method": result.method_used,
                    "status": "healthy"
                }
            }
            
            print(f"   ✅ Basic causal discovery: {result.metadata.get('n_edges', 0)} edges found")
            
            # Test bioneuro model if available
            try:
                from algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel
                
                # Create olfactory-style data
                olfactory_data = pd.DataFrame({
                    'receptor_response_0': np.random.randn(50),
                    'receptor_response_1': np.random.randn(50),
                    'neural_firing_rate_0': np.random.randn(50),
                    'neural_firing_rate_1': np.random.randn(50)
                })
                
                olfactory_model = OlfactoryNeuralCausalModel()
                olfactory_result = olfactory_model.fit_discover(olfactory_data)
                
                functionality_results["olfactory_causal_discovery"] = {
                    "success": True,
                    "edges_found": olfactory_result.metadata.get("n_causal_edges", 0),
                    "neural_pathways": len(olfactory_result.neural_pathways),
                    "status": "healthy"
                }
                
                print(f"   ✅ Olfactory causal discovery: {olfactory_result.metadata.get('n_causal_edges', 0)} edges found")
                
            except Exception as e:
                functionality_results["olfactory_causal_discovery"] = {
                    "success": False,
                    "error": str(e),
                    "status": "warning"
                }
                self.warnings.append(f"Olfactory functionality test failed: {str(e)}")
                print(f"   ⚠️  Olfactory causal discovery: Failed")
            
            self.results["functionality"] = functionality_results
            
        except Exception as e:
            self.errors.append(f"Core functionality check failed: {str(e)}")
            print(f"   ❌ Core functionality: {str(e)}")
    
    def check_configuration(self):
        """Check configuration files and settings."""
        print("⚙️  Checking configuration...")
        
        config_checks = {}
        
        # Check for configuration files
        config_paths = [
            "config/production.json",
            "config/development.json",
            ".env"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                config_checks[config_path] = {
                    "exists": True,
                    "readable": os.access(config_path, os.R_OK),
                    "status": "healthy"
                }
                print(f"   ✅ {config_path}: Found")
                
                # Try to validate JSON files
                if config_path.endswith('.json'):
                    try:
                        with open(config_path, 'r') as f:
                            json.load(f)
                        config_checks[config_path]["valid_json"] = True
                    except json.JSONDecodeError as e:
                        config_checks[config_path]["valid_json"] = False
                        config_checks[config_path]["status"] = "error"
                        self.errors.append(f"Invalid JSON in {config_path}: {str(e)}")
                        print(f"   ❌ {config_path}: Invalid JSON")
            else:
                config_checks[config_path] = {
                    "exists": False,
                    "status": "warning"
                }
                self.warnings.append(f"Configuration file not found: {config_path}")
                print(f"   ⚠️  {config_path}: Not found")
        
        # Check environment variables
        env_vars = [
            "BIONEURO_LOG_LEVEL",
            "BIONEURO_DATA_DIR",
            "BIONEURO_RESULTS_DIR"
        ]
        
        env_checks = {}
        for var in env_vars:
            value = os.environ.get(var)
            env_checks[var] = {
                "set": value is not None,
                "value": value if value else None,
                "status": "warning" if value is None else "healthy"
            }
            
            if value:
                print(f"   ✅ {var}: {value}")
            else:
                print(f"   ⚠️  {var}: Not set")
                self.warnings.append(f"Environment variable not set: {var}")
        
        config_checks["environment_variables"] = env_checks
        self.results["configuration"] = config_checks
    
    def check_performance(self):
        """Check performance characteristics."""
        print("🚀 Checking performance...")
        
        try:
            # Simple performance test
            start_time = time.time()
            
            # Matrix operations test
            matrix_size = 100
            a = np.random.randn(matrix_size, matrix_size)
            b = np.random.randn(matrix_size, matrix_size)
            c = np.dot(a, b)
            
            matrix_time = time.time() - start_time
            
            # Data processing test
            start_time = time.time()
            
            data = pd.DataFrame(np.random.randn(1000, 10))
            processed = data.describe()
            correlations = data.corr()
            
            data_processing_time = time.time() - start_time
            
            performance_results = {
                "matrix_operations": {
                    "time_seconds": matrix_time,
                    "status": "healthy" if matrix_time < 1.0 else "warning"
                },
                "data_processing": {
                    "time_seconds": data_processing_time,
                    "status": "healthy" if data_processing_time < 0.5 else "warning"
                }
            }
            
            if matrix_time > 1.0:
                self.warnings.append(f"Slow matrix operations: {matrix_time:.2f}s")
            
            if data_processing_time > 0.5:
                self.warnings.append(f"Slow data processing: {data_processing_time:.2f}s")
            
            print(f"   ✅ Matrix operations: {matrix_time:.3f}s")
            print(f"   ✅ Data processing: {data_processing_time:.3f}s")
            
            self.results["performance"] = performance_results
            
        except Exception as e:
            self.errors.append(f"Performance check failed: {str(e)}")
            print(f"   ❌ Performance: {str(e)}")
    
    def check_file_permissions(self):
        """Check file and directory permissions."""
        print("🔒 Checking file permissions...")
        
        permission_checks = {}
        
        # Check key directories
        directories = [
            "src/",
            "examples/",
            "scripts/",
            "tests/"
        ]
        
        for directory in directories:
            if os.path.exists(directory):
                readable = os.access(directory, os.R_OK)
                executable = os.access(directory, os.X_OK)
                
                permission_checks[directory] = {
                    "exists": True,
                    "readable": readable,
                    "executable": executable,
                    "status": "healthy" if readable and executable else "warning"
                }
                
                if readable and executable:
                    print(f"   ✅ {directory}: Accessible")
                else:
                    print(f"   ⚠️  {directory}: Permission issues")
                    self.warnings.append(f"Permission issues with {directory}")
            else:
                permission_checks[directory] = {
                    "exists": False,
                    "status": "warning"
                }
                print(f"   ⚠️  {directory}: Not found")
        
        # Check script executability
        scripts = [
            "scripts/run_production_benchmark.py",
            "scripts/health_check.py"
        ]
        
        for script in scripts:
            if os.path.exists(script):
                executable = os.access(script, os.X_OK)
                permission_checks[script] = {
                    "exists": True,
                    "executable": executable,
                    "status": "healthy" if executable else "warning"
                }
                
                if executable:
                    print(f"   ✅ {script}: Executable")
                else:
                    print(f"   ⚠️  {script}: Not executable")
                    self.warnings.append(f"Script not executable: {script}")
        
        self.results["permissions"] = permission_checks
    
    def check_disk_space(self):
        """Check available disk space."""
        print("💾 Checking disk space...")
        
        try:
            current_dir = os.getcwd()
            disk_usage = psutil.disk_usage(current_dir)
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_percent = (used_gb / total_gb) * 100
            
            self.results["disk_space"] = {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "used_percent": used_percent,
                "status": "healthy" if used_percent < 90 else "warning"
            }
            
            if used_percent > 90:
                self.warnings.append(f"Low disk space: {used_percent:.1f}% used")
            elif free_gb < 1.0:
                self.warnings.append(f"Very low disk space: {free_gb:.1f}GB free")
            
            print(f"   ✅ Disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)")
            
        except Exception as e:
            self.errors.append(f"Disk space check failed: {str(e)}")
            print(f"   ❌ Disk space: {str(e)}")
    
    def generate_summary(self):
        """Generate and display health check summary."""
        print()
        print("=" * 60)
        print("📋 HEALTH CHECK SUMMARY")
        print("=" * 60)
        
        total_checks = 0
        healthy_checks = 0
        warning_checks = 0
        error_checks = 0
        
        # Count check results
        for category, checks in self.results.items():
            if isinstance(checks, dict):
                for check_name, check_result in checks.items():
                    if isinstance(check_result, dict):  # Only process dict results
                        total_checks += 1
                        status = check_result.get("status", "unknown")
                        
                        if status == "healthy":
                            healthy_checks += 1
                        elif status == "warning":
                            warning_checks += 1
                        elif status == "error":
                            error_checks += 1
        
        print(f"Total checks: {total_checks}")
        print(f"✅ Healthy: {healthy_checks}")
        print(f"⚠️  Warnings: {warning_checks}")
        print(f"❌ Errors: {error_checks}")
        print()
        
        # Overall status
        if error_checks > 0:
            print("🔴 OVERALL STATUS: UNHEALTHY")
            print("Critical issues found that may prevent proper operation.")
        elif warning_checks > 0:
            print("🟡 OVERALL STATUS: WARNING")
            print("Some issues found that may impact performance or functionality.")
        else:
            print("🟢 OVERALL STATUS: HEALTHY")
            print("All checks passed successfully.")
        
        print()
        
        # Show warnings
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        # Show errors
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        
        if not DEPENDENCIES_AVAILABLE:
            print("   • Install required dependencies: pip install -r requirements.txt")
        
        if error_checks > 0:
            print("   • Fix critical errors before deploying to production")
        
        if warning_checks > 0:
            print("   • Review and address warnings for optimal performance")
        
        if self.results.get("memory", {}).get("usage_percent", 0) > 85:
            print("   • Consider increasing available memory")
        
        if not self.results.get("python", {}).get("virtual_env", False):
            print("   • Use a virtual environment for better isolation")
        
        env_vars_missing = sum(1 for var_info in self.results.get("configuration", {}).get("environment_variables", {}).values() 
                              if not var_info.get("set", False))
        if env_vars_missing > 0:
            print("   • Set recommended environment variables for production")
        
        print("   • Run regular health checks to monitor system status")
        print("   • Monitor logs for runtime issues")


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(description="Production health check for bioneuro-olfactory toolkit")
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if unhealthy"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run health checks
    checker = SystemHealthChecker()
    results = checker.run_all_checks()
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Health check results saved to: {output_path}")
    
    # Exit with appropriate code
    if args.exit_code and results["overall_status"] != "healthy":
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()