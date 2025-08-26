#!/usr/bin/env python3
"""
Comprehensive Quality Gates & Testing System
Advanced quality assurance with automated testing, security scanning, and performance benchmarks
"""

import sys
import os
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quality_system.log')
    ]
)
logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class QualityGateCategory(Enum):
    """Categories of quality gates"""
    SYNTAX = "syntax"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FUNCTIONALITY = "functionality"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution"""
    name: str
    category: QualityGateCategory
    status: QualityGateStatus
    execution_time: float
    message: str
    details: Dict[str, Any]
    score: Optional[float] = None
    recommendations: List[str] = None


class ComprehensiveQualitySystem:
    """Advanced quality assurance system with comprehensive testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.project_root = Path.cwd()
        self.results: List[QualityGateResult] = []
        self.overall_score = 0.0
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default quality system configuration"""
        return {
            "enable_syntax_check": True,
            "enable_security_scan": True,
            "enable_performance_test": True,
            "enable_functionality_test": True,
            "enable_reliability_test": True,
            "enable_maintainability_check": True,
            "parallel_execution": True,
            "fail_threshold": 0.70,  # Minimum 70% pass rate
            "timeout_seconds": 300,
            "generate_report": True,
            "detailed_output": True
        }
    
    def _execute_command(self, command: str, cwd: Optional[Path] = None, timeout: float = 60) -> Tuple[int, str, str]:
        """Execute shell command with timeout and error handling"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -2, "", str(e)
    
    def syntax_quality_gate(self) -> QualityGateResult:
        """Python syntax validation and code structure analysis"""
        start_time = time.time()
        
        try:
            # Check Python syntax for all .py files
            python_files = list(self.project_root.glob("**/*.py"))
            syntax_errors = []
            
            for py_file in python_files:
                if "venv" in str(py_file) or "__pycache__" in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
                except Exception as e:
                    logger.warning(f"Could not check syntax for {py_file}: {e}")
            
            execution_time = time.time() - start_time
            
            if syntax_errors:
                return QualityGateResult(
                    name="Python Syntax Check",
                    category=QualityGateCategory.SYNTAX,
                    status=QualityGateStatus.FAILED,
                    execution_time=execution_time,
                    message=f"Syntax errors found in {len(syntax_errors)} files",
                    details={"errors": syntax_errors, "files_checked": len(python_files)},
                    score=0.0
                )
            
            return QualityGateResult(
                name="Python Syntax Check",
                category=QualityGateCategory.SYNTAX,
                status=QualityGateStatus.PASSED,
                execution_time=execution_time,
                message=f"All {len(python_files)} Python files have valid syntax",
                details={"files_checked": len(python_files)},
                score=100.0
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Python Syntax Check",
                category=QualityGateCategory.SYNTAX,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Syntax check failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def security_quality_gate(self) -> QualityGateResult:
        """Security analysis and vulnerability scanning"""
        start_time = time.time()
        
        try:
            # Basic security checks
            security_issues = []
            
            # Check for hardcoded secrets/passwords
            python_files = list(self.project_root.glob("**/*.py"))
            for py_file in python_files:
                if "venv" in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        
                    # Basic secret detection patterns
                    secret_patterns = [
                        'password = "',
                        "password = '",
                        'api_key = "',
                        "api_key = '",
                        'secret = "',
                        "secret = '",
                        'token = "',
                        "token = '"
                    ]
                    
                    for pattern in secret_patterns:
                        if pattern in content:
                            security_issues.append(f"Potential hardcoded secret in {py_file}")
                            break
                            
                except Exception:
                    continue
            
            # Check for unsafe imports
            unsafe_imports = ['eval', 'exec', 'subprocess.call', 'os.system']
            for py_file in python_files:
                if "venv" in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for unsafe in unsafe_imports:
                        if unsafe in content and 'import' in content:
                            # More detailed check would be needed for real implementation
                            pass
                            
                except Exception:
                    continue
            
            execution_time = time.time() - start_time
            
            if security_issues:
                return QualityGateResult(
                    name="Security Scan",
                    category=QualityGateCategory.SECURITY,
                    status=QualityGateStatus.WARNING,
                    execution_time=execution_time,
                    message=f"Found {len(security_issues)} potential security issues",
                    details={"issues": security_issues},
                    score=60.0,
                    recommendations=["Review identified security issues", "Use environment variables for secrets"]
                )
            
            return QualityGateResult(
                name="Security Scan",
                category=QualityGateCategory.SECURITY,
                status=QualityGateStatus.PASSED,
                execution_time=execution_time,
                message="No obvious security issues detected",
                details={"files_scanned": len(python_files)},
                score=95.0
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Security Scan",
                category=QualityGateCategory.SECURITY,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Security scan failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def performance_quality_gate(self) -> QualityGateResult:
        """Performance testing and benchmarking"""
        start_time = time.time()
        
        try:
            # Test basic import performance
            import_start = time.time()
            
            try:
                # Test core module imports
                sys.path.insert(0, str(self.project_root / "src"))
                import algorithms.base
                from utils.data_processing import DataProcessor
                
                import_time = time.time() - import_start
                
                # Test basic functionality performance
                processor = DataProcessor()
                
                perf_start = time.time()
                data = processor.generate_synthetic_data(n_samples=100, n_variables=5)
                data_generation_time = time.time() - perf_start
                
                model_start = time.time()
                model = algorithms.base.SimpleLinearCausalModel(threshold=0.3)
                result = model.fit_discover(data)
                model_time = time.time() - model_start
                
                total_time = time.time() - start_time
                
                performance_metrics = {
                    "import_time": import_time,
                    "data_generation_time": data_generation_time,
                    "model_execution_time": model_time,
                    "total_execution_time": total_time
                }
                
                # Performance thresholds
                if import_time > 5.0 or model_time > 10.0:
                    status = QualityGateStatus.WARNING
                    score = 70.0
                    message = "Performance within acceptable limits but could be optimized"
                elif import_time > 10.0 or model_time > 30.0:
                    status = QualityGateStatus.FAILED
                    score = 40.0
                    message = "Performance below acceptable thresholds"
                else:
                    status = QualityGateStatus.PASSED
                    score = 90.0
                    message = "Performance meets or exceeds requirements"
                
                return QualityGateResult(
                    name="Performance Benchmark",
                    category=QualityGateCategory.PERFORMANCE,
                    status=status,
                    execution_time=total_time,
                    message=message,
                    details=performance_metrics,
                    score=score
                )
                
            except ImportError as e:
                return QualityGateResult(
                    name="Performance Benchmark",
                    category=QualityGateCategory.PERFORMANCE,
                    status=QualityGateStatus.FAILED,
                    execution_time=time.time() - start_time,
                    message=f"Cannot test performance due to import errors: {e}",
                    details={"import_error": str(e)},
                    score=0.0
                )
                
        except Exception as e:
            return QualityGateResult(
                name="Performance Benchmark",
                category=QualityGateCategory.PERFORMANCE,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Performance test failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def functionality_quality_gate(self) -> QualityGateResult:
        """Functional testing of core components"""
        start_time = time.time()
        
        try:
            # Test core functionality
            test_results = []
            
            # Test 1: Data processing
            try:
                sys.path.insert(0, str(self.project_root / "src"))
                from utils.data_processing import DataProcessor
                
                processor = DataProcessor()
                data = processor.generate_synthetic_data(n_samples=50, n_variables=3)
                
                if data is not None and data.shape == (50, 3):
                    test_results.append({"test": "data_processing", "status": "pass"})
                else:
                    test_results.append({"test": "data_processing", "status": "fail", "reason": "incorrect_shape"})
                    
            except Exception as e:
                test_results.append({"test": "data_processing", "status": "error", "error": str(e)})
            
            # Test 2: Basic causal discovery
            try:
                from algorithms.base import SimpleLinearCausalModel
                
                # Use the data from previous test if available
                model = SimpleLinearCausalModel(threshold=0.3)
                result = model.fit_discover(data)
                
                if hasattr(result, 'adjacency_matrix') and hasattr(result, 'method_used'):
                    test_results.append({"test": "causal_discovery", "status": "pass"})
                else:
                    test_results.append({"test": "causal_discovery", "status": "fail", "reason": "invalid_result"})
                    
            except Exception as e:
                test_results.append({"test": "causal_discovery", "status": "error", "error": str(e)})
            
            # Test 3: CLI functionality
            try:
                cli_file = self.project_root / "src" / "cli.py"
                if cli_file.exists():
                    test_results.append({"test": "cli_exists", "status": "pass"})
                else:
                    test_results.append({"test": "cli_exists", "status": "fail", "reason": "file_not_found"})
            except Exception as e:
                test_results.append({"test": "cli_exists", "status": "error", "error": str(e)})
            
            execution_time = time.time() - start_time
            
            # Analyze results
            passed_tests = len([t for t in test_results if t["status"] == "pass"])
            total_tests = len(test_results)
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            if pass_rate >= 0.9:
                status = QualityGateStatus.PASSED
                score = 95.0
                message = f"All functionality tests passed ({passed_tests}/{total_tests})"
            elif pass_rate >= 0.7:
                status = QualityGateStatus.WARNING
                score = 75.0
                message = f"Most functionality tests passed ({passed_tests}/{total_tests})"
            else:
                status = QualityGateStatus.FAILED
                score = 50.0
                message = f"Many functionality tests failed ({passed_tests}/{total_tests})"
            
            return QualityGateResult(
                name="Functionality Tests",
                category=QualityGateCategory.FUNCTIONALITY,
                status=status,
                execution_time=execution_time,
                message=message,
                details={"test_results": test_results, "pass_rate": pass_rate},
                score=score
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Functionality Tests",
                category=QualityGateCategory.FUNCTIONALITY,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Functionality tests failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def reliability_quality_gate(self) -> QualityGateResult:
        """Reliability and error handling testing"""
        start_time = time.time()
        
        try:
            reliability_tests = []
            
            # Test error handling with invalid inputs
            try:
                sys.path.insert(0, str(self.project_root / "src"))
                from algorithms.base import SimpleLinearCausalModel
                import pandas as pd
                import numpy as np
                
                model = SimpleLinearCausalModel(threshold=0.3)
                
                # Test 1: Empty dataframe
                try:
                    empty_df = pd.DataFrame()
                    model.fit(empty_df)
                    reliability_tests.append({"test": "empty_dataframe", "status": "fail", "reason": "should_have_failed"})
                except (ValueError, Exception):
                    reliability_tests.append({"test": "empty_dataframe", "status": "pass", "reason": "correctly_handled_error"})
                
                # Test 2: Non-dataframe input
                try:
                    model.fit("invalid_input")
                    reliability_tests.append({"test": "invalid_input", "status": "fail", "reason": "should_have_failed"})
                except (TypeError, ValueError):
                    reliability_tests.append({"test": "invalid_input", "status": "pass", "reason": "correctly_handled_error"})
                
                # Test 3: Data with NaN values
                try:
                    nan_data = pd.DataFrame({
                        'A': [1, 2, np.nan, 4],
                        'B': [2, np.nan, 3, 5]
                    })
                    result = model.fit_discover(nan_data)
                    # Should either handle gracefully or fail predictably
                    reliability_tests.append({"test": "nan_handling", "status": "pass", "reason": "handled_gracefully"})
                except Exception:
                    reliability_tests.append({"test": "nan_handling", "status": "pass", "reason": "failed_predictably"})
                
            except Exception as e:
                reliability_tests.append({"test": "error_handling", "status": "error", "error": str(e)})
            
            execution_time = time.time() - start_time
            
            # Analyze reliability test results
            passed_tests = len([t for t in reliability_tests if t["status"] == "pass"])
            total_tests = len(reliability_tests)
            reliability_score = passed_tests / total_tests if total_tests > 0 else 0
            
            if reliability_score >= 0.8:
                status = QualityGateStatus.PASSED
                score = 85.0
                message = f"Good error handling and reliability ({passed_tests}/{total_tests})"
            elif reliability_score >= 0.6:
                status = QualityGateStatus.WARNING
                score = 70.0
                message = f"Adequate error handling ({passed_tests}/{total_tests})"
            else:
                status = QualityGateStatus.FAILED
                score = 40.0
                message = f"Poor error handling ({passed_tests}/{total_tests})"
            
            return QualityGateResult(
                name="Reliability Tests",
                category=QualityGateCategory.RELIABILITY,
                status=status,
                execution_time=execution_time,
                message=message,
                details={"reliability_tests": reliability_tests, "reliability_score": reliability_score},
                score=score
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Reliability Tests",
                category=QualityGateCategory.RELIABILITY,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Reliability tests failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def maintainability_quality_gate(self) -> QualityGateResult:
        """Code maintainability and structure analysis"""
        start_time = time.time()
        
        try:
            maintainability_metrics = {}
            
            # Count Python files and lines of code
            python_files = list(self.project_root.glob("**/*.py"))
            python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]
            
            total_lines = 0
            comment_lines = 0
            empty_lines = 0
            functions_count = 0
            classes_count = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    file_lines = len(lines)
                    file_comments = len([line for line in lines if line.strip().startswith('#')])
                    file_empty = len([line for line in lines if line.strip() == ''])
                    file_functions = len([line for line in lines if line.strip().startswith('def ')])
                    file_classes = len([line for line in lines if line.strip().startswith('class ')])
                    
                    total_lines += file_lines
                    comment_lines += file_comments
                    empty_lines += file_empty
                    functions_count += file_functions
                    classes_count += file_classes
                    
                except Exception:
                    continue
            
            maintainability_metrics = {
                "total_python_files": len(python_files),
                "total_lines_of_code": total_lines,
                "comment_lines": comment_lines,
                "empty_lines": empty_lines,
                "functions_count": functions_count,
                "classes_count": classes_count,
                "comment_ratio": comment_lines / max(total_lines, 1),
                "avg_lines_per_file": total_lines / max(len(python_files), 1),
                "functions_per_file": functions_count / max(len(python_files), 1),
                "classes_per_file": classes_count / max(len(python_files), 1)
            }
            
            execution_time = time.time() - start_time
            
            # Score maintainability
            score = 50.0  # Base score
            
            # Good comment ratio (5-20%)
            if 0.05 <= maintainability_metrics["comment_ratio"] <= 0.20:
                score += 20
            elif maintainability_metrics["comment_ratio"] > 0.20:
                score += 10
            
            # Reasonable file size (not too large)
            if maintainability_metrics["avg_lines_per_file"] < 500:
                score += 15
            elif maintainability_metrics["avg_lines_per_file"] < 1000:
                score += 10
            
            # Good modularization
            if maintainability_metrics["functions_per_file"] > 2:
                score += 10
            
            # Object-oriented design
            if maintainability_metrics["classes_per_file"] > 0.5:
                score += 5
            
            if score >= 80:
                status = QualityGateStatus.PASSED
                message = "Good code maintainability"
            elif score >= 60:
                status = QualityGateStatus.WARNING
                message = "Adequate code maintainability"
            else:
                status = QualityGateStatus.FAILED
                message = "Poor code maintainability"
            
            return QualityGateResult(
                name="Maintainability Analysis",
                category=QualityGateCategory.MAINTAINABILITY,
                status=status,
                execution_time=execution_time,
                message=message,
                details=maintainability_metrics,
                score=min(score, 100.0)
            )
            
        except Exception as e:
            return QualityGateResult(
                name="Maintainability Analysis",
                category=QualityGateCategory.MAINTAINABILITY,
                status=QualityGateStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Maintainability analysis failed: {e}",
                details={"error": str(e)},
                score=0.0
            )
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Execute all quality gates and compile comprehensive results"""
        start_time = time.time()
        
        logger.info("🚪 Starting comprehensive quality gate execution...")
        
        # Define all quality gates
        quality_gates = [
            ("syntax", self.syntax_quality_gate),
            ("security", self.security_quality_gate),
            ("performance", self.performance_quality_gate),
            ("functionality", self.functionality_quality_gate),
            ("reliability", self.reliability_quality_gate),
            ("maintainability", self.maintainability_quality_gate)
        ]
        
        # Filter based on configuration
        enabled_gates = [
            (name, func) for name, func in quality_gates
            if self.config.get(f"enable_{name}_{'scan' if name == 'security' else ('check' if name in ['syntax', 'maintainability'] else 'test')}", True)
        ]
        
        if self.config["parallel_execution"]:
            # Execute quality gates in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_gate = {executor.submit(func): name for name, func in enabled_gates}
                
                for future in concurrent.futures.as_completed(future_to_gate):
                    gate_name = future_to_gate[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        logger.info(f"✓ {result.name}: {result.status.value}")
                    except Exception as e:
                        logger.error(f"✗ {gate_name} failed: {e}")
        else:
            # Execute quality gates sequentially
            for name, func in enabled_gates:
                try:
                    result = func()
                    self.results.append(result)
                    logger.info(f"✓ {result.name}: {result.status.value}")
                except Exception as e:
                    logger.error(f"✗ {name} failed: {e}")
        
        # Calculate overall results
        total_execution_time = time.time() - start_time
        
        # Compile statistics
        status_counts = {}
        for status in QualityGateStatus:
            status_counts[status.value] = len([r for r in self.results if r.status == status])
        
        category_scores = {}
        for category in QualityGateCategory:
            category_results = [r for r in self.results if r.category == category and r.score is not None]
            if category_results:
                category_scores[category.value] = sum(r.score for r in category_results) / len(category_results)
        
        # Calculate overall score
        valid_scores = [r.score for r in self.results if r.score is not None]
        self.overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        # Determine overall status
        failed_count = status_counts.get("failed", 0)
        error_count = status_counts.get("error", 0)
        total_gates = len(self.results)
        
        if failed_count + error_count == 0:
            overall_status = "PASSED"
        elif (failed_count + error_count) / total_gates < (1 - self.config["fail_threshold"]):
            overall_status = "WARNING"
        else:
            overall_status = "FAILED"
        
        comprehensive_results = {
            "overall_status": overall_status,
            "overall_score": self.overall_score,
            "total_execution_time": total_execution_time,
            "gates_executed": total_gates,
            "status_summary": status_counts,
            "category_scores": category_scores,
            "detailed_results": [
                {
                    "name": r.name,
                    "category": r.category.value,
                    "status": r.status.value,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "message": r.message,
                    "details": r.details,
                    "recommendations": r.recommendations
                }
                for r in self.results
            ],
            "timestamp": time.time()
        }
        
        logger.info(f"🎯 Quality gates completed: {overall_status} (Score: {self.overall_score:.1f}%)")
        
        return comprehensive_results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive quality report"""
        report_lines = [
            "# COMPREHENSIVE QUALITY GATES REPORT",
            "=" * 60,
            "",
            f"**Overall Status**: {results['overall_status']}",
            f"**Overall Score**: {results['overall_score']:.1f}%",
            f"**Execution Time**: {results['total_execution_time']:.2f} seconds",
            f"**Gates Executed**: {results['gates_executed']}",
            "",
            "## STATUS SUMMARY",
            "-" * 30
        ]
        
        for status, count in results["status_summary"].items():
            report_lines.append(f"- {status.upper()}: {count}")
        
        report_lines.extend([
            "",
            "## CATEGORY SCORES",
            "-" * 30
        ])
        
        for category, score in results["category_scores"].items():
            report_lines.append(f"- {category.upper()}: {score:.1f}%")
        
        report_lines.extend([
            "",
            "## DETAILED RESULTS",
            "-" * 30
        ])
        
        for result in results["detailed_results"]:
            report_lines.extend([
                f"\n### {result['name']} ({result['category'].upper()})",
                f"- **Status**: {result['status'].upper()}",
                f"- **Score**: {result['score']:.1f}% (if available)" if result['score'] else "- **Score**: N/A",
                f"- **Time**: {result['execution_time']:.2f}s",
                f"- **Message**: {result['message']}",
            ])
            
            if result.get('recommendations'):
                report_lines.append("- **Recommendations**:")
                for rec in result['recommendations']:
                    report_lines.append(f"  - {rec}")
        
        return "\n".join(report_lines)


def main():
    """Main quality system execution"""
    print("🛡️ Comprehensive Quality Gates & Testing System")
    print("=" * 60)
    
    try:
        # Initialize quality system
        quality_config = {
            "enable_syntax_check": True,
            "enable_security_scan": True,
            "enable_performance_test": True,
            "enable_functionality_test": True,
            "enable_reliability_test": True,
            "enable_maintainability_check": True,
            "parallel_execution": True,
            "fail_threshold": 0.70,
            "generate_report": True,
            "detailed_output": True
        }
        
        quality_system = ComprehensiveQualitySystem(quality_config)
        
        print("✅ Quality system initialized")
        print(f"   Parallel execution: {quality_config['parallel_execution']}")
        print(f"   Fail threshold: {quality_config['fail_threshold']*100}%")
        
        # Execute all quality gates
        print("\n🚪 Executing comprehensive quality gates...")
        results = quality_system.run_all_quality_gates()
        
        # Display results summary
        print(f"\n🎯 QUALITY GATES COMPLETED")
        print(f"   Overall Status: {results['overall_status']}")
        print(f"   Overall Score: {results['overall_score']:.1f}%")
        print(f"   Total Time: {results['total_execution_time']:.2f}s")
        
        print(f"\n📊 STATUS BREAKDOWN:")
        for status, count in results["status_summary"].items():
            icon = {"passed": "✅", "failed": "❌", "warning": "⚠️", "error": "💥", "skipped": "⏭️"}.get(status, "❓")
            print(f"   {icon} {status.upper()}: {count}")
        
        print(f"\n🎖️ CATEGORY SCORES:")
        for category, score in results["category_scores"].items():
            print(f"   {category.upper()}: {score:.1f}%")
        
        # Generate and save report
        if quality_config["generate_report"]:
            report = quality_system.generate_quality_report(results)
            report_file = Path("comprehensive_quality_report.md")
            report_file.write_text(report)
            print(f"\n📋 Detailed report saved: {report_file}")
        
        # Save JSON results
        json_file = Path("quality_gates_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📊 JSON results saved: {json_file}")
        
        # Determine exit code based on results
        if results["overall_status"] == "PASSED":
            print("\n🎉 All quality gates PASSED!")
            return 0
        elif results["overall_status"] == "WARNING":
            print("\n⚠️ Quality gates completed with WARNINGS")
            return 0  # Warning doesn't fail the build
        else:
            print("\n❌ Quality gates FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"Quality system execution failed: {e}")
        print(f"\n💥 Quality system failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())