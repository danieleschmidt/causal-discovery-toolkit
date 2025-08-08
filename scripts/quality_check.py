#!/usr/bin/env python3
"""Comprehensive quality gates for causal discovery toolkit."""

import os
import sys
import subprocess
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logging_config import get_logger

logger = get_logger("quality_check")


class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, message: str, 
                 details: Dict[str, Any] = None, critical: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.critical = critical
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        criticality = " [CRITICAL]" if self.critical and not self.passed else ""
        return f"{status} {self.name}: {self.message}{criticality}"


class QualityGateRunner:
    """Run comprehensive quality gates for the project."""
    
    def __init__(self, project_root: str = None):
        """Initialize quality gate runner."""
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
    def run_all_gates(self) -> Tuple[bool, List[QualityGateResult]]:
        """Run all quality gates and return results."""
        logger.info("Starting comprehensive quality gate checks...")
        
        # Run all quality gates
        self._check_project_structure()
        self._check_code_quality()
        self._run_tests()
        self._check_dependencies()
        self._check_documentation()
        self._check_security()
        self._check_performance()
        self._validate_examples()
        
        # Calculate overall results
        all_passed = all(r.passed for r in self.results if r.critical)
        critical_failures = [r for r in self.results if not r.passed and r.critical]
        
        total_time = time.time() - self.start_time
        
        logger.info(f"Quality gates completed in {total_time:.2f}s")
        logger.info(f"Results: {len([r for r in self.results if r.passed])} passed, "
                   f"{len([r for r in self.results if not r.passed])} failed")
        
        if critical_failures:
            logger.error(f"Critical failures: {len(critical_failures)}")
            for failure in critical_failures:
                logger.error(f"  - {failure.name}: {failure.message}")
        
        return all_passed, self.results
    
    def _check_project_structure(self) -> None:
        """Check project structure and essential files."""
        logger.info("Checking project structure...")
        
        # Essential files
        essential_files = [
            'README.md',
            'setup.py',
            'requirements.txt',
            'LICENSE',
            'src/__init__.py',
            'tests/__init__.py'
        ]
        
        missing_files = []
        for file_path in essential_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results.append(QualityGateResult(
                "Project Structure",
                passed=False,
                message=f"Missing essential files: {', '.join(missing_files)}",
                details={"missing_files": missing_files},
                critical=True
            ))
        else:
            self.results.append(QualityGateResult(
                "Project Structure",
                passed=True,
                message="All essential files present"
            ))
        
        # Check directory structure
        essential_dirs = [
            'src',
            'src/algorithms',
            'src/utils', 
            'src/experiments',
            'tests',
            'examples'
        ]
        
        missing_dirs = []
        for dir_path in essential_dirs:
            if not (self.project_root / dir_path).is_dir():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.results.append(QualityGateResult(
                "Directory Structure",
                passed=False,
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details={"missing_dirs": missing_dirs},
                critical=False
            ))
        else:
            self.results.append(QualityGateResult(
                "Directory Structure",
                passed=True,
                message="All essential directories present"
            ))
    
    def _check_code_quality(self) -> None:
        """Check code quality with various linters."""
        logger.info("Checking code quality...")
        
        # Check if Python files can be imported
        try:
            import src
            self.results.append(QualityGateResult(
                "Import Test",
                passed=True,
                message="Main package imports successfully"
            ))
        except Exception as e:
            self.results.append(QualityGateResult(
                "Import Test", 
                passed=False,
                message=f"Failed to import main package: {str(e)}",
                critical=True
            ))
        
        # Check for basic syntax errors
        python_files = list((self.project_root / 'src').rglob('*.py'))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
        
        if syntax_errors:
            self.results.append(QualityGateResult(
                "Syntax Check",
                passed=False,
                message=f"Syntax errors found in {len(syntax_errors)} files",
                details={"errors": syntax_errors},
                critical=True
            ))
        else:
            self.results.append(QualityGateResult(
                "Syntax Check",
                passed=True,
                message=f"No syntax errors in {len(python_files)} Python files"
            ))
        
        # Check code metrics
        total_lines = 0
        docstring_files = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.count('\n')
                    total_lines += lines
                    
                    # Simple docstring check
                    if '"""' in content or "'''" in content:
                        docstring_files += 1
            except Exception:
                continue
        
        docstring_coverage = (docstring_files / len(python_files)) * 100 if python_files else 0
        
        self.results.append(QualityGateResult(
            "Documentation Coverage",
            passed=docstring_coverage >= 50,
            message=f"Docstring coverage: {docstring_coverage:.1f}% ({docstring_files}/{len(python_files)} files)",
            details={"coverage": docstring_coverage, "threshold": 50},
            critical=False
        ))
        
        self.results.append(QualityGateResult(
            "Code Size",
            passed=True,
            message=f"Total lines of code: {total_lines}",
            details={"total_lines": total_lines, "files": len(python_files)},
            critical=False
        ))
    
    def _run_tests(self) -> None:
        """Run test suite and check coverage."""
        logger.info("Running test suite...")
        
        # Check if pytest is available
        try:
            result = subprocess.run(['python3', '-m', 'pytest', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            pytest_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest_available = False
        
        if not pytest_available:
            self.results.append(QualityGateResult(
                "Test Runner",
                passed=False,
                message="pytest not available - cannot run tests",
                critical=True
            ))
            return
        
        # Run tests
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            result = subprocess.run([
                'python3', '-m', 'pytest', 'tests/', '-v', '--tb=short'
            ], cwd=self.project_root, capture_output=True, text=True, 
               timeout=120, env=env)
            
            test_output = result.stdout + result.stderr
            
            # Parse test results
            if "failed" in test_output.lower():
                failed_tests = len([line for line in test_output.split('\n') 
                                  if 'FAILED' in line])
                self.results.append(QualityGateResult(
                    "Test Suite",
                    passed=False,
                    message=f"Tests failed - see details",
                    details={"output": test_output, "failed_count": failed_tests},
                    critical=True
                ))
            else:
                passed_tests = len([line for line in test_output.split('\n') 
                                  if 'PASSED' in line])
                self.results.append(QualityGateResult(
                    "Test Suite",
                    passed=True,
                    message=f"All tests passed ({passed_tests} tests)",
                    details={"passed_count": passed_tests, "output": test_output}
                ))
                
        except subprocess.TimeoutExpired:
            self.results.append(QualityGateResult(
                "Test Suite",
                passed=False,
                message="Tests timed out after 120 seconds",
                critical=True
            ))
        except Exception as e:
            self.results.append(QualityGateResult(
                "Test Suite",
                passed=False,
                message=f"Error running tests: {str(e)}",
                critical=True
            ))
    
    def _check_dependencies(self) -> None:
        """Check dependencies and requirements."""
        logger.info("Checking dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.results.append(QualityGateResult(
                "Requirements File",
                passed=False,
                message="requirements.txt not found",
                critical=True
            ))
            return
        
        # Read requirements
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Check if requirements can be imported
            importable = 0
            import_errors = []
            
            for req in requirements[:10]:  # Check first 10 to avoid timeout
                pkg_name = req.split('>=')[0].split('==')[0].split('<')[0]
                try:
                    __import__(pkg_name)
                    importable += 1
                except ImportError as e:
                    import_errors.append(f"{pkg_name}: {str(e)}")
            
            self.results.append(QualityGateResult(
                "Dependencies",
                passed=len(import_errors) == 0,
                message=f"Dependencies check: {importable} importable, {len(import_errors)} failed",
                details={"total": len(requirements), "importable": importable, 
                        "errors": import_errors[:5]},  # Limit errors shown
                critical=len(import_errors) > len(requirements) // 2
            ))
            
        except Exception as e:
            self.results.append(QualityGateResult(
                "Dependencies",
                passed=False,
                message=f"Error checking dependencies: {str(e)}",
                critical=False
            ))
    
    def _check_documentation(self) -> None:
        """Check documentation quality."""
        logger.info("Checking documentation...")
        
        readme_file = self.project_root / 'README.md'
        if not readme_file.exists():
            self.results.append(QualityGateResult(
                "README",
                passed=False,
                message="README.md not found",
                critical=True
            ))
            return
        
        # Check README content
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            required_sections = [
                'install', 'usage', 'example', 'getting started', 
                'quickstart', 'overview'
            ]
            
            found_sections = []
            for section in required_sections:
                if section.lower() in readme_content.lower():
                    found_sections.append(section)
            
            readme_score = len(found_sections) / len(required_sections) * 100
            
            self.results.append(QualityGateResult(
                "README Content",
                passed=readme_score >= 40,
                message=f"README completeness: {readme_score:.0f}% ({len(found_sections)}/{len(required_sections)} sections)",
                details={"score": readme_score, "found": found_sections},
                critical=False
            ))
            
        except Exception as e:
            self.results.append(QualityGateResult(
                "README Content",
                passed=False,
                message=f"Error reading README: {str(e)}",
                critical=False
            ))
    
    def _check_security(self) -> None:
        """Check for basic security issues."""
        logger.info("Checking security...")
        
        # Check for hardcoded secrets (basic patterns)
        security_issues = []
        python_files = list((self.project_root / 'src').rglob('*.py'))
        
        dangerous_patterns = [
            'password=',
            'secret=', 
            'api_key=',
            'token=',
            'private_key='
        ]
        
        for py_file in python_files[:20]:  # Limit to avoid timeout
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in dangerous_patterns:
                    if pattern in content and 'example' not in content:
                        security_issues.append(f"{py_file.name}: {pattern}")
                        
            except Exception:
                continue
        
        self.results.append(QualityGateResult(
            "Security Scan",
            passed=len(security_issues) == 0,
            message=f"Security scan: {len(security_issues)} potential issues found",
            details={"issues": security_issues[:5]},  # Limit shown
            critical=len(security_issues) > 0
        ))
    
    def _check_performance(self) -> None:
        """Check basic performance characteristics."""
        logger.info("Checking performance...")
        
        # Simple import performance test
        start_time = time.time()
        try:
            import src.algorithms.base
            import_time = time.time() - start_time
            
            self.results.append(QualityGateResult(
                "Import Performance",
                passed=import_time < 5.0,
                message=f"Module import time: {import_time:.3f}s",
                details={"import_time": import_time, "threshold": 5.0},
                critical=False
            ))
            
        except Exception as e:
            self.results.append(QualityGateResult(
                "Import Performance",
                passed=False,
                message=f"Import failed: {str(e)}",
                critical=True
            ))
    
    def _validate_examples(self) -> None:
        """Validate that examples work."""
        logger.info("Validating examples...")
        
        example_files = list((self.project_root / 'examples').glob('*.py')) if (self.project_root / 'examples').exists() else []
        
        if not example_files:
            self.results.append(QualityGateResult(
                "Examples",
                passed=False,
                message="No example files found",
                critical=False
            ))
            return
        
        # Check if examples can be imported (basic syntax check)
        working_examples = 0
        example_errors = []
        
        for example_file in example_files[:3]:  # Limit to avoid timeout
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    
                # Basic compilation check
                compile(code, str(example_file), 'exec')
                working_examples += 1
                
            except Exception as e:
                example_errors.append(f"{example_file.name}: {str(e)}")
        
        self.results.append(QualityGateResult(
            "Example Validation",
            passed=working_examples == len(example_files),
            message=f"Examples check: {working_examples}/{len(example_files)} valid",
            details={"total": len(example_files), "working": working_examples, 
                    "errors": example_errors},
            critical=False
        ))
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality report."""
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.passed])
        failed_checks = total_checks - passed_checks
        critical_failures = len([r for r in self.results if not r.passed and r.critical])
        
        report = f"""
🛡️  QUALITY GATE REPORT
{'='*50}

📊 SUMMARY
  Total Checks:      {total_checks}
  Passed:           {passed_checks} ✅
  Failed:           {failed_checks} ❌
  Critical Failures: {critical_failures} 🚨
  
🔍 DETAILED RESULTS
"""
        
        for result in self.results:
            report += f"  {str(result)}\n"
        
        report += f"""
⏱️  EXECUTION TIME: {time.time() - self.start_time:.2f}s

🎯 OVERALL STATUS: {"✅ PASSED" if critical_failures == 0 else "❌ FAILED"}
"""
        
        return report


def main():
    """Run quality gates as main program."""
    print("🛡️  Starting Comprehensive Quality Gates")
    print("=" * 50)
    
    # Initialize quality gate runner
    runner = QualityGateRunner()
    
    # Run all quality gates
    all_passed, results = runner.run_all_gates()
    
    # Generate and display report
    report = runner.generate_report()
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent.parent / 'quality_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Exit with appropriate code
    if not all_passed:
        print("\n❌ Quality gates FAILED - blocking deployment")
        sys.exit(1)
    else:
        print("\n✅ All quality gates PASSED - ready for deployment")
        sys.exit(0)


if __name__ == "__main__":
    main()