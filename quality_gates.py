"""Quality gates and validation for the causal discovery toolkit."""

import subprocess
import sys
import os
from pathlib import Path
import time
import json


def run_command(command, description):
    """Run a command and return the result."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(
            command.split(), 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_code_quality():
    """Run code quality checks."""
    print("\n📋 CODE QUALITY CHECKS")
    print("-" * 30)
    
    quality_results = {}
    
    # Check if we can run basic imports
    success, stdout, stderr = run_command(
        "venv/bin/python -c \"import src; print('All imports successful')\"",
        "Testing imports"
    )
    quality_results['imports'] = success
    if success:
        print("✅ All imports successful")
    else:
        print(f"❌ Import issues: {stderr}")
    
    # Test basic functionality
    success, stdout, stderr = run_command(
        "venv/bin/python examples/basic_usage.py",
        "Testing basic functionality"
    )
    quality_results['basic_functionality'] = success
    if success:
        print("✅ Basic functionality test passed")
    else:
        print(f"❌ Basic functionality test failed")
    
    # Test robustness features
    success, stdout, stderr = run_command(
        "venv/bin/python test_robustness.py",
        "Testing robustness features"
    )
    quality_results['robustness'] = success
    if success:
        print("✅ Robustness features test passed")
    else:
        print(f"❌ Robustness features test failed")
    
    # Test performance features
    success, stdout, stderr = run_command(
        "venv/bin/python test_performance_simple.py",
        "Testing performance features"
    )
    quality_results['performance'] = success
    if success:
        print("✅ Performance features test passed")
    else:
        print(f"❌ Performance features test failed")
    
    return quality_results


def check_security():
    """Run security checks."""
    print("\n🔒 SECURITY CHECKS")
    print("-" * 20)
    
    security_results = {}
    
    # Check for common security issues in Python files
    python_files = list(Path('src').rglob('*.py'))
    
    security_issues = []
    dangerous_patterns = [
        'eval(',
        'exec(',
        'subprocess.call(',
        'os.system(',
        '__import__',
        'input(',  # Can be dangerous in some contexts
    ]
    
    for file_path in python_files:
        try:
            content = file_path.read_text()
            for pattern in dangerous_patterns:
                if pattern in content:
                    security_issues.append(f"{file_path}: Found '{pattern}'")
        except Exception:
            continue
    
    if not security_issues:
        print("✅ No obvious security issues found in code")
        security_results['code_scan'] = True
    else:
        print("⚠️  Potential security issues found:")
        for issue in security_issues[:5]:  # Show first 5
            print(f"  - {issue}")
        security_results['code_scan'] = False
    
    # Check for secrets in files
    secrets_patterns = ['password', 'secret', 'api_key', 'token']
    secrets_found = []
    
    for file_path in python_files:
        try:
            content = file_path.read_text().lower()
            for pattern in secrets_patterns:
                if f'{pattern}=' in content or f'"{pattern}"' in content:
                    secrets_found.append(f"{file_path}: Potential secret '{pattern}'")
        except Exception:
            continue
    
    if not secrets_found:
        print("✅ No hardcoded secrets found")
        security_results['secrets_scan'] = True
    else:
        print("⚠️  Potential secrets found:")
        for secret in secrets_found[:3]:
            print(f"  - {secret}")
        security_results['secrets_scan'] = False
    
    return security_results


def check_dependencies():
    """Check dependency security and versions."""
    print("\n📦 DEPENDENCY CHECKS")
    print("-" * 22)
    
    dependency_results = {}
    
    # Check if requirements.txt exists and is readable
    req_file = Path('requirements.txt')
    if req_file.exists():
        try:
            requirements = req_file.read_text().splitlines()
            print(f"✅ Found {len(requirements)} dependencies in requirements.txt")
            dependency_results['requirements_file'] = True
        except Exception as e:
            print(f"❌ Error reading requirements.txt: {e}")
            dependency_results['requirements_file'] = False
    else:
        print("⚠️  No requirements.txt found")
        dependency_results['requirements_file'] = False
    
    # Check for virtual environment
    if Path('venv').exists():
        print("✅ Virtual environment found")
        dependency_results['virtual_env'] = True
    else:
        print("⚠️  No virtual environment found")
        dependency_results['virtual_env'] = False
    
    return dependency_results


def check_documentation():
    """Check documentation completeness."""
    print("\n📚 DOCUMENTATION CHECKS")
    print("-" * 25)
    
    doc_results = {}
    
    # Check for README
    if Path('README.md').exists():
        readme_content = Path('README.md').read_text()
        if len(readme_content) > 500:
            print("✅ Comprehensive README.md found")
            doc_results['readme'] = True
        else:
            print("⚠️  README.md exists but is quite short")
            doc_results['readme'] = False
    else:
        print("❌ No README.md found")
        doc_results['readme'] = False
    
    # Check for docstrings in main modules
    python_files = list(Path('src').rglob('*.py'))
    files_with_docstrings = 0
    
    for file_path in python_files:
        try:
            content = file_path.read_text()
            if '"""' in content and 'def ' in content:
                files_with_docstrings += 1
        except Exception:
            continue
    
    if files_with_docstrings > len(python_files) * 0.7:
        print(f"✅ Good docstring coverage ({files_with_docstrings}/{len(python_files)} files)")
        doc_results['docstrings'] = True
    else:
        print(f"⚠️  Limited docstring coverage ({files_with_docstrings}/{len(python_files)} files)")
        doc_results['docstrings'] = False
    
    # Check for examples
    examples_dir = Path('examples')
    if examples_dir.exists():
        example_files = list(examples_dir.glob('*.py'))
        print(f"✅ Found {len(example_files)} example files")
        doc_results['examples'] = True
    else:
        print("❌ No examples directory found")
        doc_results['examples'] = False
    
    return doc_results


def generate_quality_report(results):
    """Generate a comprehensive quality report."""
    print("\n📊 QUALITY GATE SUMMARY")
    print("=" * 30)
    
    all_categories = ['code_quality', 'security', 'dependencies', 'documentation']
    category_scores = {}
    
    for category, category_results in results.items():
        if category_results:
            passed = sum(1 for v in category_results.values() if v)
            total = len(category_results)
            score = passed / total if total > 0 else 0
            category_scores[category] = score
            
            status = "✅ PASS" if score >= 0.8 else "⚠️  WARN" if score >= 0.6 else "❌ FAIL"
            print(f"{category.upper():15} {status} ({passed}/{total} checks passed)")
        else:
            category_scores[category] = 0
            print(f"{category.upper():15} ❌ FAIL (No data)")
    
    # Overall score
    overall_score = sum(category_scores.values()) / len(category_scores)
    print(f"\nOVERALL SCORE: {overall_score:.1%}")
    
    if overall_score >= 0.8:
        print("🎉 QUALITY GATES PASSED! Ready for production.")
        return True
    elif overall_score >= 0.6:
        print("⚠️  Quality gates partially passed. Review warnings before deployment.")
        return False
    else:
        print("❌ Quality gates failed. Address issues before deployment.")
        return False


def main():
    """Run all quality gates."""
    print("🚀 AUTONOMOUS SDLC - QUALITY GATES")
    print("=" * 40)
    
    start_time = time.time()
    
    # Change to repo directory
    os.chdir(Path(__file__).parent)
    
    results = {}
    
    # Run all quality checks
    results['code_quality'] = check_code_quality()
    results['security'] = check_security()
    results['dependencies'] = check_dependencies()
    results['documentation'] = check_documentation()
    
    # Generate final report
    overall_pass = generate_quality_report(results)
    
    # Save results to file
    report_data = {
        'timestamp': time.time(),
        'execution_time': time.time() - start_time,
        'results': results,
        'overall_pass': overall_pass,
        'summary': f"Quality gates {'PASSED' if overall_pass else 'FAILED'}"
    }
    
    with open('quality_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n⏱️  Quality gates completed in {time.time() - start_time:.1f} seconds")
    print(f"📄 Detailed report saved to quality_report.json")
    
    if overall_pass:
        print("\n🎯 RECOMMENDATION: Proceed with deployment")
    else:
        print("\n⚠️  RECOMMENDATION: Address quality issues before deployment")
    
    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)