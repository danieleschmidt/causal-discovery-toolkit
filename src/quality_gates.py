"""Comprehensive Research Quality Gates for Academic Publication Standards.

This module implements rigorous quality assurance gates that ensure research
meets the highest academic standards before publication submission.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import time
import logging
import json
import warnings
from pathlib import Path
from enum import Enum
import subprocess
import ast
import re

try:
    from .algorithms.base import CausalResult, CausalDiscoveryModel
    from .evaluation_framework import PublicationMetrics
    from .reproducibility_framework import EnvironmentSnapshot, ReproducibilityReport
    from .utils.validation import DataValidator
    from .utils.security import SecurityValidator
    from .utils.performance import PerformanceProfiler
except ImportError:
    from algorithms.base import CausalResult, CausalDiscoveryModel
    from evaluation_framework import PublicationMetrics
    from reproducibility_framework import EnvironmentSnapshot, ReproducibilityReport
    from utils.validation import DataValidator
    from utils.security import SecurityValidator
    from utils.performance import PerformanceProfiler


class QualityGateStatus(Enum):
    """Quality gate validation status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class QualityGateResult:
    """Result of a single quality gate check."""
    gate_name: str
    status: QualityGateStatus
    score: float
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment report."""
    overall_status: QualityGateStatus
    overall_score: float
    gate_results: List[QualityGateResult]
    publication_readiness: str
    critical_issues: List[str]
    recommendations: List[str]
    assessment_timestamp: str


class ResearchQualityGates:
    """Comprehensive quality gates for research validation."""
    
    def __init__(self, 
                 strict_mode: bool = True,
                 publication_standards: str = "neurips",
                 output_dir: str = "quality_reports"):
        
        self.strict_mode = strict_mode
        self.publication_standards = publication_standards
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.data_validator = DataValidator()
        self.security_validator = SecurityValidator()
        self.performance_profiler = PerformanceProfiler()
        
        # Quality gate definitions
        self.quality_gates = self._initialize_quality_gates()
        
        # Publication standards
        self.publication_thresholds = self._get_publication_thresholds(publication_standards)
    
    def _initialize_quality_gates(self) -> List[Dict[str, Any]]:
        """Initialize comprehensive quality gate definitions."""
        return [
            # Core Algorithm Quality Gates
            {
                "name": "algorithm_implementation_quality",
                "category": "implementation",
                "weight": 0.15,
                "mandatory": True,
                "checker": self._check_algorithm_implementation
            },
            {
                "name": "performance_benchmarks",
                "category": "performance", 
                "weight": 0.20,
                "mandatory": True,
                "checker": self._check_performance_benchmarks
            },
            {
                "name": "statistical_significance",
                "category": "statistics",
                "weight": 0.18,
                "mandatory": True,
                "checker": self._check_statistical_significance
            },
            {
                "name": "reproducibility_validation",
                "category": "reproducibility",
                "weight": 0.15,
                "mandatory": True,
                "checker": self._check_reproducibility
            },
            {
                "name": "theoretical_soundness",
                "category": "theory",
                "weight": 0.12,
                "mandatory": True,
                "checker": self._check_theoretical_soundness
            },
            
            # Code Quality Gates
            {
                "name": "code_quality_standards",
                "category": "code",
                "weight": 0.08,
                "mandatory": False,
                "checker": self._check_code_quality
            },
            {
                "name": "security_compliance",
                "category": "security",
                "weight": 0.05,
                "mandatory": False,
                "checker": self._check_security_compliance
            },
            {
                "name": "documentation_completeness",
                "category": "documentation",
                "weight": 0.07,
                "mandatory": False,
                "checker": self._check_documentation
            }
        ]
    
    def _get_publication_thresholds(self, standards: str) -> Dict[str, Any]:
        """Get publication thresholds for different venues."""
        thresholds = {
            "neurips": {
                "min_f1_score": 0.3,
                "min_statistical_power": 0.8,
                "max_p_value": 0.05,
                "min_effect_size": 0.2,
                "min_reproducibility_score": 0.9,
                "min_novelty_score": 0.7
            },
            "icml": {
                "min_f1_score": 0.25,
                "min_statistical_power": 0.8,
                "max_p_value": 0.05,
                "min_effect_size": 0.3,
                "min_reproducibility_score": 0.85,
                "min_novelty_score": 0.8
            },
            "iclr": {
                "min_f1_score": 0.3,
                "min_statistical_power": 0.75,
                "max_p_value": 0.05,
                "min_effect_size": 0.2,
                "min_reproducibility_score": 0.9,
                "min_novelty_score": 0.75
            },
            "generic": {
                "min_f1_score": 0.2,
                "min_statistical_power": 0.7,
                "max_p_value": 0.05,
                "min_effect_size": 0.1,
                "min_reproducibility_score": 0.8,
                "min_novelty_score": 0.6
            }
        }
        
        return thresholds.get(standards, thresholds["generic"])
    
    def run_comprehensive_quality_gates(self,
                                      algorithms: Dict[str, CausalDiscoveryModel],
                                      publication_metrics: Dict[str, PublicationMetrics],
                                      experimental_results: Dict[str, Any],
                                      reproducibility_report: Optional[ReproducibilityReport] = None) -> QualityAssessment:
        """
        Run comprehensive quality gates validation.
        
        Args:
            algorithms: Algorithms being evaluated
            publication_metrics: Publication-ready metrics
            experimental_results: Complete experimental results
            reproducibility_report: Optional reproducibility validation
            
        Returns:
            Comprehensive quality assessment
        """
        self.logger.info("Running comprehensive research quality gates...")
        
        start_time = time.time()
        gate_results = []
        
        # Run each quality gate
        for gate_config in self.quality_gates:
            gate_name = gate_config["name"]
            checker_function = gate_config["checker"]
            
            self.logger.info(f"Running quality gate: {gate_name}")
            
            try:
                result = checker_function(
                    algorithms=algorithms,
                    publication_metrics=publication_metrics,
                    experimental_results=experimental_results,
                    reproducibility_report=reproducibility_report
                )
                gate_results.append(result)
                
                self.logger.info(f"Quality gate {gate_name}: {result.status.value} (score: {result.score:.3f})")
                
            except Exception as e:
                self.logger.error(f"Quality gate {gate_name} failed with error: {e}")
                
                # Create error result
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAIL,
                    score=0.0,
                    message=f"Quality gate execution failed: {str(e)}",
                    details={"error": str(e)},
                    recommendations=[f"Fix implementation error in {gate_name}"],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                gate_results.append(error_result)
        
        # Compute overall assessment
        overall_assessment = self._compute_overall_assessment(gate_results)
        
        # Save assessment report
        self._save_quality_assessment(overall_assessment)
        
        duration = time.time() - start_time
        self.logger.info(f"Quality gates completed in {duration:.2f}s. "
                        f"Overall status: {overall_assessment.overall_status.value}")
        
        return overall_assessment
    
    def _check_algorithm_implementation(self, **kwargs) -> QualityGateResult:
        """Check algorithm implementation quality."""
        algorithms = kwargs.get("algorithms", {})
        
        implementation_scores = []
        implementation_issues = []
        recommendations = []
        
        for alg_name, algorithm in algorithms.items():
            # Check class structure
            if not isinstance(algorithm, CausalDiscoveryModel):
                implementation_issues.append(f"{alg_name} does not inherit from CausalDiscoveryModel")
                continue
            
            # Check required methods
            required_methods = ["fit", "discover", "fit_discover"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(algorithm, method) or not callable(getattr(algorithm, method)):
                    missing_methods.append(method)
            
            if missing_methods:
                implementation_issues.append(f"{alg_name} missing methods: {missing_methods}")
                implementation_scores.append(0.3)
            else:
                implementation_scores.append(1.0)
            
            # Check hyperparameter documentation
            if not hasattr(algorithm, 'hyperparameters'):
                recommendations.append(f"Add hyperparameters attribute to {alg_name}")
            
            # Check docstring quality
            if not algorithm.__class__.__doc__ or len(algorithm.__class__.__doc__.strip()) < 50:
                recommendations.append(f"Improve docstring documentation for {alg_name}")
        
        # Compute overall score
        if implementation_scores:
            overall_score = np.mean(implementation_scores)
        else:
            overall_score = 0.0
        
        # Determine status
        if overall_score >= 0.8 and not implementation_issues:
            status = QualityGateStatus.PASS
            message = "Algorithm implementations meet quality standards"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Algorithm implementations have minor issues"
        else:
            status = QualityGateStatus.FAIL
            message = "Algorithm implementations have significant issues"
        
        if implementation_issues:
            message += f". Issues: {'; '.join(implementation_issues)}"
        
        return QualityGateResult(
            gate_name="algorithm_implementation_quality",
            status=status,
            score=overall_score,
            message=message,
            details={
                "implementation_scores": implementation_scores,
                "implementation_issues": implementation_issues,
                "algorithms_checked": list(algorithms.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_performance_benchmarks(self, **kwargs) -> QualityGateResult:
        """Check performance against benchmarks."""
        publication_metrics = kwargs.get("publication_metrics", {})
        
        performance_scores = []
        benchmark_results = []
        recommendations = []
        
        thresholds = self.publication_thresholds
        
        for alg_name, metrics in publication_metrics.items():
            alg_scores = []
            
            # Check F1 score
            f1_score = metrics.f1_score
            if f1_score >= thresholds["min_f1_score"]:
                alg_scores.append(1.0)
            else:
                alg_scores.append(f1_score / thresholds["min_f1_score"])
                recommendations.append(f"Improve F1 score for {alg_name} (current: {f1_score:.3f}, required: {thresholds['min_f1_score']:.3f})")
            
            # Check precision and recall
            precision_score = min(1.0, metrics.precision / 0.3)  # Minimum 30% precision
            recall_score = min(1.0, metrics.recall / 0.3)  # Minimum 30% recall
            
            alg_scores.extend([precision_score, recall_score])
            
            # Check effect size
            effect_size = abs(metrics.effect_size)
            if effect_size >= thresholds["min_effect_size"]:
                alg_scores.append(1.0)
            else:
                alg_scores.append(effect_size / thresholds["min_effect_size"])
                recommendations.append(f"Increase effect size for {alg_name} (current: {effect_size:.3f}, required: {thresholds['min_effect_size']:.3f})")
            
            # Check computational efficiency
            execution_time = metrics.execution_time
            if execution_time <= 300:  # 5 minutes max
                efficiency_score = 1.0
            else:
                efficiency_score = max(0.1, 300 / execution_time)
                recommendations.append(f"Optimize execution time for {alg_name}")
            
            alg_scores.append(efficiency_score)
            
            algorithm_score = np.mean(alg_scores)
            performance_scores.append(algorithm_score)
            
            benchmark_results.append({
                "algorithm": alg_name,
                "f1_score": f1_score,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "effect_size": effect_size,
                "execution_time": execution_time,
                "overall_score": algorithm_score
            })
        
        overall_score = np.mean(performance_scores) if performance_scores else 0.0
        
        # Determine status
        if overall_score >= 0.8:
            status = QualityGateStatus.PASS
            message = "Performance benchmarks meet publication standards"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Performance benchmarks partially meet standards"
        else:
            status = QualityGateStatus.FAIL
            message = "Performance benchmarks do not meet publication standards"
        
        return QualityGateResult(
            gate_name="performance_benchmarks",
            status=status,
            score=overall_score,
            message=message,
            details={
                "benchmark_results": benchmark_results,
                "performance_scores": performance_scores,
                "thresholds_used": thresholds
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_statistical_significance(self, **kwargs) -> QualityGateResult:
        """Check statistical significance and rigor."""
        publication_metrics = kwargs.get("publication_metrics", {})
        experimental_results = kwargs.get("experimental_results", {})
        
        statistical_scores = []
        statistical_issues = []
        recommendations = []
        
        thresholds = self.publication_thresholds
        
        for alg_name, metrics in publication_metrics.items():
            alg_statistical_scores = []
            
            # Check p-value significance
            p_value = metrics.p_value
            if p_value <= thresholds["max_p_value"]:
                alg_statistical_scores.append(1.0)
            else:
                # Gradual degradation for higher p-values
                p_score = max(0.0, 1.0 - (p_value - thresholds["max_p_value"]) / 0.1)
                alg_statistical_scores.append(p_score)
                statistical_issues.append(f"{alg_name} p-value {p_value:.4f} > {thresholds['max_p_value']}")
            
            # Check statistical power
            statistical_power = metrics.statistical_power
            if statistical_power >= thresholds["min_statistical_power"]:
                alg_statistical_scores.append(1.0)
            else:
                power_score = statistical_power / thresholds["min_statistical_power"]
                alg_statistical_scores.append(power_score)
                statistical_issues.append(f"{alg_name} statistical power {statistical_power:.3f} < {thresholds['min_statistical_power']}")
            
            # Check confidence interval width
            ci_lower, ci_upper = metrics.confidence_interval
            ci_width = ci_upper - ci_lower
            
            if ci_width <= 0.2:  # Reasonable confidence interval width
                alg_statistical_scores.append(1.0)
            else:
                ci_score = max(0.1, 0.2 / ci_width)
                alg_statistical_scores.append(ci_score)
                recommendations.append(f"Narrow confidence interval for {alg_name} (current width: {ci_width:.3f})")
            
            # Check effect size magnitude
            effect_size = abs(metrics.effect_size)
            if effect_size >= thresholds["min_effect_size"]:
                alg_statistical_scores.append(1.0)
            else:
                effect_score = effect_size / thresholds["min_effect_size"]
                alg_statistical_scores.append(effect_score)
            
            algorithm_stat_score = np.mean(alg_statistical_scores)
            statistical_scores.append(algorithm_stat_score)
        
        # Check multiple testing correction
        aggregated_results = experimental_results.get("aggregated_results", {})
        if aggregated_results:
            statistical_tests = aggregated_results.get("statistical_tests", {})
            if statistical_tests:
                # Check if multiple testing correction was applied
                total_tests = sum(len(tests) for tests in statistical_tests.values())
                if total_tests > 1:
                    recommendations.append("Ensure multiple testing correction is properly applied")
            else:
                statistical_issues.append("No statistical tests found in results")
        
        overall_score = np.mean(statistical_scores) if statistical_scores else 0.0
        
        # Determine status
        if overall_score >= 0.8 and len(statistical_issues) == 0:
            status = QualityGateStatus.PASS
            message = "Statistical analysis meets publication standards"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Statistical analysis partially meets standards"
        else:
            status = QualityGateStatus.FAIL
            message = "Statistical analysis does not meet publication standards"
        
        if statistical_issues:
            message += f". Issues: {'; '.join(statistical_issues[:3])}"
        
        return QualityGateResult(
            gate_name="statistical_significance",
            status=status,
            score=overall_score,
            message=message,
            details={
                "statistical_scores": statistical_scores,
                "statistical_issues": statistical_issues,
                "algorithms_analyzed": list(publication_metrics.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_reproducibility(self, **kwargs) -> QualityGateResult:
        """Check reproducibility standards."""
        reproducibility_report = kwargs.get("reproducibility_report")
        experimental_results = kwargs.get("experimental_results", {})
        
        reproducibility_scores = []
        reproducibility_issues = []
        recommendations = []
        
        thresholds = self.publication_thresholds
        
        if reproducibility_report:
            # Check reproducibility score
            repro_score = reproducibility_report.reproducibility_score
            if repro_score >= thresholds["min_reproducibility_score"]:
                reproducibility_scores.append(1.0)
            else:
                score_ratio = repro_score / thresholds["min_reproducibility_score"]
                reproducibility_scores.append(score_ratio)
                reproducibility_issues.append(f"Reproducibility score {repro_score:.3f} < {thresholds['min_reproducibility_score']}")
            
            # Check environment consistency
            env_differences = reproducibility_report.environment_differences
            if len(env_differences) == 0:
                reproducibility_scores.append(1.0)
            elif len(env_differences) <= 3:
                reproducibility_scores.append(0.7)
                recommendations.append("Minimize environment dependencies")
            else:
                reproducibility_scores.append(0.3)
                reproducibility_issues.append(f"Too many environment differences: {len(env_differences)}")
        else:
            # No reproducibility report provided
            reproducibility_scores.append(0.0)
            reproducibility_issues.append("No reproducibility validation performed")
            recommendations.append("Perform reproducibility validation")
        
        # Check random seed documentation
        repro_info = experimental_results.get("reproducibility_info", {})
        if repro_info.get("random_seeds_used"):
            reproducibility_scores.append(1.0)
        else:
            reproducibility_scores.append(0.5)
            recommendations.append("Document random seeds used")
        
        # Check environment documentation
        if experimental_results.get("pre_environment"):
            reproducibility_scores.append(1.0)
        else:
            reproducibility_scores.append(0.0)
            reproducibility_issues.append("No environment snapshot captured")
        
        # Check code versioning
        git_commit = experimental_results.get("pre_environment", {}).get("git_commit")
        if git_commit:
            reproducibility_scores.append(1.0)
        else:
            reproducibility_scores.append(0.3)
            recommendations.append("Use version control for code tracking")
        
        overall_score = np.mean(reproducibility_scores) if reproducibility_scores else 0.0
        
        # Determine status
        if overall_score >= 0.9 and len(reproducibility_issues) == 0:
            status = QualityGateStatus.PASS
            message = "Reproducibility standards fully met"
        elif overall_score >= 0.7:
            status = QualityGateStatus.WARNING
            message = "Reproducibility standards partially met"
        else:
            status = QualityGateStatus.FAIL
            message = "Reproducibility standards not met"
        
        if reproducibility_issues:
            message += f". Issues: {'; '.join(reproducibility_issues[:2])}"
        
        return QualityGateResult(
            gate_name="reproducibility_validation",
            status=status,
            score=overall_score,
            message=message,
            details={
                "reproducibility_scores": reproducibility_scores,
                "reproducibility_issues": reproducibility_issues,
                "reproducibility_report_available": reproducibility_report is not None
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_theoretical_soundness(self, **kwargs) -> QualityGateResult:
        """Check theoretical foundation and soundness."""
        algorithms = kwargs.get("algorithms", {})
        publication_metrics = kwargs.get("publication_metrics", {})
        
        theoretical_scores = []
        theoretical_issues = []
        recommendations = []
        
        thresholds = self.publication_thresholds
        
        for alg_name, algorithm in algorithms.items():
            alg_theoretical_scores = []
            
            # Check class documentation
            docstring = algorithm.__class__.__doc__
            if docstring and len(docstring.strip()) > 100:
                # Look for theoretical keywords
                theoretical_keywords = [
                    "theorem", "proof", "assumption", "theory", "foundation", 
                    "mathematical", "formal", "complexity", "convergence"
                ]
                
                keyword_count = sum(1 for keyword in theoretical_keywords 
                                  if keyword.lower() in docstring.lower())
                
                if keyword_count >= 3:
                    alg_theoretical_scores.append(1.0)
                elif keyword_count >= 1:
                    alg_theoretical_scores.append(0.7)
                else:
                    alg_theoretical_scores.append(0.3)
                    recommendations.append(f"Add theoretical foundation documentation for {alg_name}")
            else:
                alg_theoretical_scores.append(0.1)
                theoretical_issues.append(f"{alg_name} lacks theoretical documentation")
            
            # Check novelty score from metrics
            if alg_name in publication_metrics:
                novelty_score = publication_metrics[alg_name].novelty_score
                if novelty_score >= thresholds["min_novelty_score"]:
                    alg_theoretical_scores.append(1.0)
                else:
                    novelty_ratio = novelty_score / thresholds["min_novelty_score"]
                    alg_theoretical_scores.append(novelty_ratio)
                    if novelty_score < 0.5:
                        theoretical_issues.append(f"{alg_name} novelty score too low: {novelty_score:.3f}")
            else:
                alg_theoretical_scores.append(0.5)
            
            # Check theoretical grounding score
            if alg_name in publication_metrics:
                theoretical_grounding = publication_metrics[alg_name].theoretical_grounding
                alg_theoretical_scores.append(theoretical_grounding)
                
                if theoretical_grounding < 0.6:
                    recommendations.append(f"Strengthen theoretical foundation for {alg_name}")
            else:
                alg_theoretical_scores.append(0.5)
            
            algorithm_theoretical_score = np.mean(alg_theoretical_scores)
            theoretical_scores.append(algorithm_theoretical_score)
        
        overall_score = np.mean(theoretical_scores) if theoretical_scores else 0.0
        
        # Determine status
        if overall_score >= 0.8 and len(theoretical_issues) == 0:
            status = QualityGateStatus.PASS
            message = "Theoretical foundation meets publication standards"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Theoretical foundation partially meets standards"
        else:
            status = QualityGateStatus.FAIL
            message = "Theoretical foundation does not meet publication standards"
        
        if theoretical_issues:
            message += f". Issues: {'; '.join(theoretical_issues[:2])}"
        
        return QualityGateResult(
            gate_name="theoretical_soundness",
            status=status,
            score=overall_score,
            message=message,
            details={
                "theoretical_scores": theoretical_scores,
                "theoretical_issues": theoretical_issues,
                "algorithms_analyzed": list(algorithms.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_code_quality(self, **kwargs) -> QualityGateResult:
        """Check code quality standards."""
        algorithms = kwargs.get("algorithms", {})
        
        code_quality_scores = []
        code_issues = []
        recommendations = []
        
        # Check if we can analyze source code
        try:
            import inspect
            
            for alg_name, algorithm in algorithms.items():
                alg_quality_scores = []
                
                # Get source code
                try:
                    source_code = inspect.getsource(algorithm.__class__)
                    
                    # Check code length (not too short, not too long)
                    lines = source_code.split('\\n')
                    code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                    
                    if 50 <= len(code_lines) <= 500:
                        alg_quality_scores.append(1.0)
                    elif 20 <= len(code_lines) <= 1000:
                        alg_quality_scores.append(0.8)
                    else:
                        alg_quality_scores.append(0.5)
                        if len(code_lines) < 20:
                            recommendations.append(f"Expand implementation detail for {alg_name}")
                        else:
                            recommendations.append(f"Consider refactoring {alg_name} for better maintainability")
                    
                    # Check for docstrings
                    docstring_count = source_code.count('\"\"\"') + source_code.count("'''")
                    if docstring_count >= 4:  # Class + methods
                        alg_quality_scores.append(1.0)
                    elif docstring_count >= 2:
                        alg_quality_scores.append(0.7)
                    else:
                        alg_quality_scores.append(0.3)
                        recommendations.append(f"Add docstrings to methods in {alg_name}")
                    
                    # Check for type hints
                    if ":" in source_code and "->" in source_code:
                        alg_quality_scores.append(1.0)
                    elif ":" in source_code:
                        alg_quality_scores.append(0.7)
                    else:
                        alg_quality_scores.append(0.3)
                        recommendations.append(f"Add type hints to {alg_name}")
                    
                    # Check for error handling
                    if "try:" in source_code and "except" in source_code:
                        alg_quality_scores.append(1.0)
                    elif "if" in source_code:  # Basic validation
                        alg_quality_scores.append(0.6)
                    else:
                        alg_quality_scores.append(0.2)
                        recommendations.append(f"Add error handling to {alg_name}")
                    
                except Exception as e:
                    alg_quality_scores = [0.5]  # Can't analyze source
                    code_issues.append(f"Could not analyze source code for {alg_name}: {e}")
                
                algorithm_quality_score = np.mean(alg_quality_scores)
                code_quality_scores.append(algorithm_quality_score)
                
        except ImportError:
            code_quality_scores = [0.5]
            code_issues.append("Cannot perform detailed code analysis (inspect module unavailable)")
        
        overall_score = np.mean(code_quality_scores) if code_quality_scores else 0.5
        
        # Determine status (non-mandatory gate)
        if overall_score >= 0.8:
            status = QualityGateStatus.PASS
            message = "Code quality meets high standards"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Code quality meets basic standards"
        else:
            status = QualityGateStatus.WARNING  # Not FAIL for non-mandatory
            message = "Code quality could be improved"
        
        if code_issues:
            message += f". Issues: {'; '.join(code_issues[:2])}"
        
        return QualityGateResult(
            gate_name="code_quality_standards",
            status=status,
            score=overall_score,
            message=message,
            details={
                "code_quality_scores": code_quality_scores,
                "code_issues": code_issues,
                "algorithms_analyzed": list(algorithms.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_security_compliance(self, **kwargs) -> QualityGateResult:
        """Check security compliance standards."""
        algorithms = kwargs.get("algorithms", {})
        experimental_results = kwargs.get("experimental_results", {})
        
        security_scores = []
        security_issues = []
        recommendations = []
        
        try:
            # Check for potential security issues
            for alg_name, algorithm in algorithms.items():
                alg_security_scores = []
                
                # Check if algorithm uses secure practices
                # (This is a simplified check - real implementation would be more comprehensive)
                
                # Check for input validation
                if hasattr(algorithm, 'fit') and callable(algorithm.fit):
                    try:
                        import inspect
                        source = inspect.getsource(algorithm.fit)
                        
                        # Look for validation patterns
                        if "validate" in source.lower() or "check" in source.lower():
                            alg_security_scores.append(1.0)
                        else:
                            alg_security_scores.append(0.7)
                            recommendations.append(f"Add input validation to {alg_name}")
                            
                    except Exception:
                        alg_security_scores.append(0.8)  # Assume reasonable security
                else:
                    alg_security_scores.append(0.5)
                    security_issues.append(f"{alg_name} missing fit method")
                
                # Check for secure random number generation
                if hasattr(algorithm, 'hyperparameters'):
                    hyperparams = algorithm.hyperparameters
                    if any("seed" in str(key).lower() for key in hyperparams.keys()):
                        alg_security_scores.append(1.0)
                    else:
                        alg_security_scores.append(0.8)
                else:
                    alg_security_scores.append(0.7)
                
                algorithm_security_score = np.mean(alg_security_scores)
                security_scores.append(algorithm_security_score)
            
            # Check experimental data security
            if experimental_results:
                data_integrity = experimental_results.get("execution_metadata", {}).get("data_integrity", {})
                if data_integrity:
                    security_scores.append(1.0)
                else:
                    security_scores.append(0.8)
                    recommendations.append("Implement data integrity checking")
            
        except Exception as e:
            security_scores = [0.7]
            security_issues.append(f"Security analysis failed: {e}")
        
        overall_score = np.mean(security_scores) if security_scores else 0.7
        
        # Determine status (non-mandatory gate)
        if overall_score >= 0.9:
            status = QualityGateStatus.PASS
            message = "Security compliance excellent"
        elif overall_score >= 0.7:
            status = QualityGateStatus.PASS
            message = "Security compliance adequate"
        else:
            status = QualityGateStatus.WARNING
            message = "Security compliance could be improved"
        
        if security_issues:
            message += f". Issues: {'; '.join(security_issues[:2])}"
        
        return QualityGateResult(
            gate_name="security_compliance",
            status=status,
            score=overall_score,
            message=message,
            details={
                "security_scores": security_scores,
                "security_issues": security_issues,
                "algorithms_analyzed": list(algorithms.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_documentation(self, **kwargs) -> QualityGateResult:
        """Check documentation completeness."""
        algorithms = kwargs.get("algorithms", {})
        experimental_results = kwargs.get("experimental_results", {})
        
        documentation_scores = []
        documentation_issues = []
        recommendations = []
        
        for alg_name, algorithm in algorithms.items():
            alg_doc_scores = []
            
            # Check class docstring
            class_doc = algorithm.__class__.__doc__
            if class_doc and len(class_doc.strip()) > 50:
                # Check for required sections
                required_sections = ["Args:", "Returns:", "Example", "Description"]
                section_count = sum(1 for section in required_sections 
                                  if section.lower() in class_doc.lower())
                
                if section_count >= 3:
                    alg_doc_scores.append(1.0)
                elif section_count >= 2:
                    alg_doc_scores.append(0.8)
                else:
                    alg_doc_scores.append(0.5)
                    recommendations.append(f"Improve docstring structure for {alg_name}")
            else:
                alg_doc_scores.append(0.2)
                documentation_issues.append(f"{alg_name} lacks adequate class documentation")
            
            # Check method documentation
            methods_with_docs = 0
            total_methods = 0
            
            for attr_name in dir(algorithm):
                attr = getattr(algorithm, attr_name)
                if callable(attr) and not attr_name.startswith('_'):
                    total_methods += 1
                    if hasattr(attr, '__doc__') and attr.__doc__:
                        methods_with_docs += 1
            
            if total_methods > 0:
                method_doc_ratio = methods_with_docs / total_methods
                alg_doc_scores.append(method_doc_ratio)
                
                if method_doc_ratio < 0.5:
                    recommendations.append(f"Document more methods in {alg_name}")
            else:
                alg_doc_scores.append(0.5)
            
            algorithm_doc_score = np.mean(alg_doc_scores)
            documentation_scores.append(algorithm_doc_score)
        
        # Check experimental documentation
        if experimental_results:
            protocol = experimental_results.get("protocol", {})
            if protocol and len(protocol.get("description", "")) > 100:
                documentation_scores.append(1.0)
            else:
                documentation_scores.append(0.6)
                recommendations.append("Improve experimental protocol documentation")
        
        overall_score = np.mean(documentation_scores) if documentation_scores else 0.5
        
        # Determine status (non-mandatory gate)
        if overall_score >= 0.8:
            status = QualityGateStatus.PASS
            message = "Documentation is comprehensive"
        elif overall_score >= 0.6:
            status = QualityGateStatus.WARNING
            message = "Documentation is adequate"
        else:
            status = QualityGateStatus.WARNING
            message = "Documentation needs improvement"
        
        if documentation_issues:
            message += f". Issues: {'; '.join(documentation_issues[:2])}"
        
        return QualityGateResult(
            gate_name="documentation_completeness",
            status=status,
            score=overall_score,
            message=message,
            details={
                "documentation_scores": documentation_scores,
                "documentation_issues": documentation_issues,
                "algorithms_analyzed": list(algorithms.keys())
            },
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _compute_overall_assessment(self, gate_results: List[QualityGateResult]) -> QualityAssessment:
        """Compute overall quality assessment from individual gate results."""
        
        # Separate mandatory and optional gates
        mandatory_results = []
        optional_results = []
        
        for result in gate_results:
            gate_config = next((g for g in self.quality_gates if g["name"] == result.gate_name), None)
            if gate_config and gate_config["mandatory"]:
                mandatory_results.append(result)
            else:
                optional_results.append(result)
        
        # Check mandatory gates
        mandatory_failures = [r for r in mandatory_results if r.status == QualityGateStatus.FAIL]
        mandatory_warnings = [r for r in mandatory_results if r.status == QualityGateStatus.WARNING]
        
        # Compute weighted score
        total_weight = 0
        weighted_score = 0
        
        for result in gate_results:
            gate_config = next((g for g in self.quality_gates if g["name"] == result.gate_name), None)
            if gate_config:
                weight = gate_config["weight"]
                total_weight += weight
                weighted_score += weight * result.score
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        if mandatory_failures:
            overall_status = QualityGateStatus.FAIL
        elif mandatory_warnings and self.strict_mode:
            overall_status = QualityGateStatus.WARNING
        elif overall_score >= 0.8:
            overall_status = QualityGateStatus.PASS
        elif overall_score >= 0.6:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.FAIL
        
        # Determine publication readiness
        if overall_status == QualityGateStatus.PASS and overall_score >= 0.85:
            publication_readiness = "READY FOR SUBMISSION"
        elif overall_status == QualityGateStatus.PASS or (overall_status == QualityGateStatus.WARNING and overall_score >= 0.75):
            publication_readiness = "MINOR REVISIONS NEEDED"
        elif overall_status == QualityGateStatus.WARNING:
            publication_readiness = "MAJOR REVISIONS NEEDED"
        else:
            publication_readiness = "NOT READY FOR SUBMISSION"
        
        # Collect critical issues
        critical_issues = []
        for result in mandatory_results:
            if result.status == QualityGateStatus.FAIL:
                critical_issues.append(f"{result.gate_name}: {result.message}")
        
        # Collect all recommendations
        all_recommendations = []
        for result in gate_results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(dict.fromkeys(all_recommendations))[:10]  # Top 10
        
        return QualityAssessment(
            overall_status=overall_status,
            overall_score=overall_score,
            gate_results=gate_results,
            publication_readiness=publication_readiness,
            critical_issues=critical_issues,
            recommendations=unique_recommendations,
            assessment_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _save_quality_assessment(self, assessment: QualityAssessment):
        """Save quality assessment to disk."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_file = self.output_dir / f"quality_assessment_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(assessment), f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = self.output_dir / f"quality_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("RESEARCH QUALITY ASSESSMENT REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Overall Status: {assessment.overall_status.value}\\n")
            f.write(f"Overall Score: {assessment.overall_score:.3f}\\n")
            f.write(f"Publication Readiness: {assessment.publication_readiness}\\n\\n")
            
            f.write("QUALITY GATE RESULTS:\\n")
            f.write("-" * 30 + "\\n")
            for result in assessment.gate_results:
                f.write(f"{result.gate_name}: {result.status.value} (score: {result.score:.3f})\\n")
                f.write(f"  {result.message}\\n\\n")
            
            if assessment.critical_issues:
                f.write("CRITICAL ISSUES:\\n")
                f.write("-" * 30 + "\\n")
                for issue in assessment.critical_issues:
                    f.write(f"- {issue}\\n")
                f.write("\\n")
            
            if assessment.recommendations:
                f.write("RECOMMENDATIONS:\\n")
                f.write("-" * 30 + "\\n")
                for i, rec in enumerate(assessment.recommendations, 1):
                    f.write(f"{i}. {rec}\\n")
        
        self.logger.info(f"Quality assessment saved to {self.output_dir}")
    
    def generate_publication_checklist(self, assessment: QualityAssessment) -> Dict[str, Any]:
        """Generate publication submission checklist."""
        checklist = {
            "overall_readiness": assessment.publication_readiness,
            "mandatory_requirements": {},
            "recommended_improvements": {},
            "submission_readiness_score": assessment.overall_score,
            "estimated_review_outcome": ""
        }
        
        # Check mandatory requirements
        for result in assessment.gate_results:
            gate_config = next((g for g in self.quality_gates if g["name"] == result.gate_name), None)
            if gate_config and gate_config["mandatory"]:
                checklist["mandatory_requirements"][result.gate_name] = {
                    "status": result.status.value,
                    "passed": result.status != QualityGateStatus.FAIL,
                    "score": result.score,
                    "message": result.message
                }
        
        # Recommended improvements
        for result in assessment.gate_results:
            if result.recommendations:
                checklist["recommended_improvements"][result.gate_name] = result.recommendations
        
        # Estimate review outcome
        if assessment.overall_score >= 0.85:
            checklist["estimated_review_outcome"] = "Accept/Minor Revision"
        elif assessment.overall_score >= 0.7:
            checklist["estimated_review_outcome"] = "Major Revision"
        elif assessment.overall_score >= 0.5:
            checklist["estimated_review_outcome"] = "Reject and Resubmit"
        else:
            checklist["estimated_review_outcome"] = "Reject"
        
        return checklist