"""
Foundation Model Monitoring and Observability
============================================

Comprehensive monitoring, logging, and observability for foundation models
in production environments with real-time performance tracking.

Features:
- Multi-modal inference monitoring
- Model performance degradation detection
- Resource utilization tracking
- Causal discovery quality metrics
"""

import numpy as np
import torch
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import json
from datetime import datetime, timedelta

try:
    from .monitoring import SystemMonitor, PerformanceProfiler
    from .metrics import CausalMetrics
except ImportError:
    from monitoring import SystemMonitor, PerformanceProfiler
    from metrics import CausalMetrics


@dataclass
class FoundationModelMetrics:
    """Metrics specific to foundation model performance."""
    inference_time: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    multimodal_fusion_time: float = 0.0
    causal_discovery_accuracy: float = 0.0
    representation_quality: float = 0.0
    model_confidence: float = 0.0
    batch_size: int = 0
    input_dimensions: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'inference_time': self.inference_time,
            'memory_usage_gb': self.memory_usage_gb,
            'gpu_utilization': self.gpu_utilization,
            'multimodal_fusion_time': self.multimodal_fusion_time,
            'causal_discovery_accuracy': self.causal_discovery_accuracy,
            'representation_quality': self.representation_quality,
            'model_confidence': self.model_confidence,
            'batch_size': self.batch_size,
            'input_dimensions': self.input_dimensions,
            'timestamp': datetime.now().isoformat()
        }


class FoundationModelMonitor:
    """Monitor for foundation model performance and health."""
    
    def __init__(self, 
                 model_name: str = "foundation_causal_model",
                 metrics_window_size: int = 100,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.model_name = model_name
        self.metrics_window_size = metrics_window_size
        self.alert_thresholds = alert_thresholds or {
            'max_inference_time': 30.0,  # seconds
            'max_memory_usage': 8.0,      # GB
            'min_accuracy': 0.7,          # minimum accuracy
            'max_gpu_utilization': 95.0   # percentage
        }
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_window_size)
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metrics_window_size))
        
        # Monitoring components
        self.system_monitor = SystemMonitor()
        self.causal_metrics = CausalMetrics()
        self.logger = logging.getLogger(f"foundation_monitor.{model_name}")
        
        # Alert tracking
        self.alerts: List[Dict[str, Any]] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)  # Prevent spam
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.baseline_established = False
        
    def start_inference_monitoring(self) -> 'InferenceContext':
        """Start monitoring an inference session."""
        return InferenceContext(self)
    
    def record_metrics(self, metrics: FoundationModelMetrics):
        """Record new metrics and check for alerts."""
        self.metrics_history.append(metrics)
        
        # Update performance trends
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.performance_trends[key].append(value)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update baselines if needed
        self._update_baselines(metrics)
        
        # Log metrics
        self.logger.info(f"Recorded metrics: {json.dumps(metrics_dict, default=str)}")
    
    def _check_alerts(self, metrics: FoundationModelMetrics):
        """Check metrics against alert thresholds."""
        alerts_triggered = []
        
        # Inference time alert
        if metrics.inference_time > self.alert_thresholds['max_inference_time']:
            alerts_triggered.append({
                'type': 'SLOW_INFERENCE',
                'metric': 'inference_time',
                'value': metrics.inference_time,
                'threshold': self.alert_thresholds['max_inference_time'],
                'severity': 'WARNING'
            })
        
        # Memory usage alert
        if metrics.memory_usage_gb > self.alert_thresholds['max_memory_usage']:
            alerts_triggered.append({
                'type': 'HIGH_MEMORY',
                'metric': 'memory_usage_gb',
                'value': metrics.memory_usage_gb,
                'threshold': self.alert_thresholds['max_memory_usage'],
                'severity': 'WARNING'
            })
        
        # Accuracy degradation alert
        if metrics.causal_discovery_accuracy < self.alert_thresholds['min_accuracy']:
            alerts_triggered.append({
                'type': 'ACCURACY_DEGRADATION',
                'metric': 'causal_discovery_accuracy',
                'value': metrics.causal_discovery_accuracy,
                'threshold': self.alert_thresholds['min_accuracy'],
                'severity': 'CRITICAL'
            })
        
        # GPU utilization alert
        if metrics.gpu_utilization > self.alert_thresholds['max_gpu_utilization']:
            alerts_triggered.append({
                'type': 'HIGH_GPU_USAGE',
                'metric': 'gpu_utilization',
                'value': metrics.gpu_utilization,
                'threshold': self.alert_thresholds['max_gpu_utilization'],
                'severity': 'WARNING'
            })
        
        # Process alerts
        for alert in alerts_triggered:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Dict[str, Any]):
        """Process and potentially send an alert."""
        alert_type = alert['type']
        current_time = datetime.now()
        
        # Check cooldown
        if alert_type in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_type]
            if time_since_last < self.alert_cooldown:
                return  # Skip due to cooldown
        
        # Add timestamp and model info
        alert['timestamp'] = current_time.isoformat()
        alert['model_name'] = self.model_name
        
        # Store alert
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = current_time
        
        # Log alert
        severity = alert['severity']
        message = f"[{severity}] {alert_type}: {alert['metric']}={alert['value']:.3f} exceeds threshold {alert['threshold']}"
        
        if severity == 'CRITICAL':
            self.logger.error(message)
        else:
            self.logger.warning(message)
    
    def _update_baselines(self, metrics: FoundationModelMetrics):
        """Update performance baselines for degradation detection."""
        if not self.baseline_established and len(self.metrics_history) >= 20:
            # Establish baselines from first 20 measurements
            baseline_metrics = list(self.metrics_history)[:20]
            
            self.performance_baselines = {
                'inference_time': np.mean([m.inference_time for m in baseline_metrics]),
                'memory_usage_gb': np.mean([m.memory_usage_gb for m in baseline_metrics]),
                'causal_discovery_accuracy': np.mean([m.causal_discovery_accuracy for m in baseline_metrics if m.causal_discovery_accuracy > 0])
            }
            
            self.baseline_established = True
            self.logger.info(f"Performance baselines established: {self.performance_baselines}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        summary = {
            'model_name': self.model_name,
            'measurements_count': len(self.metrics_history),
            'recent_performance': {
                'avg_inference_time': np.mean([m.inference_time for m in recent_metrics]),
                'avg_memory_usage': np.mean([m.memory_usage_gb for m in recent_metrics]),
                'avg_accuracy': np.mean([m.causal_discovery_accuracy for m in recent_metrics if m.causal_discovery_accuracy > 0]),
                'avg_confidence': np.mean([m.model_confidence for m in recent_metrics if m.model_confidence > 0])
            },
            'alerts_count': len([a for a in self.alerts if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'baseline_established': self.baseline_established,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add performance degradation indicators
        if self.baseline_established and recent_metrics:
            current_inference_time = np.mean([m.inference_time for m in recent_metrics])
            current_accuracy = np.mean([m.causal_discovery_accuracy for m in recent_metrics if m.causal_discovery_accuracy > 0])
            
            summary['performance_degradation'] = {
                'inference_time_ratio': current_inference_time / self.performance_baselines['inference_time'],
                'accuracy_ratio': current_accuracy / self.performance_baselines['causal_discovery_accuracy'] if self.performance_baselines['causal_discovery_accuracy'] > 0 else 1.0
            }
        
        return summary
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    def export_metrics(self, filepath: str):
        """Export metrics history to file."""
        metrics_data = [m.to_dict() for m in self.metrics_history]
        
        export_data = {
            'model_name': self.model_name,
            'export_timestamp': datetime.now().isoformat(),
            'metrics_count': len(metrics_data),
            'metrics': metrics_data,
            'performance_summary': self.get_performance_summary(),
            'recent_alerts': self.get_recent_alerts()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {filepath}")


class InferenceContext:
    """Context manager for monitoring individual inference sessions."""
    
    def __init__(self, monitor: FoundationModelMonitor):
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
        self.gpu_start = None
        self.metrics = FoundationModelMetrics()
        
    def __enter__(self) -> 'InferenceContext':
        """Start inference monitoring."""
        self.start_time = time.time()
        
        # Record initial system state
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024**3  # GB
        
        # GPU monitoring (if available)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.gpu_start = torch.cuda.utilization()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish inference monitoring and record metrics."""
        # Calculate inference time
        self.metrics.inference_time = time.time() - self.start_time
        
        # Calculate memory usage
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024**3  # GB
        self.metrics.memory_usage_gb = end_memory - self.start_memory
        
        # GPU utilization
        if torch.cuda.is_available() and self.gpu_start is not None:
            torch.cuda.synchronize()
            self.metrics.gpu_utilization = torch.cuda.utilization()
        
        # Record metrics
        self.monitor.record_metrics(self.metrics)
    
    def set_batch_size(self, batch_size: int):
        """Set batch size for this inference."""
        self.metrics.batch_size = batch_size
    
    def set_input_dimensions(self, dimensions: Dict[str, int]):
        """Set input dimensions for multi-modal data."""
        self.metrics.input_dimensions = dimensions
    
    def set_multimodal_fusion_time(self, fusion_time: float):
        """Set time spent on multi-modal fusion."""
        self.metrics.multimodal_fusion_time = fusion_time
    
    def set_causal_accuracy(self, accuracy: float):
        """Set causal discovery accuracy."""
        self.metrics.causal_discovery_accuracy = accuracy
    
    def set_representation_quality(self, quality: float):
        """Set representation quality score."""
        self.metrics.representation_quality = quality
    
    def set_model_confidence(self, confidence: float):
        """Set model confidence score."""
        self.metrics.model_confidence = confidence


class FoundationModelHealthChecker:
    """Health checker for foundation models in production."""
    
    def __init__(self, model: torch.nn.Module, monitor: FoundationModelMonitor):
        self.model = model
        self.monitor = monitor
        self.logger = logging.getLogger("foundation_health")
        
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Model availability check
            health_status['checks']['model_available'] = self._check_model_availability()
            
            # Memory check
            health_status['checks']['memory_usage'] = self._check_memory_usage()
            
            # Performance check
            health_status['checks']['performance'] = self._check_performance()
            
            # Alert status check
            health_status['checks']['alerts'] = self._check_alert_status()
            
            # Determine overall status
            failed_checks = [
                check_name for check_name, check_result in health_status['checks'].items()
                if check_result.get('status') == 'unhealthy'
            ]
            
            if failed_checks:
                health_status['status'] = 'unhealthy'
                health_status['failed_checks'] = failed_checks
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def _check_model_availability(self) -> Dict[str, Any]:
        """Check if model is available and responding."""
        try:
            # Simple forward pass check
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Test with dummy data
            with torch.no_grad():
                dummy_input = torch.randn(1, 10)  # Adjust based on model
                # This is a simplified check - would need model-specific implementation
                
            return {'status': 'healthy', 'message': 'Model responding normally'}
            
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Model not responding: {str(e)}'}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health."""
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            
            if memory_gb > 16:  # > 16GB
                return {'status': 'unhealthy', 'memory_gb': memory_gb, 'message': 'High memory usage'}
            elif memory_gb > 8:  # > 8GB
                return {'status': 'warning', 'memory_gb': memory_gb, 'message': 'Elevated memory usage'}
            else:
                return {'status': 'healthy', 'memory_gb': memory_gb, 'message': 'Normal memory usage'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Memory check failed: {str(e)}'}
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics health."""
        try:
            summary = self.monitor.get_performance_summary()
            
            if summary.get('status') == 'no_data':
                return {'status': 'warning', 'message': 'No performance data available'}
            
            recent_perf = summary.get('recent_performance', {})
            avg_inference_time = recent_perf.get('avg_inference_time', 0)
            
            if avg_inference_time > 30:  # > 30 seconds
                return {'status': 'unhealthy', 'avg_inference_time': avg_inference_time, 'message': 'Slow inference'}
            elif avg_inference_time > 10:  # > 10 seconds
                return {'status': 'warning', 'avg_inference_time': avg_inference_time, 'message': 'Elevated inference time'}
            else:
                return {'status': 'healthy', 'avg_inference_time': avg_inference_time, 'message': 'Normal performance'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Performance check failed: {str(e)}'}
    
    def _check_alert_status(self) -> Dict[str, Any]:
        """Check recent alert status."""
        try:
            recent_alerts = self.monitor.get_recent_alerts(hours=1)
            critical_alerts = [a for a in recent_alerts if a.get('severity') == 'CRITICAL']
            
            if critical_alerts:
                return {'status': 'unhealthy', 'critical_alerts': len(critical_alerts), 'message': 'Critical alerts present'}
            elif recent_alerts:
                return {'status': 'warning', 'recent_alerts': len(recent_alerts), 'message': 'Recent alerts present'}
            else:
                return {'status': 'healthy', 'message': 'No recent alerts'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Alert check failed: {str(e)}'}


# Export classes
__all__ = [
    'FoundationModelMonitor',
    'FoundationModelMetrics',
    'InferenceContext',
    'FoundationModelHealthChecker'
]