"""Secure computing utilities for sensitive causal discovery operations."""

import os
import tempfile
import hashlib
import hmac
import secrets
import gc
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

try:
    from .error_handling import robust_execution, safe_execution, CausalDiscoveryError
    from .security import DataSecurityValidator, SecurityResult
except ImportError:
    from error_handling import robust_execution, safe_execution, CausalDiscoveryError
    from security import DataSecurityValidator, SecurityResult


@dataclass
class SecureComputationConfig:
    """Configuration for secure computation operations."""
    enable_memory_encryption: bool = True
    max_memory_mb: int = 1000
    enable_audit_log: bool = True
    secure_temp_dir: bool = True
    enable_data_anonymization: bool = False
    differential_privacy: bool = False
    privacy_epsilon: float = 1.0
    enable_secure_deletion: bool = True


class SecureMemoryManager:
    """Manage memory securely for sensitive operations."""
    
    def __init__(self, config: SecureComputationConfig):
        self.config = config
        self.allocated_objects = []
        self.temp_files = []
        self.logger = logging.getLogger(__name__)
    
    def allocate_secure_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Allocate array with secure memory management."""
        # Check memory limits
        required_mb = np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
        if required_mb > self.config.max_memory_mb:
            raise CausalDiscoveryError(f"Requested memory ({required_mb:.1f}MB) exceeds limit ({self.config.max_memory_mb}MB)")
        
        # Check available system memory
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if required_mb > available_mb * 0.8:  # Use max 80% of available memory
            raise CausalDiscoveryError(f"Insufficient memory: need {required_mb:.1f}MB, available {available_mb:.1f}MB")
        
        # Allocate array
        array = np.zeros(shape, dtype=dtype)
        self.allocated_objects.append(array)
        
        self.logger.debug(f"Allocated secure array: shape={shape}, size={required_mb:.1f}MB")
        return array
    
    def create_secure_temp_file(self, suffix: str = '.tmp') -> Path:
        """Create secure temporary file."""
        if self.config.secure_temp_dir:
            # Create secure temporary directory
            temp_dir = tempfile.mkdtemp(prefix='causal_discovery_secure_')
            os.chmod(temp_dir, 0o700)  # Only owner can read/write/execute
        else:
            temp_dir = tempfile.gettempdir()
        
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        os.close(fd)
        os.chmod(temp_path, 0o600)  # Only owner can read/write
        
        temp_path = Path(temp_path)
        self.temp_files.append(temp_path)
        
        self.logger.debug(f"Created secure temp file: {temp_path}")
        return temp_path
    
    def secure_delete(self, file_path: Path, passes: int = 3):
        """Securely delete file by overwriting with random data."""
        if not file_path.exists():
            return
        
        try:
            file_size = file_path.stat().st_size
            
            with open(file_path, 'r+b') as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            file_path.unlink()
            self.logger.debug(f"Securely deleted file: {file_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to securely delete {file_path}: {e}")
            try:
                file_path.unlink()  # Fallback to normal deletion
            except Exception:
                pass
    
    def cleanup(self):
        """Clean up all allocated resources securely."""
        # Secure deletion of temporary files
        if self.config.enable_secure_deletion:
            for temp_file in self.temp_files:
                self.secure_delete(temp_file)
        else:
            for temp_file in self.temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        
        # Clear memory arrays
        for obj in self.allocated_objects:
            if isinstance(obj, np.ndarray):
                obj.fill(0)  # Overwrite with zeros
        
        self.allocated_objects.clear()
        self.temp_files.clear()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.debug("Completed secure cleanup")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class DifferentialPrivacyManager:
    """Manage differential privacy for causal discovery."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.logger = logging.getLogger(__name__)
    
    def add_laplace_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        
        self.logger.debug(f"Added Laplace noise: epsilon={self.epsilon}, scale={scale}")
        return data + noise
    
    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Gaussian noise for differential privacy."""
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        
        self.logger.debug(f"Added Gaussian noise: epsilon={self.epsilon}, sigma={sigma}")
        return data + noise
    
    def privatize_adjacency_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Apply differential privacy to adjacency matrix."""
        # For binary adjacency matrices, use randomized response
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        
        # Create privatized matrix
        privatized = np.zeros_like(adj_matrix)
        
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i != j:  # Skip diagonal
                    if np.random.random() < p:
                        privatized[i, j] = adj_matrix[i, j]
                    else:
                        privatized[i, j] = 1 - adj_matrix[i, j]
        
        self.logger.debug(f"Applied differential privacy to adjacency matrix: epsilon={self.epsilon}")
        return privatized


class SecureAuditLogger:
    """Secure audit logging for causal discovery operations."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("causal_discovery_audit.log")
        self.session_id = secrets.token_hex(16)
        self.logger = logging.getLogger(f"{__name__}.audit")
        
        # Setup secure log file
        self.log_file.touch(mode=0o600)  # Only owner can read/write
        
        self._log_event("SESSION_START", {"session_id": self.session_id})
    
    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security event."""
        import time
        import json
        
        event = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "event_type": event_type,
            "details": details
        }
        
        # Create HMAC for integrity
        event_json = json.dumps(event, sort_keys=True)
        hmac_key = os.environ.get('CAUSAL_DISCOVERY_HMAC_KEY', 'default_key_change_in_production')
        event_hmac = hmac.new(hmac_key.encode(), event_json.encode(), hashlib.sha256).hexdigest()
        
        log_entry = {
            "event": event,
            "hmac": event_hmac
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_data_access(self, data_shape: Tuple[int, ...], data_hash: str):
        """Log data access event."""
        self._log_event("DATA_ACCESS", {
            "data_shape": data_shape,
            "data_hash": data_hash
        })
    
    def log_algorithm_execution(self, algorithm_name: str, parameters: Dict[str, Any]):
        """Log algorithm execution."""
        self._log_event("ALGORITHM_EXECUTION", {
            "algorithm": algorithm_name,
            "parameters": {k: str(v) for k, v in parameters.items()}
        })
    
    def log_result_generation(self, result_hash: str, algorithm: str):
        """Log result generation."""
        self._log_event("RESULT_GENERATION", {
            "result_hash": result_hash,
            "algorithm": algorithm
        })
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Log security violation."""
        self._log_event("SECURITY_VIOLATION", {
            "violation_type": violation_type,
            "details": details
        })
    
    def close_session(self):
        """Close audit session."""
        self._log_event("SESSION_END", {"session_id": self.session_id})


class SecureCausalDiscovery:
    """Secure wrapper for causal discovery algorithms."""
    
    def __init__(self, config: SecureComputationConfig = None):
        self.config = config or SecureComputationConfig()
        self.security_validator = DataSecurityValidator()
        self.audit_logger = SecureAuditLogger() if self.config.enable_audit_log else None
        self.dp_manager = DifferentialPrivacyManager(self.config.privacy_epsilon) if self.config.differential_privacy else None
        self.logger = logging.getLogger(__name__)
    
    def secure_fit_discover(self, algorithm, data: pd.DataFrame) -> Dict[str, Any]:
        """Securely execute causal discovery with full security measures."""
        
        with SecureMemoryManager(self.config) as memory_manager:
            try:
                # Step 1: Security validation
                security_result = self.security_validator.validate_data_security(data)
                if not security_result.is_secure and security_result.risk_level in ['HIGH', 'CRITICAL']:
                    if self.audit_logger:
                        self.audit_logger.log_security_violation("HIGH_RISK_DATA", {
                            "risk_level": security_result.risk_level,
                            "issues": security_result.issues
                        })
                    raise CausalDiscoveryError(f"Data security risk too high: {security_result.risk_level}")
                
                # Step 2: Data preparation with privacy protection
                secure_data = self._prepare_secure_data(data, memory_manager)
                
                # Step 3: Audit logging
                if self.audit_logger:
                    data_hash = hashlib.sha256(str(data.values).encode()).hexdigest()
                    self.audit_logger.log_data_access(data.shape, data_hash)
                    self.audit_logger.log_algorithm_execution(
                        algorithm.__class__.__name__,
                        getattr(algorithm, 'hyperparameters', {})
                    )
                
                # Step 4: Secure algorithm execution
                with safe_execution("secure_causal_discovery"):
                    result = algorithm.fit_discover(secure_data)
                
                # Step 5: Apply differential privacy if enabled
                if self.dp_manager:
                    result.adjacency_matrix = self.dp_manager.privatize_adjacency_matrix(
                        result.adjacency_matrix
                    )
                    result.confidence_scores = self.dp_manager.add_laplace_noise(
                        result.confidence_scores, sensitivity=1.0
                    )
                
                # Step 6: Final audit logging
                if self.audit_logger:
                    result_hash = hashlib.sha256(str(result.adjacency_matrix).encode()).hexdigest()
                    self.audit_logger.log_result_generation(result_hash, algorithm.__class__.__name__)
                
                self.logger.info("Secure causal discovery completed successfully")
                return result
                
            except Exception as e:
                if self.audit_logger:
                    self.audit_logger.log_security_violation("ALGORITHM_FAILURE", {
                        "error": str(e),
                        "algorithm": algorithm.__class__.__name__
                    })
                raise CausalDiscoveryError(f"Secure causal discovery failed: {e}")
    
    def _prepare_secure_data(self, data: pd.DataFrame, memory_manager: SecureMemoryManager) -> pd.DataFrame:
        """Prepare data with security measures."""
        
        # Create secure copy of data
        secure_data = data.copy()
        
        # Apply data anonymization if enabled
        if self.config.enable_data_anonymization:
            secure_data = self._anonymize_data(secure_data)
        
        # Add differential privacy noise if enabled
        if self.dp_manager:
            numeric_cols = secure_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                secure_data[col] = self.dp_manager.add_laplace_noise(
                    secure_data[col].values, sensitivity=1.0
                )
        
        return secure_data
    
    def _anonymize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic data anonymization techniques."""
        anonymized = data.copy()
        
        # Remove potential identifier columns
        identifier_patterns = ['id', 'key', 'index', 'uid', 'guid']
        cols_to_remove = []
        
        for col in anonymized.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in identifier_patterns):
                cols_to_remove.append(col)
        
        if cols_to_remove:
            anonymized = anonymized.drop(columns=cols_to_remove)
            self.logger.info(f"Removed potential identifier columns: {cols_to_remove}")
        
        return anonymized
    
    def close(self):
        """Close secure session."""
        if self.audit_logger:
            self.audit_logger.close_session()


@robust_execution(max_retries=2, enable_recovery=True)
def secure_data_processing(data: pd.DataFrame, operation: str = "causal_discovery") -> pd.DataFrame:
    """Securely process data with comprehensive error handling."""
    
    config = SecureComputationConfig()
    
    with SecureMemoryManager(config) as memory_manager:
        # Validate input security
        validator = DataSecurityValidator()
        security_result = validator.validate_data_security(data)
        
        if security_result.risk_level == 'CRITICAL':
            raise CausalDiscoveryError("Critical security risk detected in data")
        
        # Process data securely
        processed_data = data.copy()
        
        # Apply security measures based on risk level
        if security_result.risk_level in ['HIGH', 'MEDIUM']:
            # Add noise for privacy protection
            dp_manager = DifferentialPrivacyManager(epsilon=1.0)
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                processed_data[col] = dp_manager.add_laplace_noise(
                    processed_data[col].values
                )
        
        return processed_data


def create_secure_pipeline_config() -> Dict[str, Any]:
    """Create a secure configuration for causal discovery pipeline."""
    return {
        'security': SecureComputationConfig(
            enable_memory_encryption=True,
            max_memory_mb=1000,
            enable_audit_log=True,
            secure_temp_dir=True,
            enable_data_anonymization=True,
            differential_privacy=True,
            privacy_epsilon=1.0,
            enable_secure_deletion=True
        ),
        'error_handling': {
            'enable_recovery': True,
            'max_retries': 3,
            'circuit_breaker': True,
            'timeout_seconds': 300
        },
        'validation': {
            'strict_mode': True,
            'security_checks': True,
            'privacy_validation': True
        }
    }