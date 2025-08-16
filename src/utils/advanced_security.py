"""Advanced security framework for research and production environments."""

import hashlib
import hmac
import secrets
import base64
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class SecurityAuditLog:
    """Security audit log entry."""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    action: str
    resource: str
    success: bool
    details: Dict[str, Any]


@dataclass
class DataPrivacyReport:
    """Data privacy analysis report."""
    privacy_score: float
    anonymization_level: str
    sensitive_fields_detected: List[str]
    privacy_risks: List[str]
    recommendations: List[str]


class DifferentialPrivacyMechanism:
    """Differential privacy implementation for causal discovery."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.budget_used = 0.0
        
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy."""
        if self.budget_used + sensitivity > self.epsilon:
            raise ValueError("Privacy budget exceeded")
        
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        self.budget_used += sensitivity
        
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise for (ε,δ)-differential privacy."""
        if self.budget_used + sensitivity > self.epsilon:
            raise ValueError("Privacy budget exceeded")
        
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        self.budget_used += sensitivity
        
        return value + noise
    
    def privatize_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Apply differential privacy to adjacency matrix."""
        privatized = adjacency_matrix.copy().astype(float)
        sensitivity = 1.0  # Single edge addition/removal
        
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if i != j:  # Don't add noise to diagonal
                    privatized[i, j] = self.add_laplace_noise(
                        privatized[i, j], sensitivity
                    )
                    # Clip to [0, 1] range
                    privatized[i, j] = np.clip(privatized[i, j], 0, 1)
        
        return privatized
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.budget_used)


class SecureDataProcessor:
    """Secure data processing with encryption and access control."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.cipher_suite = Fernet(master_key)
        self.access_log = []
        self.authorized_users = set()
        
    def encrypt_data(self, data: pd.DataFrame, include_metadata: bool = True) -> bytes:
        """Encrypt dataframe with optional metadata."""
        # Convert to JSON
        data_dict = {
            'data': data.to_dict(),
            'metadata': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'encrypted_at': time.time()
            } if include_metadata else {}
        }
        
        data_json = json.dumps(data_dict)
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Decrypt dataframe."""
        try:
            decrypted_json = self.cipher_suite.decrypt(encrypted_data).decode()
            data_dict = json.loads(decrypted_json)
            
            # Reconstruct dataframe
            df = pd.DataFrame.from_dict(data_dict['data'])
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def hash_sensitive_column(self, column: pd.Series, salt: Optional[str] = None) -> pd.Series:
        """Hash sensitive column values."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        def hash_value(value):
            combined = f"{salt}{value}".encode()
            return hashlib.sha256(combined).hexdigest()[:16]
        
        return column.apply(hash_value)
    
    def anonymize_dataset(self, data: pd.DataFrame, 
                         sensitive_columns: List[str],
                         anonymization_method: str = "hash") -> pd.DataFrame:
        """Anonymize sensitive columns in dataset."""
        anonymized = data.copy()
        
        for col in sensitive_columns:
            if col in anonymized.columns:
                if anonymization_method == "hash":
                    anonymized[col] = self.hash_sensitive_column(anonymized[col])
                elif anonymization_method == "noise":
                    if anonymized[col].dtype in ['int64', 'float64']:
                        noise_scale = anonymized[col].std() * 0.1
                        noise = np.random.normal(0, noise_scale, len(anonymized))
                        anonymized[col] += noise
                elif anonymization_method == "remove":
                    anonymized = anonymized.drop(columns=[col])
        
        return anonymized


class AccessControlManager:
    """Role-based access control for research resources."""
    
    def __init__(self):
        self.users = {}
        self.roles = {
            'admin': ['read', 'write', 'execute', 'admin'],
            'researcher': ['read', 'write', 'execute'],
            'analyst': ['read', 'execute'],
            'viewer': ['read']
        }
        self.audit_log = []
    
    def create_user(self, user_id: str, role: str, password: str) -> bool:
        """Create new user with role."""
        if role not in self.roles:
            return False
        
        # Hash password
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), 
                                          salt.encode(), 100000)
        
        self.users[user_id] = {
            'role': role,
            'password_hash': password_hash.hex(),
            'salt': salt,
            'created_at': time.time(),
            'last_login': None
        }
        
        self._log_event('user_created', user_id, 'create_user', True)
        return True
    
    def authenticate_user(self, user_id: str, password: str) -> bool:
        """Authenticate user credentials."""
        if user_id not in self.users:
            self._log_event('auth_failed', user_id, 'authenticate', False)
            return False
        
        user = self.users[user_id]
        salt = user['salt']
        expected_hash = user['password_hash']
        
        # Hash provided password
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(),
                                          salt.encode(), 100000)
        
        if hmac.compare_digest(password_hash.hex(), expected_hash):
            self.users[user_id]['last_login'] = time.time()
            self._log_event('auth_success', user_id, 'authenticate', True)
            return True
        
        self._log_event('auth_failed', user_id, 'authenticate', False)
        return False
    
    def check_permission(self, user_id: str, action: str) -> bool:
        """Check if user has permission for action."""
        if user_id not in self.users:
            return False
        
        user_role = self.users[user_id]['role']
        return action in self.roles.get(user_role, [])
    
    def _log_event(self, event_type: str, user_id: str, action: str, success: bool):
        """Log security event."""
        log_entry = SecurityAuditLog(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource='access_control',
            success=success,
            details={}
        )
        self.audit_log.append(log_entry)


class DataPrivacyAnalyzer:
    """Analyze datasets for privacy risks and compliance."""
    
    def __init__(self):
        self.sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
    
    def analyze_privacy(self, data: pd.DataFrame) -> DataPrivacyReport:
        """Comprehensive privacy analysis of dataset."""
        sensitive_fields = []
        privacy_risks = []
        recommendations = []
        
        # Check for sensitive patterns
        for col in data.columns:
            if data[col].dtype == 'object':  # String columns
                for pattern_name, pattern in self.sensitive_patterns.items():
                    if data[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                        sensitive_fields.append(f"{col} (contains {pattern_name})")
                        privacy_risks.append(f"Column {col} contains {pattern_name} data")
        
        # Check for potential PII columns by name
        pii_keywords = ['name', 'id', 'address', 'phone', 'email', 'ssn', 'age', 'birth']
        for col in data.columns:
            col_lower = col.lower()
            for keyword in pii_keywords:
                if keyword in col_lower:
                    if col not in [sf.split(' (')[0] for sf in sensitive_fields]:
                        sensitive_fields.append(f"{col} (potential PII)")
                        privacy_risks.append(f"Column {col} may contain personally identifiable information")
        
        # Calculate privacy score
        max_risk_score = len(data.columns) * 2  # Max 2 points per column
        current_risk = len(sensitive_fields)
        privacy_score = max(0, 1 - current_risk / max_risk_score)
        
        # Determine anonymization level
        if privacy_score >= 0.9:
            anonymization_level = "low_risk"
        elif privacy_score >= 0.7:
            anonymization_level = "medium_risk"
        else:
            anonymization_level = "high_risk"
        
        # Generate recommendations
        if sensitive_fields:
            recommendations.append("Consider anonymizing or removing sensitive fields")
            recommendations.append("Apply differential privacy mechanisms")
            recommendations.append("Implement access controls for sensitive data")
        
        if privacy_score < 0.8:
            recommendations.append("Conduct thorough privacy impact assessment")
            recommendations.append("Review data retention and disposal policies")
        
        return DataPrivacyReport(
            privacy_score=privacy_score,
            anonymization_level=anonymization_level,
            sensitive_fields_detected=sensitive_fields,
            privacy_risks=privacy_risks,
            recommendations=recommendations
        )


class SecureComputationEnvironment:
    """Secure environment for sensitive computations."""
    
    def __init__(self, 
                 enable_audit_logging: bool = True,
                 privacy_budget: float = 1.0):
        self.dp_mechanism = DifferentialPrivacyMechanism(epsilon=privacy_budget)
        self.data_processor = SecureDataProcessor()
        self.access_manager = AccessControlManager()
        self.privacy_analyzer = DataPrivacyAnalyzer()
        
        self.enable_audit_logging = enable_audit_logging
        self.computation_log = []
        
        if enable_audit_logging:
            self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Setup audit logging."""
        self.logger = logging.getLogger('secure_computation')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler('security_audit.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def secure_causal_discovery(self,
                              algorithm: Any,
                              data: pd.DataFrame,
                              user_id: str,
                              apply_privacy: bool = True) -> Dict[str, Any]:
        """Perform causal discovery in secure environment."""
        # Check user permissions
        if not self.access_manager.check_permission(user_id, 'execute'):
            raise PermissionError(f"User {user_id} does not have execute permission")
        
        # Analyze privacy risks
        privacy_report = self.privacy_analyzer.analyze_privacy(data)
        
        # Apply privacy protection if needed
        if apply_privacy and privacy_report.privacy_score < 0.8:
            if self.enable_audit_logging:
                self.logger.warning(f"High privacy risk detected, applying protection measures")
            
            # Anonymize sensitive data
            sensitive_columns = [field.split(' (')[0] for field in privacy_report.sensitive_fields_detected]
            data = self.data_processor.anonymize_dataset(data, sensitive_columns)
        
        # Run algorithm
        start_time = time.time()
        
        try:
            algorithm.fit(data)
            result = algorithm.predict(data)
            
            # Apply differential privacy to results if requested
            if apply_privacy:
                result.adjacency_matrix = self.dp_mechanism.privatize_adjacency_matrix(
                    result.adjacency_matrix
                )
            
            execution_time = time.time() - start_time
            
            # Log successful computation
            if self.enable_audit_logging:
                self.logger.info(f"Secure computation completed for user {user_id}")
            
            computation_log = {
                'user_id': user_id,
                'algorithm': algorithm.__class__.__name__,
                'execution_time': execution_time,
                'privacy_applied': apply_privacy,
                'privacy_score': privacy_report.privacy_score,
                'privacy_budget_used': self.dp_mechanism.budget_used,
                'timestamp': time.time()
            }
            
            self.computation_log.append(computation_log)
            
            return {
                'result': result,
                'privacy_report': privacy_report,
                'computation_log': computation_log,
                'remaining_privacy_budget': self.dp_mechanism.get_remaining_budget()
            }
            
        except Exception as e:
            if self.enable_audit_logging:
                self.logger.error(f"Secure computation failed for user {user_id}: {e}")
            raise
    
    def export_audit_logs(self, output_path: Path):
        """Export audit logs for compliance reporting."""
        audit_data = {
            'access_logs': [
                {
                    'timestamp': log.timestamp,
                    'event_type': log.event_type,
                    'user_id': log.user_id,
                    'action': log.action,
                    'success': log.success,
                    'details': log.details
                }
                for log in self.access_manager.audit_log
            ],
            'computation_logs': self.computation_log,
            'generated_at': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(audit_data, f, indent=2)


def create_secure_research_environment(privacy_budget: float = 1.0) -> SecureComputationEnvironment:
    """Factory function to create secure research environment."""
    env = SecureComputationEnvironment(
        enable_audit_logging=True,
        privacy_budget=privacy_budget
    )
    
    # Create default users
    env.access_manager.create_user('admin', 'admin', 'secure_admin_password')
    env.access_manager.create_user('researcher1', 'researcher', 'research_password')
    env.access_manager.create_user('analyst1', 'analyst', 'analyst_password')
    
    return env