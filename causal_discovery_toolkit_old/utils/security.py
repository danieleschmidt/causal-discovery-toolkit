"""Security utilities for causal discovery toolkit."""

import hashlib
import hmac
import secrets
import os
from typing import Dict, Any, Optional, List, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass
try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class SecurityResult:
    """Result of security validation."""
    is_secure: bool
    issues: List[str]
    recommendations: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class DataSecurityValidator:
    """Validate data for security concerns."""
    
    def __init__(self):
        """Initialize security validator."""
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address pattern
        ]
        
        self.pii_column_indicators = [
            'ssn', 'social', 'email', 'phone', 'address', 'name', 
            'firstname', 'lastname', 'fullname', 'credit', 'card',
            'account', 'password', 'pin', 'secret', 'key', 'token'
        ]
    
    def validate_data_security(self, data: pd.DataFrame) -> SecurityResult:
        """Validate DataFrame for security and privacy concerns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            SecurityResult with security assessment
        """
        issues = []
        recommendations = []
        risk_level = "LOW"
        
        # Check for PII in column names
        pii_columns = self._detect_pii_columns(data.columns)
        if pii_columns:
            issues.append(f"Potential PII columns detected: {pii_columns}")
            recommendations.append("Consider anonymizing or removing PII columns")
            risk_level = "HIGH"
        
        # Check for sensitive data patterns in string columns
        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            sensitive_data_found = self._check_sensitive_patterns(data[string_columns])
            if sensitive_data_found:
                issues.append(f"Sensitive data patterns found in columns: {sensitive_data_found}")
                recommendations.append("Sanitize or hash sensitive data before analysis")
                risk_level = "CRITICAL"
        
        # Check for data leakage indicators
        leakage_indicators = self._check_data_leakage(data)
        if leakage_indicators:
            issues.append(f"Potential data leakage indicators: {leakage_indicators}")
            recommendations.append("Review data for future information or target leakage")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Check data size for privacy concerns
        if len(data) < 100:
            issues.append("Small dataset may pose re-identification risks")
            recommendations.append("Consider differential privacy or data aggregation")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Check for high-cardinality categorical variables (potential identifiers)
        high_card_cols = self._check_high_cardinality(data)
        if high_card_cols:
            issues.append(f"High-cardinality columns (potential identifiers): {high_card_cols}")
            recommendations.append("Consider grouping or removing high-cardinality features")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        is_secure = len(issues) == 0 or risk_level == "LOW"
        
        return SecurityResult(
            is_secure=is_secure,
            issues=issues,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _detect_pii_columns(self, columns: pd.Index) -> List[str]:
        """Detect columns that may contain PII."""
        pii_columns = []
        for col in columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in self.pii_column_indicators):
                pii_columns.append(col)
        return pii_columns
    
    def _check_sensitive_patterns(self, data: pd.DataFrame) -> List[str]:
        """Check for sensitive data patterns in string columns."""
        import re
        sensitive_columns = []
        
        for col in data.columns:
            for pattern in self.sensitive_patterns:
                if data[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                    sensitive_columns.append(col)
                    break
        
        return sensitive_columns
    
    def _check_data_leakage(self, data: pd.DataFrame) -> List[str]:
        """Check for potential data leakage indicators."""
        leakage_indicators = []
        
        # Check for perfect predictors (constant or near-constant variance)
        numeric_data = data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            if numeric_data[col].var() < 1e-10:
                leakage_indicators.append(f"{col} (near-constant)")
        
        # Check for suspiciously high correlations
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            high_corr = np.where(corr_matrix > 0.99)
            if len(high_corr[0]) > 0:
                for i, j in zip(high_corr[0], high_corr[1]):
                    if i < j:  # Avoid duplicates
                        col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                        leakage_indicators.append(f"{col1}-{col2} (correlation={corr_matrix.iloc[i,j]:.3f})")
        
        return leakage_indicators
    
    def _check_high_cardinality(self, data: pd.DataFrame) -> List[str]:
        """Check for high-cardinality categorical variables."""
        high_card_columns = []
        
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.8:  # More than 80% unique values
                    high_card_columns.append(f"{col} (uniqueness={unique_ratio:.1%})")
        
        return high_card_columns


class SecureDataHandler:
    """Handle data securely with encryption and hashing capabilities."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize secure data handler.
        
        Args:
            key: Encryption key. If None, generates a new one.
        """
        self.key = key or self._generate_key()
        self.salt = secrets.token_bytes(32)
    
    def _generate_key(self) -> bytes:
        """Generate a secure random key."""
        return secrets.token_bytes(32)
    
    def hash_data(self, data: Any, algorithm: str = 'sha256') -> str:
        """Securely hash data.
        
        Args:
            data: Data to hash
            algorithm: Hashing algorithm
            
        Returns:
            Hex-encoded hash
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm {algorithm} not available")
        
        data_bytes = str(data).encode('utf-8')
        hasher = hashlib.new(algorithm)
        hasher.update(self.salt + data_bytes)
        
        return hasher.hexdigest()
    
    def hash_column(self, series: pd.Series, algorithm: str = 'sha256') -> pd.Series:
        """Hash all values in a pandas Series.
        
        Args:
            series: Series to hash
            algorithm: Hashing algorithm
            
        Returns:
            Series with hashed values
        """
        return series.apply(lambda x: self.hash_data(x, algorithm))
    
    def create_hmac(self, data: Any, algorithm: str = 'sha256') -> str:
        """Create HMAC for data integrity.
        
        Args:
            data: Data to create HMAC for
            algorithm: HMAC algorithm
            
        Returns:
            Hex-encoded HMAC
        """
        data_bytes = str(data).encode('utf-8')
        hmac_obj = hmac.new(self.key, data_bytes, getattr(hashlib, algorithm))
        return hmac_obj.hexdigest()
    
    def verify_hmac(self, data: Any, expected_hmac: str, algorithm: str = 'sha256') -> bool:
        """Verify HMAC for data integrity.
        
        Args:
            data: Data to verify
            expected_hmac: Expected HMAC value
            algorithm: HMAC algorithm
            
        Returns:
            True if HMAC is valid
        """
        computed_hmac = self.create_hmac(data, algorithm)
        return hmac.compare_digest(computed_hmac, expected_hmac)
    
    def anonymize_dataframe(self, data: pd.DataFrame, 
                          columns_to_hash: Optional[List[str]] = None,
                          columns_to_remove: Optional[List[str]] = None) -> pd.DataFrame:
        """Anonymize DataFrame by hashing or removing sensitive columns.
        
        Args:
            data: DataFrame to anonymize
            columns_to_hash: Columns to hash instead of remove
            columns_to_remove: Columns to completely remove
            
        Returns:
            Anonymized DataFrame
        """
        anonymized = data.copy()
        
        if columns_to_remove:
            anonymized = anonymized.drop(columns=columns_to_remove, errors='ignore')
            logger.info(f"Removed columns: {columns_to_remove}")
        
        if columns_to_hash:
            for col in columns_to_hash:
                if col in anonymized.columns:
                    anonymized[col] = self.hash_column(anonymized[col])
                    logger.info(f"Hashed column: {col}")
        
        return anonymized


class AccessControlManager:
    """Manage access control and permissions."""
    
    def __init__(self):
        """Initialize access control manager."""
        self.permissions: Dict[str, Set[str]] = {}
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
    
    def add_permission(self, user: str, permission: str) -> None:
        """Add permission for user.
        
        Args:
            user: User identifier
            permission: Permission string
        """
        if user not in self.permissions:
            self.permissions[user] = set()
        self.permissions[user].add(permission)
        logger.info(f"Added permission '{permission}' for user '{user}'")
    
    def check_permission(self, user: str, permission: str) -> bool:
        """Check if user has permission.
        
        Args:
            user: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_permissions = self.permissions.get(user, set())
        has_permission = permission in user_permissions or 'admin' in user_permissions
        
        if not has_permission:
            logger.warning(f"Access denied: User '{user}' lacks permission '{permission}'")
        
        return has_permission
    
    def create_session_token(self, user: str, permissions: Optional[List[str]] = None) -> str:
        """Create a session token for user.
        
        Args:
            user: User identifier
            permissions: Optional list of permissions for this session
            
        Returns:
            Session token
        """
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = {
            'user': user,
            'permissions': set(permissions) if permissions else self.permissions.get(user, set()),
            'created_at': pd.Timestamp.now(),
            'expires_at': pd.Timestamp.now() + pd.Timedelta(hours=24)
        }
        
        logger.info(f"Created session token for user '{user}'")
        return token
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token.
        
        Args:
            token: Session token to validate
            
        Returns:
            Session info if valid, None otherwise
        """
        if token not in self.session_tokens:
            logger.warning("Invalid session token")
            return None
        
        session = self.session_tokens[token]
        if pd.Timestamp.now() > session['expires_at']:
            logger.warning("Expired session token")
            del self.session_tokens[token]
            return None
        
        return session


def sanitize_input(data: Any, max_length: int = 1000) -> str:
    """Sanitize input data to prevent injection attacks.
    
    Args:
        data: Input data to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    # Convert to string and limit length
    sanitized = str(data)[:max_length]
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '`', '|', ';', '\n', '\r', '\t']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized


def secure_random_seed(length: int = 32) -> int:
    """Generate cryptographically secure random seed.
    
    Args:
        length: Length of random bytes to generate
        
    Returns:
        Secure random integer
    """
    random_bytes = secrets.token_bytes(length)
    return int.from_bytes(random_bytes, byteorder='big') % (2**31)


class AuditLogger:
    """Audit logging for security events."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Optional file to write audit logs
        """
        self.audit_logger = get_logger("audit")
        self.log_file = log_file
    
    def log_access_attempt(self, user: str, resource: str, 
                          success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log access attempt.
        
        Args:
            user: User attempting access
            resource: Resource being accessed
            success: Whether access was successful
            details: Additional details
        """
        log_entry = {
            'event_type': 'access_attempt',
            'user': user,
            'resource': resource,
            'success': success,
            'timestamp': pd.Timestamp.now().isoformat(),
            'details': details or {}
        }
        
        self.audit_logger.info(f"Access attempt: {log_entry}")
        
        if self.log_file:
            self._write_to_file(log_entry)
    
    def log_data_operation(self, user: str, operation: str, 
                          data_info: Dict[str, Any]) -> None:
        """Log data operation.
        
        Args:
            user: User performing operation
            operation: Type of operation
            data_info: Information about the data
        """
        log_entry = {
            'event_type': 'data_operation',
            'user': user,
            'operation': operation,
            'data_info': data_info,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.audit_logger.info(f"Data operation: {log_entry}")
        
        if self.log_file:
            self._write_to_file(log_entry)
    
    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        try:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log to file: {e}")


# Global instances
global_audit_logger = AuditLogger()
global_access_manager = AccessControlManager()
global_security_validator = DataSecurityValidator()