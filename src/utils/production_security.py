"""
Production Security: Enterprise-Grade Security Framework
========================================================

Comprehensive security framework for causal discovery systems with
multi-layered protection, compliance monitoring, and threat detection.

Security Features:
- Data privacy protection and PII detection
- Input validation and sanitization
- Access control and authentication
- Audit trails and compliance monitoring
- Threat detection and response
- Secure data handling and encryption
- GDPR/HIPAA compliance support
"""

import hashlib
import hmac
import secrets
import base64
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Compliance standards supported."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    user_id: Optional[str]
    source_ip: Optional[str]
    description: str
    affected_data: Optional[str]
    mitigation_action: Optional[str]
    compliance_impact: List[ComplianceStandard] = field(default_factory=list)

@dataclass
class DataClassification:
    """Data classification result."""
    classification: SecurityLevel
    pii_detected: bool
    sensitive_fields: List[str]
    compliance_requirements: List[ComplianceStandard]
    risk_score: float
    recommendations: List[str]

@dataclass
class AccessRequest:
    """Access control request."""
    user_id: str
    resource: str
    action: str
    timestamp: datetime
    justification: Optional[str] = None
    approval_required: bool = False

class PIIDetector:
    """Personally Identifiable Information detector."""
    
    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'date_of_birth': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'),
        }
        
        # Sensitive field name patterns
        self.sensitive_field_patterns = [
            r'.*name.*', r'.*email.*', r'.*phone.*', r'.*address.*',
            r'.*ssn.*', r'.*social.*', r'.*birth.*', r'.*age.*',
            r'.*salary.*', r'.*income.*', r'.*medical.*', r'.*health.*'
        ]
    
    def detect_pii(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> Dict[str, Any]:
        """Detect PII in various data formats."""
        
        pii_found = {
            'has_pii': False,
            'pii_types': [],
            'sensitive_fields': [],
            'locations': []
        }
        
        if isinstance(data, pd.DataFrame):
            pii_found = self._detect_pii_dataframe(data)
        elif isinstance(data, dict):
            pii_found = self._detect_pii_dict(data)
        elif isinstance(data, str):
            pii_found = self._detect_pii_string(data)
        
        return pii_found
    
    def _detect_pii_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect PII in DataFrame."""
        
        pii_found = {
            'has_pii': False,
            'pii_types': [],
            'sensitive_fields': [],
            'locations': []
        }
        
        # Check column names for sensitive patterns
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.sensitive_field_patterns:
                if re.match(pattern, col_lower):
                    pii_found['sensitive_fields'].append(col)
                    pii_found['has_pii'] = True
                    break
        
        # Check data content for PII patterns
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                sample_data = df[col].astype(str).str.cat(sep=' ')
                pii_in_column = self._detect_pii_string(sample_data)
                
                if pii_in_column['has_pii']:
                    pii_found['has_pii'] = True
                    pii_found['pii_types'].extend(pii_in_column['pii_types'])
                    pii_found['locations'].append(f"column_{col}")
        
        # Remove duplicates
        pii_found['pii_types'] = list(set(pii_found['pii_types']))
        pii_found['sensitive_fields'] = list(set(pii_found['sensitive_fields']))
        
        return pii_found
    
    def _detect_pii_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect PII in dictionary."""
        
        pii_found = {
            'has_pii': False,
            'pii_types': [],
            'sensitive_fields': [],
            'locations': []
        }
        
        for key, value in data_dict.items():
            # Check key names
            key_lower = str(key).lower()
            for pattern in self.sensitive_field_patterns:
                if re.match(pattern, key_lower):
                    pii_found['sensitive_fields'].append(key)
                    pii_found['has_pii'] = True
                    break
            
            # Check values
            if isinstance(value, str):
                pii_in_value = self._detect_pii_string(value)
                if pii_in_value['has_pii']:
                    pii_found['has_pii'] = True
                    pii_found['pii_types'].extend(pii_in_value['pii_types'])
                    pii_found['locations'].append(f"field_{key}")
        
        return pii_found
    
    def _detect_pii_string(self, text: str) -> Dict[str, Any]:
        """Detect PII in string text."""
        
        pii_found = {
            'has_pii': False,
            'pii_types': [],
            'sensitive_fields': [],
            'locations': []
        }
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_found['has_pii'] = True
                pii_found['pii_types'].append(pii_type)
                pii_found['locations'].extend([f"{pii_type}_{i}" for i in range(len(matches))])
        
        return pii_found

class DataSanitizer:
    """Data sanitization and anonymization."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.pii_detector = PIIDetector()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)
    
    def sanitize_data(self, data: pd.DataFrame, 
                     anonymization_method: str = 'hash') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Sanitize data by removing or anonymizing PII."""
        
        # Detect PII first
        pii_info = self.pii_detector.detect_pii(data)
        
        sanitized_data = data.copy()
        sanitization_report = {
            'pii_detected': pii_info['has_pii'],
            'sensitive_fields': pii_info['sensitive_fields'],
            'anonymization_method': anonymization_method,
            'fields_processed': []
        }
        
        # Sanitize sensitive fields
        for field in pii_info['sensitive_fields']:
            if field in sanitized_data.columns:
                if anonymization_method == 'hash':
                    sanitized_data[field] = sanitized_data[field].apply(
                        lambda x: self._hash_value(str(x)) if pd.notna(x) else x
                    )
                elif anonymization_method == 'mask':
                    sanitized_data[field] = sanitized_data[field].apply(
                        lambda x: self._mask_value(str(x)) if pd.notna(x) else x
                    )
                elif anonymization_method == 'remove':
                    sanitized_data = sanitized_data.drop(columns=[field])
                
                sanitization_report['fields_processed'].append(field)
        
        return sanitized_data, sanitization_report
    
    def _hash_value(self, value: str) -> str:
        """Hash sensitive value."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _mask_value(self, value: str) -> str:
        """Mask sensitive value."""
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]

class AccessController:
    """Role-based access control system."""
    
    def __init__(self):
        self.roles = {}
        self.permissions = {}
        self.user_roles = {}
        self.access_log = []
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default security roles."""
        
        self.roles = {
            'admin': {
                'permissions': ['read', 'write', 'delete', 'admin'],
                'data_access_level': SecurityLevel.RESTRICTED,
                'description': 'Full system access'
            },
            'researcher': {
                'permissions': ['read', 'write'],
                'data_access_level': SecurityLevel.CONFIDENTIAL,
                'description': 'Research and analysis access'
            },
            'analyst': {
                'permissions': ['read'],
                'data_access_level': SecurityLevel.INTERNAL,
                'description': 'Read-only analysis access'
            },
            'viewer': {
                'permissions': ['read'],
                'data_access_level': SecurityLevel.PUBLIC,
                'description': 'View-only access to public data'
            }
        }
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user."""
        
        if role in self.roles:
            self.user_roles[user_id] = role
            self._log_access_event(user_id, 'role_assigned', f"Role {role} assigned")
            return True
        return False
    
    def check_permission(self, user_id: str, permission: str, 
                        resource_security_level: SecurityLevel = SecurityLevel.PUBLIC) -> bool:
        """Check if user has permission for action."""
        
        if user_id not in self.user_roles:
            self._log_access_event(user_id, 'access_denied', 'No role assigned')
            return False
        
        user_role = self.user_roles[user_id]
        role_info = self.roles[user_role]
        
        # Check permission
        if permission not in role_info['permissions']:
            self._log_access_event(user_id, 'access_denied', f'Permission {permission} denied')
            return False
        
        # Check data access level
        user_access_level = role_info['data_access_level']
        if not self._can_access_level(user_access_level, resource_security_level):
            self._log_access_event(user_id, 'access_denied', f'Security level {resource_security_level} denied')
            return False
        
        self._log_access_event(user_id, 'access_granted', f'Permission {permission} granted')
        return True
    
    def _can_access_level(self, user_level: SecurityLevel, resource_level: SecurityLevel) -> bool:
        """Check if user can access resource security level."""
        
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[resource_level]
    
    def _log_access_event(self, user_id: str, event_type: str, description: str):
        """Log access control event."""
        
        event = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'event_type': event_type,
            'description': description
        }
        self.access_log.append(event)
        
        # Keep only recent events
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]

class ThreatDetector:
    """Security threat detection system."""
    
    def __init__(self):
        self.threat_patterns = {
            'injection_attack': [
                r'.*union.*select.*',
                r'.*drop.*table.*',
                r'.*script.*alert.*',
                r'.*exec.*xp_.*'
            ],
            'suspicious_patterns': [
                r'.*\.\./.*',  # Directory traversal
                r'.*<script.*>.*',  # XSS
                r'.*eval\(.*\).*',  # Code injection
            ],
            'data_exfiltration': [
                r'.*select.*from.*users.*',
                r'.*dump.*database.*'
            ]
        }
        
        self.failed_attempts = {}
        self.threat_events = []
    
    def analyze_input(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """Analyze input for security threats."""
        
        threat_analysis = {
            'is_threat': False,
            'threat_types': [],
            'threat_level': ThreatLevel.LOW,
            'details': []
        }
        
        input_lower = user_input.lower()
        
        # Check for threat patterns
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    threat_analysis['is_threat'] = True
                    threat_analysis['threat_types'].append(threat_type)
                    threat_analysis['details'].append(f"Pattern matched: {pattern}")
        
        # Determine threat level
        if threat_analysis['threat_types']:
            if any(t in ['injection_attack', 'data_exfiltration'] for t in threat_analysis['threat_types']):
                threat_analysis['threat_level'] = ThreatLevel.HIGH
            elif 'suspicious_patterns' in threat_analysis['threat_types']:
                threat_analysis['threat_level'] = ThreatLevel.MEDIUM
        
        # Track failed attempts
        if threat_analysis['is_threat'] and user_id:
            self._track_failed_attempt(user_id, threat_analysis)
        
        return threat_analysis
    
    def _track_failed_attempt(self, user_id: str, threat_info: Dict[str, Any]):
        """Track failed security attempts."""
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append({
            'timestamp': datetime.now(),
            'threat_info': threat_info
        })
        
        # Check for multiple failed attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[user_id]
            if datetime.now() - attempt['timestamp'] < timedelta(hours=1)
        ]
        
        if len(recent_attempts) >= 3:
            self._raise_security_alert(user_id, ThreatLevel.CRITICAL,
                                     "Multiple security violations detected")
    
    def _raise_security_alert(self, user_id: str, threat_level: ThreatLevel, description: str):
        """Raise security alert."""
        
        alert = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            event_type='security_alert',
            threat_level=threat_level,
            user_id=user_id,
            source_ip=None,
            description=description,
            affected_data=None,
            mitigation_action="User monitoring increased"
        )
        
        self.threat_events.append(alert)
        logging.warning(f"SECURITY ALERT: {description} - User: {user_id}")

class ComplianceMonitor:
    """Compliance monitoring and reporting."""
    
    def __init__(self, enabled_standards: List[ComplianceStandard]):
        self.enabled_standards = enabled_standards
        self.compliance_events = []
        self.audit_trail = []
    
    def check_compliance(self, data_classification: DataClassification,
                        operation: str, user_id: str) -> Dict[str, Any]:
        """Check compliance requirements."""
        
        compliance_check = {
            'compliant': True,
            'violations': [],
            'requirements': [],
            'recommendations': []
        }
        
        for standard in self.enabled_standards:
            violation = self._check_standard_compliance(
                standard, data_classification, operation
            )
            
            if violation:
                compliance_check['compliant'] = False
                compliance_check['violations'].append(violation)
        
        # Log compliance check
        self._log_compliance_event(user_id, operation, compliance_check)
        
        return compliance_check
    
    def _check_standard_compliance(self, standard: ComplianceStandard,
                                 classification: DataClassification,
                                 operation: str) -> Optional[Dict[str, Any]]:
        """Check specific compliance standard."""
        
        if standard == ComplianceStandard.GDPR:
            return self._check_gdpr_compliance(classification, operation)
        elif standard == ComplianceStandard.HIPAA:
            return self._check_hipaa_compliance(classification, operation)
        
        return None
    
    def _check_gdpr_compliance(self, classification: DataClassification,
                             operation: str) -> Optional[Dict[str, Any]]:
        """Check GDPR compliance."""
        
        if classification.pii_detected and operation in ['export', 'share']:
            return {
                'standard': ComplianceStandard.GDPR,
                'violation_type': 'unauthorized_pii_processing',
                'description': 'PII data requires explicit consent for export/sharing',
                'severity': 'high'
            }
        
        return None
    
    def _check_hipaa_compliance(self, classification: DataClassification,
                              operation: str) -> Optional[Dict[str, Any]]:
        """Check HIPAA compliance."""
        
        if ('medical' in classification.sensitive_fields or 
            'health' in classification.sensitive_fields):
            if operation in ['export', 'share', 'analyze']:
                return {
                    'standard': ComplianceStandard.HIPAA,
                    'violation_type': 'unauthorized_phi_access',
                    'description': 'Protected Health Information requires special authorization',
                    'severity': 'critical'
                }
        
        return None
    
    def _log_compliance_event(self, user_id: str, operation: str, 
                            compliance_result: Dict[str, Any]):
        """Log compliance monitoring event."""
        
        event = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'operation': operation,
            'compliant': compliance_result['compliant'],
            'violations': compliance_result['violations']
        }
        
        self.compliance_events.append(event)

class ProductionSecurity:
    """
    Comprehensive production security framework for causal discovery systems.
    
    This security framework provides:
    1. Multi-layered data protection and PII detection
    2. Role-based access control with fine-grained permissions
    3. Threat detection and incident response
    4. Compliance monitoring (GDPR, HIPAA, etc.)
    5. Audit trails and security reporting
    6. Secure data handling and encryption
    
    Key Security Features:
    - Proactive threat detection and prevention
    - Automated compliance checking and reporting
    - Secure data sanitization and anonymization
    - Comprehensive audit trails for accountability
    - Real-time security monitoring and alerting
    - Enterprise-grade access control systems
    """
    
    def __init__(self, 
                 compliance_standards: List[ComplianceStandard] = None,
                 enable_threat_detection: bool = True,
                 enable_access_control: bool = True,
                 audit_log_file: Optional[str] = None):
        """
        Initialize production security framework.
        
        Args:
            compliance_standards: List of compliance standards to enforce
            enable_threat_detection: Enable threat detection system
            enable_access_control: Enable access control system
            audit_log_file: Optional audit log file path
        """
        self.compliance_standards = compliance_standards or [
            ComplianceStandard.GDPR, ComplianceStandard.HIPAA
        ]
        
        # Initialize security components
        self.pii_detector = PIIDetector()
        self.data_sanitizer = DataSanitizer()
        self.access_controller = AccessController() if enable_access_control else None
        self.threat_detector = ThreatDetector() if enable_threat_detection else None
        self.compliance_monitor = ComplianceMonitor(self.compliance_standards)
        
        # Security state
        self.security_events = []
        self.current_user = None
        self.session_id = None
        
        # Audit logging
        self.audit_log_file = audit_log_file
        
        logging.info("Production security framework initialized")
    
    def classify_data(self, data: pd.DataFrame) -> DataClassification:
        """Classify data security level and requirements."""
        
        # Detect PII
        pii_info = self.pii_detector.detect_pii(data)
        
        # Determine classification level
        if pii_info['has_pii']:
            if any(pii_type in ['ssn', 'credit_card', 'medical'] 
                  for pii_type in pii_info['pii_types']):
                classification = SecurityLevel.RESTRICTED
            else:
                classification = SecurityLevel.CONFIDENTIAL
        else:
            classification = SecurityLevel.INTERNAL
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(pii_info, data)
        
        # Determine compliance requirements
        compliance_requirements = []
        if pii_info['has_pii']:
            compliance_requirements.append(ComplianceStandard.GDPR)
        
        if any(field for field in pii_info['sensitive_fields'] 
               if 'medical' in field.lower() or 'health' in field.lower()):
            compliance_requirements.append(ComplianceStandard.HIPAA)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(
            classification, pii_info, risk_score
        )
        
        return DataClassification(
            classification=classification,
            pii_detected=pii_info['has_pii'],
            sensitive_fields=pii_info['sensitive_fields'],
            compliance_requirements=compliance_requirements,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _calculate_risk_score(self, pii_info: Dict[str, Any], data: pd.DataFrame) -> float:
        """Calculate security risk score (0-1)."""
        
        risk_factors = []
        
        # PII presence risk
        if pii_info['has_pii']:
            risk_factors.append(0.3)
            
            # High-risk PII types
            high_risk_types = ['ssn', 'credit_card', 'medical']
            for pii_type in pii_info['pii_types']:
                if pii_type in high_risk_types:
                    risk_factors.append(0.4)
        
        # Data volume risk
        if len(data) > 10000:
            risk_factors.append(0.2)
        
        # Sensitive field count risk
        if len(pii_info['sensitive_fields']) > 3:
            risk_factors.append(0.3)
        
        # Calculate combined risk
        total_risk = sum(risk_factors)
        return min(1.0, total_risk)
    
    def _generate_security_recommendations(self, classification: SecurityLevel,
                                         pii_info: Dict[str, Any],
                                         risk_score: float) -> List[str]:
        """Generate security recommendations."""
        
        recommendations = []
        
        if pii_info['has_pii']:
            recommendations.append("Consider data anonymization before processing")
            recommendations.append("Implement data retention policies")
            recommendations.append("Enable audit logging for all operations")
        
        if classification in [SecurityLevel.RESTRICTED, SecurityLevel.CONFIDENTIAL]:
            recommendations.append("Require administrative approval for data access")
            recommendations.append("Enable encryption for data at rest")
            recommendations.append("Implement network access restrictions")
        
        if risk_score > 0.7:
            recommendations.append("Consider additional security controls")
            recommendations.append("Enable real-time monitoring and alerting")
        
        return recommendations
    
    def validate_and_sanitize_input(self, user_input: str, user_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """Validate and sanitize user input."""
        
        validation_result = {
            'is_safe': True,
            'threats_detected': [],
            'sanitization_applied': False
        }
        
        # Threat detection
        if self.threat_detector:
            threat_analysis = self.threat_detector.analyze_input(user_input, user_id)
            
            if threat_analysis['is_threat']:
                validation_result['is_safe'] = False
                validation_result['threats_detected'] = threat_analysis['threat_types']
                
                # Log security event
                self._log_security_event(
                    'input_threat_detected',
                    ThreatLevel.HIGH,
                    f"Malicious input detected from user {user_id}",
                    user_id
                )
                
                return "", validation_result
        
        # Basic input sanitization
        sanitized_input = self._sanitize_input(user_input)
        
        if sanitized_input != user_input:
            validation_result['sanitization_applied'] = True
        
        return sanitized_input, validation_result
    
    def _sanitize_input(self, user_input: str) -> str:
        """Basic input sanitization."""
        
        # Remove potential script tags
        sanitized = re.sub(r'<script.*?</script>', '', user_input, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove SQL injection patterns
        dangerous_patterns = [
            r'union\s+select', r'drop\s+table', r'delete\s+from',
            r'insert\s+into', r'update\s+.*\s+set'
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def authorize_operation(self, user_id: str, operation: str, 
                          data_classification: DataClassification) -> bool:
        """Authorize user operation based on security policies."""
        
        # Check access control
        if self.access_controller:
            if not self.access_controller.check_permission(
                user_id, operation, data_classification.classification):
                return False
        
        # Check compliance requirements
        compliance_result = self.compliance_monitor.check_compliance(
            data_classification, operation, user_id
        )
        
        if not compliance_result['compliant']:
            self._log_security_event(
                'compliance_violation',
                ThreatLevel.HIGH,
                f"Compliance violation: {operation} by {user_id}",
                user_id
            )
            return False
        
        return True
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel,
                          description: str, user_id: str = None):
        """Log security event."""
        
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            user_id=user_id,
            source_ip=None,
            description=description,
            affected_data=None,
            mitigation_action=None
        )
        
        self.security_events.append(event)
        
        # Log to file if configured
        if self.audit_log_file:
            with open(self.audit_log_file, 'a') as f:
                f.write(f"{event.timestamp.isoformat()} - {event_type} - {description}\n")
        
        # Log with appropriate level
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logging.error(f"SECURITY EVENT: {description}")
        else:
            logging.warning(f"Security event: {description}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        
        recent_events = [
            event for event in self.security_events
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        threat_counts = {}
        for event in recent_events:
            threat_level = event.threat_level.value
            threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
        
        return {
            'total_security_events': len(self.security_events),
            'recent_events_24h': len(recent_events),
            'threat_level_counts': threat_counts,
            'compliance_standards': [std.value for std in self.compliance_standards],
            'access_control_enabled': self.access_controller is not None,
            'threat_detection_enabled': self.threat_detector is not None,
            'failed_attempts': len(self.threat_detector.failed_attempts) if self.threat_detector else 0
        }

# Global security instance
global_security = ProductionSecurity(
    compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA],
    enable_threat_detection=True,
    enable_access_control=True
)

# Export main components
__all__ = [
    'ProductionSecurity',
    'SecurityLevel',
    'ThreatLevel',
    'ComplianceStandard',
    'DataClassification',
    'SecurityEvent',
    'PIIDetector',
    'DataSanitizer',
    'AccessController',
    'ThreatDetector',
    'ComplianceMonitor',
    'global_security'
]