"""
Production Deployment System: Enterprise-Ready Deployment Framework
===================================================================

Complete production deployment system for causal discovery algorithms with
containerization, orchestration, monitoring, and enterprise integration.

Deployment Features:
- Docker containerization with multi-stage builds
- Kubernetes orchestration and auto-scaling
- Production monitoring and observability
- Enterprise security and compliance
- CI/CD pipeline integration
- Health checks and self-healing
- Load balancing and traffic management
"""

import os
import sys
import json
import yaml
import logging
import subprocess
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import docker
import requests

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ServiceStatus(Enum):
    """Service deployment status."""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: DeploymentEnvironment
    replicas: int = 3
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    cpu_request: str = "500m"
    memory_request: str = "1Gi"
    enable_autoscaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percentage: int = 70
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    image_tag: str = "latest"
    registry: str = "terragonlabs/causal-discovery"

class ProductionDeploymentSystem:
    """
    Complete production deployment system for enterprise causal discovery.
    
    This system provides:
    1. Containerized deployment with Docker
    2. Kubernetes orchestration and scaling  
    3. Production monitoring and alerting
    4. Enterprise security and compliance
    5. CI/CD pipeline integration
    6. Health monitoring and self-healing
    7. Load balancing and high availability
    
    Enterprise Features:
    - Multi-environment deployment (dev/staging/prod)
    - Auto-scaling based on load and resource usage
    - Zero-downtime rolling updates
    - Comprehensive health checking
    - Production monitoring and observability
    - Security scanning and compliance
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize production deployment system.
        
        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig(environment=DeploymentEnvironment.PRODUCTION)
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
            logging.info("Docker client initialized")
        except Exception as e:
            self.docker_client = None
            self.docker_available = False
            logging.warning(f"Docker not available: {e}")
        
        # Deployment state
        self.deployment_status = {}
        self.service_health = {}
        
        logging.info(f"Production deployment system initialized for {self.config.environment.value}")
    
    def create_dockerfile(self) -> str:
        """Create optimized production Dockerfile."""
        
        dockerfile_content = '''# Multi-stage production Dockerfile for Causal Discovery System
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels for metadata
LABEL maintainer="Terragon Labs <support@terragonlabs.ai>" \\
      org.label-schema.build-date=$BUILD_DATE \\
      org.label-schema.version=$VERSION \\
      org.label-schema.vcs-ref=$VCS_REF \\
      org.label-schema.vcs-url="https://github.com/terragonlabs/causal-discovery" \\
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY setup.py ./

# Install application
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r causalapp && useradd -r -g causalapp causalapp

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

# Copy production configuration
COPY production_config.yaml ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \\
    chown -R causalapp:causalapp /app

# Switch to non-root user
USER causalapp

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "-m", "src.api.main"]
'''
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        return dockerfile_content
    
    def create_docker_entrypoint(self) -> str:
        """Create Docker entrypoint script."""
        
        entrypoint_content = '''#!/bin/bash
set -e

# Wait for dependencies
echo "Waiting for dependencies..."
sleep 5

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m src.database.migrate
fi

# Start application
echo "Starting Causal Discovery System..."
exec "$@"
'''
        
        with open('docker-entrypoint.sh', 'w') as f:
            f.write(entrypoint_content)
        
        # Make executable
        os.chmod('docker-entrypoint.sh', 0o755)
        
        return entrypoint_content
    
    def create_kubernetes_manifests(self) -> Dict[str, str]:
        """Create Kubernetes deployment manifests."""
        
        manifests = {}
        
        # Deployment manifest
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'causal-discovery',
                'labels': {
                    'app': 'causal-discovery',
                    'version': self.config.image_tag,
                    'environment': self.config.environment.value
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'causal-discovery'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'causal-discovery',
                            'version': self.config.image_tag
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'causal-discovery',
                            'image': f"{self.config.registry}:{self.config.image_tag}",
                            'ports': [{'containerPort': 8000}],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': self.config.environment.value},
                                {'name': 'LOG_LEVEL', 'value': 'INFO'},
                                {'name': 'METRICS_ENABLED', 'value': str(self.config.enable_monitoring).lower()}
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': self.config.health_check_path,
                                    'port': 8000
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30,
                                'timeoutSeconds': 10,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': self.config.readiness_probe_path,
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            }
                        }],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        }
                    }
                }
            }
        }
        
        # Service manifest
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'causal-discovery-service',
                'labels': {
                    'app': 'causal-discovery'
                }
            },
            'spec': {
                'selector': {
                    'app': 'causal-discovery'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
        
        # Horizontal Pod Autoscaler
        if self.config.enable_autoscaling:
            hpa = {
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': 'causal-discovery-hpa'
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': 'causal-discovery'
                    },
                    'minReplicas': self.config.min_replicas,
                    'maxReplicas': self.config.max_replicas,
                    'metrics': [{
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': self.config.target_cpu_percentage
                            }
                        }
                    }]
                }
            }
            manifests['hpa.yaml'] = yaml.dump(hpa, default_flow_style=False)
        
        # ConfigMap for application configuration
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'causal-discovery-config'
            },
            'data': {
                'config.yaml': yaml.dump({
                    'environment': self.config.environment.value,
                    'logging': {
                        'level': 'INFO',
                        'format': 'json'
                    },
                    'monitoring': {
                        'enabled': self.config.enable_monitoring,
                        'metrics_port': 9090
                    },
                    'security': {
                        'enable_auth': True,
                        'jwt_secret': '${JWT_SECRET}',
                        'cors_origins': ['*'] if self.config.environment == DeploymentEnvironment.DEVELOPMENT else []
                    }
                })
            }
        }
        
        # Convert to YAML strings
        manifests['deployment.yaml'] = yaml.dump(deployment, default_flow_style=False)
        manifests['service.yaml'] = yaml.dump(service, default_flow_style=False)
        manifests['configmap.yaml'] = yaml.dump(configmap, default_flow_style=False)
        
        return manifests
    
    def build_docker_image(self) -> bool:
        """Build production Docker image."""
        
        if not self.docker_available:
            logging.error("Docker not available for image building")
            return False
        
        try:
            logging.info("Building production Docker image...")
            
            # Create Dockerfile if it doesn't exist
            if not Path('Dockerfile').exists():
                self.create_dockerfile()
            
            # Create entrypoint script
            self.create_docker_entrypoint()
            
            # Build arguments
            build_args = {
                'BUILD_DATE': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'VERSION': self.config.image_tag,
                'VCS_REF': self._get_git_commit_hash()
            }
            
            # Build image
            image, build_logs = self.docker_client.images.build(
                path='.',
                tag=f"{self.config.registry}:{self.config.image_tag}",
                buildargs=build_args,
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logging.info(f"Docker build: {log['stream'].strip()}")
            
            logging.info(f"Docker image built successfully: {image.id}")
            return True
            
        except Exception as e:
            logging.error(f"Docker image build failed: {e}")
            return False
    
    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
        except:
            return 'unknown'
    
    def deploy_to_kubernetes(self) -> bool:
        """Deploy to Kubernetes cluster."""
        
        try:
            logging.info(f"Deploying to Kubernetes ({self.config.environment.value})...")
            
            # Create Kubernetes manifests
            manifests = self.create_kubernetes_manifests()
            
            # Apply manifests
            for filename, content in manifests.items():
                # Write manifest to file
                with open(filename, 'w') as f:
                    f.write(content)
                
                # Apply to cluster
                result = subprocess.run(['kubectl', 'apply', '-f', filename],
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info(f"Applied {filename} successfully")
                else:
                    logging.error(f"Failed to apply {filename}: {result.stderr}")
                    return False
            
            # Wait for deployment to be ready
            self._wait_for_deployment_ready()
            
            logging.info("Kubernetes deployment completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _wait_for_deployment_ready(self, timeout: int = 600):
        """Wait for deployment to be ready."""
        
        logging.info("Waiting for deployment to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check deployment status
                result = subprocess.run([
                    'kubectl', 'rollout', 'status', 'deployment/causal-discovery',
                    '--timeout=60s'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info("Deployment is ready")
                    return True
                    
            except Exception as e:
                logging.warning(f"Error checking deployment status: {e}")
            
            time.sleep(10)
        
        logging.error("Deployment readiness timeout")
        return False
    
    def create_monitoring_stack(self) -> Dict[str, str]:
        """Create monitoring and observability stack."""
        
        monitoring_manifests = {}
        
        # Prometheus ServiceMonitor
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'causal-discovery-metrics',
                'labels': {
                    'app': 'causal-discovery'
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': 'causal-discovery'
                    }
                },
                'endpoints': [{
                    'port': 'metrics',
                    'interval': '30s',
                    'path': '/metrics'
                }]
            }
        }
        
        # Grafana Dashboard ConfigMap
        dashboard = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'causal-discovery-dashboard',
                'labels': {
                    'grafana_dashboard': '1'
                }
            },
            'data': {
                'dashboard.json': json.dumps({
                    'dashboard': {
                        'title': 'Causal Discovery System',
                        'panels': [
                            {
                                'title': 'Request Rate',
                                'type': 'graph',
                                'targets': [{
                                    'expr': 'rate(http_requests_total[5m])'
                                }]
                            },
                            {
                                'title': 'Response Time',
                                'type': 'graph',
                                'targets': [{
                                    'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
                                }]
                            },
                            {
                                'title': 'Error Rate',
                                'type': 'graph',
                                'targets': [{
                                    'expr': 'rate(http_requests_total{status=~"5.."}[5m])'
                                }]
                            }
                        ]
                    }
                })
            }
        }
        
        monitoring_manifests['servicemonitor.yaml'] = yaml.dump(service_monitor, default_flow_style=False)
        monitoring_manifests['dashboard.yaml'] = yaml.dump(dashboard, default_flow_style=False)
        
        return monitoring_manifests
    
    def check_service_health(self) -> Dict[str, ServiceStatus]:
        """Check health of deployed services."""
        
        health_status = {}
        
        try:
            # Check Kubernetes deployment health
            result = subprocess.run([
                'kubectl', 'get', 'deployment', 'causal-discovery', '-o', 'json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment_info = json.loads(result.stdout)
                status = deployment_info.get('status', {})
                
                ready_replicas = status.get('readyReplicas', 0)
                desired_replicas = status.get('replicas', 0)
                
                if ready_replicas == desired_replicas and ready_replicas > 0:
                    health_status['deployment'] = ServiceStatus.HEALTHY
                elif ready_replicas > 0:
                    health_status['deployment'] = ServiceStatus.STARTING
                else:
                    health_status['deployment'] = ServiceStatus.UNHEALTHY
            else:
                health_status['deployment'] = ServiceStatus.FAILED
                
            # Check service endpoint health
            service_health = self._check_endpoint_health()
            health_status.update(service_health)
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            health_status['deployment'] = ServiceStatus.FAILED
        
        self.service_health = health_status
        return health_status
    
    def _check_endpoint_health(self) -> Dict[str, ServiceStatus]:
        """Check health of service endpoints."""
        
        endpoint_status = {}
        
        try:
            # Get service endpoint
            result = subprocess.run([
                'kubectl', 'get', 'service', 'causal-discovery-service', 
                '-o', 'jsonpath={.status.loadBalancer.ingress[0].ip}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                service_ip = result.stdout.strip()
                health_url = f"http://{service_ip}{self.config.health_check_path}"
                
                # Make health check request
                response = requests.get(health_url, timeout=10)
                
                if response.status_code == 200:
                    endpoint_status['health_endpoint'] = ServiceStatus.HEALTHY
                else:
                    endpoint_status['health_endpoint'] = ServiceStatus.UNHEALTHY
                    
            else:
                endpoint_status['health_endpoint'] = ServiceStatus.STARTING
                
        except requests.RequestException:
            endpoint_status['health_endpoint'] = ServiceStatus.UNHEALTHY
        except Exception as e:
            logging.warning(f"Endpoint health check failed: {e}")
            endpoint_status['health_endpoint'] = ServiceStatus.FAILED
        
        return endpoint_status
    
    def rollback_deployment(self) -> bool:
        """Rollback to previous deployment version."""
        
        try:
            logging.info("Rolling back deployment...")
            
            result = subprocess.run([
                'kubectl', 'rollout', 'undo', 'deployment/causal-discovery'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Wait for rollback to complete
                self._wait_for_deployment_ready()
                logging.info("Deployment rollback completed successfully")
                return True
            else:
                logging.error(f"Rollback failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Rollback failed: {e}")
            return False
    
    def scale_deployment(self, replicas: int) -> bool:
        """Scale deployment to specified number of replicas."""
        
        try:
            logging.info(f"Scaling deployment to {replicas} replicas...")
            
            result = subprocess.run([
                'kubectl', 'scale', 'deployment/causal-discovery', 
                f'--replicas={replicas}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Deployment scaled to {replicas} replicas")
                return True
            else:
                logging.error(f"Scaling failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Scaling failed: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        
        status = {
            'environment': self.config.environment.value,
            'health_status': self.check_service_health(),
            'deployment_info': {},
            'resource_usage': {},
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Get deployment info
            result = subprocess.run([
                'kubectl', 'get', 'deployment', 'causal-discovery', '-o', 'json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment_info = json.loads(result.stdout)
                status['deployment_info'] = {
                    'replicas': deployment_info.get('status', {}).get('replicas', 0),
                    'ready_replicas': deployment_info.get('status', {}).get('readyReplicas', 0),
                    'image': deployment_info.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [{}])[0].get('image', ''),
                    'creation_timestamp': deployment_info.get('metadata', {}).get('creationTimestamp', '')
                }
        
        except Exception as e:
            logging.error(f"Failed to get deployment status: {e}")
        
        return status
    
    def cleanup_deployment(self) -> bool:
        """Clean up deployment resources."""
        
        try:
            logging.info("Cleaning up deployment resources...")
            
            # Delete deployment
            subprocess.run(['kubectl', 'delete', 'deployment', 'causal-discovery'], 
                         capture_output=True)
            
            # Delete service
            subprocess.run(['kubectl', 'delete', 'service', 'causal-discovery-service'], 
                         capture_output=True)
            
            # Delete HPA if exists
            subprocess.run(['kubectl', 'delete', 'hpa', 'causal-discovery-hpa'], 
                         capture_output=True)
            
            # Delete ConfigMap
            subprocess.run(['kubectl', 'delete', 'configmap', 'causal-discovery-config'], 
                         capture_output=True)
            
            logging.info("Deployment cleanup completed")
            return True
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
            return False

def deploy_production_system(environment: str = "production",
                           replicas: int = 3,
                           enable_monitoring: bool = True) -> bool:
    """
    Deploy complete production causal discovery system.
    
    Args:
        environment: Deployment environment (development/staging/production)
        replicas: Number of replicas to deploy
        enable_monitoring: Enable monitoring stack
        
    Returns:
        True if deployment successful, False otherwise
    """
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=DeploymentEnvironment(environment),
        replicas=replicas,
        enable_monitoring=enable_monitoring,
        enable_autoscaling=True
    )
    
    # Initialize deployment system
    deployment_system = ProductionDeploymentSystem(config)
    
    try:
        # Build Docker image
        logging.info("Step 1: Building Docker image...")
        if not deployment_system.build_docker_image():
            logging.error("Docker image build failed")
            return False
        
        # Deploy to Kubernetes
        logging.info("Step 2: Deploying to Kubernetes...")
        if not deployment_system.deploy_to_kubernetes():
            logging.error("Kubernetes deployment failed")
            return False
        
        # Setup monitoring if enabled
        if enable_monitoring:
            logging.info("Step 3: Setting up monitoring...")
            monitoring_manifests = deployment_system.create_monitoring_stack()
            
            for filename, content in monitoring_manifests.items():
                with open(f"monitoring/{filename}", 'w') as f:
                    f.write(content)
        
        # Verify deployment health
        logging.info("Step 4: Verifying deployment health...")
        health_status = deployment_system.check_service_health()
        
        if all(status == ServiceStatus.HEALTHY for status in health_status.values()):
            logging.info("🎉 Production deployment completed successfully!")
            logging.info("System is healthy and ready to serve requests.")
            return True
        else:
            logging.warning("Deployment completed but some services are not healthy")
            logging.info(f"Health status: {health_status}")
            return False
            
    except Exception as e:
        logging.error(f"Production deployment failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Deploy production system
    success = deploy_production_system(
        environment="production",
        replicas=3,
        enable_monitoring=True
    )
    
    if success:
        print("✅ Production deployment successful!")
        sys.exit(0)
    else:
        print("❌ Production deployment failed!")
        sys.exit(1)