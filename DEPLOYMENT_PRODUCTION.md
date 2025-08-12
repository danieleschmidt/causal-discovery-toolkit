# Bioneuro-Olfactory Fusion Causal Discovery Toolkit - Production Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the Bioneuro-Olfactory Fusion Causal Discovery Toolkit in production environments. The toolkit is designed for research applications in neuroscience, particularly for analyzing olfactory neural signals and discovering causal relationships in multi-modal sensory data.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.8 or higher (3.10+ recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 1GB for installation, additional space for data and results
- **CPU**: Multi-core processor recommended for parallel processing

### Dependencies

Core dependencies are automatically installed via pip:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0 (for visualizations)
- psutil >= 5.0.0 (for monitoring)

Optional dependencies for advanced features:
- torch >= 2.0.0 (for deep learning models)
- jax >= 0.4.0 (for high-performance computing)
- wandb >= 0.15.0 (for experiment tracking)

## Installation

### Option 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/bioneuro-olfactory-fusion.git
cd bioneuro-olfactory-fusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Production Installation

```bash
# Create production virtual environment
python -m venv /opt/bioneuro-causal-discovery
source /opt/bioneuro-causal-discovery/bin/activate

# Install the package
pip install -r requirements.txt
pip install .
```

### Option 3: Docker Deployment

```dockerfile
# Use provided Dockerfile
docker build -t bioneuro-causal-discovery .
docker run -p 8000:8000 -v /path/to/data:/app/data bioneuro-causal-discovery
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Core configuration
BIONEURO_LOG_LEVEL=INFO
BIONEURO_DATA_DIR=/path/to/data
BIONEURO_RESULTS_DIR=/path/to/results
BIONEURO_CACHE_DIR=/path/to/cache

# Performance settings
BIONEURO_MAX_WORKERS=4
BIONEURO_MEMORY_LIMIT=8GB
BIONEURO_ENABLE_GPU=false

# Security settings
BIONEURO_SECURE_MODE=true
BIONEURO_API_KEY=your_api_key_here

# Monitoring settings
BIONEURO_ENABLE_MONITORING=true
BIONEURO_METRICS_ENDPOINT=http://localhost:9090
```

### Configuration Files

Create `config/production.json`:

```json
{
  "data_processing": {
    "sampling_rate_hz": 1000.0,
    "filter_low_cutoff": 1.0,
    "filter_high_cutoff": 100.0,
    "normalization_method": "z_score",
    "artifact_removal": true,
    "baseline_correction": true
  },
  "causal_discovery": {
    "receptor_sensitivity_threshold": 0.15,
    "neural_firing_threshold": 10.0,
    "temporal_window_ms": 100,
    "cross_modal_integration": true,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95
  },
  "performance": {
    "enable_caching": true,
    "max_cache_size": "2GB",
    "parallel_processing": true,
    "max_concurrent_jobs": 4,
    "memory_monitoring": true
  },
  "security": {
    "validate_inputs": true,
    "sanitize_outputs": true,
    "audit_logging": true,
    "secure_temp_files": true
  }
}
```

## Deployment Architectures

### 1. Single-Node Research Environment

For individual researchers or small teams:

```bash
# Install on local workstation or server
pip install -e .

# Run interactive analysis
python -m jupyter lab
# Or run batch processing
python scripts/run_production_benchmark.py --config standard_benchmark
```

### 2. High-Performance Computing (HPC) Cluster

For large-scale research projects:

```bash
# Submit job to SLURM scheduler
#!/bin/bash
#SBATCH --job-name=bioneuro-causal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load python/3.10
source /opt/bioneuro-causal-discovery/bin/activate

python scripts/run_production_benchmark.py --config comprehensive
```

### 3. Cloud Deployment (AWS/GCP/Azure)

For scalable research infrastructure:

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bioneuro-causal-discovery
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bioneuro-causal
  template:
    metadata:
      labels:
        app: bioneuro-causal
    spec:
      containers:
      - name: bioneuro-causal
        image: bioneuro-causal-discovery:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: BIONEURO_LOG_LEVEL
          value: "INFO"
        - name: BIONEURO_MAX_WORKERS
          value: "4"
```

### 4. Containerized Microservices

For API-based deployment:

```bash
# API server
docker run -d \
  --name bioneuro-api \
  -p 8000:8000 \
  -v /data:/app/data \
  -e BIONEURO_API_MODE=true \
  bioneuro-causal-discovery:latest

# Background processing workers
docker run -d \
  --name bioneuro-worker \
  -v /data:/app/data \
  -e BIONEURO_WORKER_MODE=true \
  bioneuro-causal-discovery:latest
```

## Production Usage

### 1. Basic Causal Discovery

```python
from algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel
import pandas as pd

# Load your data
data = pd.read_csv('neural_recordings.csv')

# Initialize model
model = OlfactoryNeuralCausalModel(
    receptor_sensitivity_threshold=0.15,
    neural_firing_threshold=10.0,
    temporal_window_ms=100
)

# Discover causal relationships
result = model.fit_discover(data)

# Access results
print(f"Found {result.metadata['n_causal_edges']} causal relationships")
print(f"Neural pathways: {result.neural_pathways}")
```

### 2. Batch Processing

```python
from experiments.bioneuro_research_suite import BioneuroResearchSuite, ResearchExperimentConfig

# Configure experiment
config = ResearchExperimentConfig(
    experiment_name="production_analysis",
    n_samples_range=[1000, 2000],
    n_receptors_range=[10, 20],
    n_neurons_range=[8, 16],
    save_results=True,
    output_dir="/results/production"
)

# Run comprehensive analysis
suite = BioneuroResearchSuite(config)
results = suite.run_comprehensive_study()
```

### 3. Real-time Processing

```python
from utils.bioneuro_data_processing import BioneuroDataProcessor
from algorithms.bioneuro_olfactory import OlfactoryNeuralCausalModel

# Initialize components
processor = BioneuroDataProcessor()
model = OlfactoryNeuralCausalModel()

# Process streaming data
def process_data_stream(data_stream):
    for data_chunk in data_stream:
        # Process data
        processed_data = processor.process_olfactory_signals(data_chunk)
        
        # Update model
        model.fit(processed_data)
        
        # Get real-time results
        result = model.discover()
        yield result
```

## Monitoring and Maintenance

### 1. Performance Monitoring

```python
from utils.monitoring import PerformanceMonitor

# Initialize monitoring
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Your analysis code here
# ...

# Get performance metrics
metrics = monitor.get_performance_summary()
print(f"Peak memory usage: {metrics['peak_memory']} MB")
print(f"Average CPU usage: {metrics['avg_cpu']}%")
```

### 2. Health Checks

```bash
# Check system health
python scripts/health_check.py

# Expected output:
# ✅ Memory usage: 45% (3.6GB / 8GB)
# ✅ CPU usage: 23%
# ✅ Disk space: 78% available
# ✅ Dependencies: All installed
# ✅ Configuration: Valid
```

### 3. Log Management

```bash
# Configure log rotation
# /etc/logrotate.d/bioneuro-causal

/var/log/bioneuro-causal/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 bioneuro bioneuro
}
```

### 4. Backup and Recovery

```bash
# Backup configuration and results
#!/bin/bash
BACKUP_DIR="/backup/bioneuro-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration
cp -r config/ $BACKUP_DIR/
cp -r results/ $BACKUP_DIR/

# Backup trained models
cp -r models/ $BACKUP_DIR/

# Create archive
tar -czf "${BACKUP_DIR}.tar.gz" $BACKUP_DIR
```

## Security Considerations

### 1. Data Protection

- **Encryption at Rest**: Use encrypted storage for sensitive neural data
- **Encryption in Transit**: Use HTTPS/TLS for API communications
- **Access Control**: Implement role-based access controls
- **Audit Logging**: Enable comprehensive audit trails

### 2. Input Validation

```python
from utils.security import DataSecurityValidator

# Validate all inputs
validator = DataSecurityValidator()
is_safe = validator.validate_data_security(input_data)
if not is_safe:
    raise SecurityError("Invalid or potentially malicious input")
```

### 3. Secure Configuration

```bash
# Set proper file permissions
chmod 600 config/production.json
chown bioneuro:bioneuro config/production.json

# Use secrets management
export BIONEURO_API_KEY=$(vault kv get -field=api_key secret/bioneuro)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce memory usage
   export BIONEURO_MAX_WORKERS=2
   export BIONEURO_MEMORY_LIMIT=4GB
   ```

2. **Import Errors**
   ```bash
   # Verify installation
   pip list | grep causal-discovery
   python -c "import algorithms.bioneuro_olfactory; print('OK')"
   ```

3. **Performance Issues**
   ```bash
   # Enable profiling
   export BIONEURO_PROFILE=true
   python -m cProfile -o profile.stats your_script.py
   ```

4. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER /opt/bioneuro-causal-discovery
   chmod +x scripts/*.py
   ```

### Debug Mode

```bash
# Enable debug logging
export BIONEURO_LOG_LEVEL=DEBUG
export BIONEURO_DEBUG=true

# Run with debug output
python -u your_script.py 2>&1 | tee debug.log
```

## Performance Optimization

### 1. Hardware Optimization

- **CPU**: Use multi-core processors (4+ cores recommended)
- **Memory**: 8GB+ for medium datasets, 16GB+ for large datasets
- **Storage**: SSD storage for faster I/O operations
- **GPU**: Optional, for deep learning components

### 2. Software Optimization

```python
# Use optimized algorithms
from algorithms.optimized import OptimizedCausalModel

model = OptimizedCausalModel(
    parallel_processing=True,
    use_gpu=True,
    memory_efficient=True
)
```

### 3. Distributed Processing

```python
from algorithms.distributed_discovery import DistributedCausalDiscovery

# Configure distributed processing
distributed_model = DistributedCausalDiscovery(
    n_workers=4,
    chunk_size=1000,
    backend='threading'
)
```

## Support and Maintenance

### Getting Help

- **Documentation**: Comprehensive API documentation available
- **Examples**: See `examples/` directory for usage examples
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join research community discussions

### Updates and Patches

```bash
# Check for updates
git pull origin main
pip install -r requirements.txt --upgrade

# Run compatibility check
python scripts/compatibility_check.py
```

### Contributing

For contributing to the project:

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## License and Citation

This toolkit is released under the MIT License. If you use this software in your research, please cite:

```bibtex
@software{bioneuro_olfactory_causal_discovery,
  title={Bioneuro-Olfactory Fusion Causal Discovery Toolkit},
  author={Schmidt, Daniel},
  year={2025},
  url={https://github.com/danieleschmidt/bioneuro-olfactory-fusion},
  license={MIT}
}
```

---

For technical support or research collaboration inquiries, contact: daniel@terragonlabs.ai