# 📡 API Documentation - Terragon Causal Discovery System

## Overview

The Terragon Causal Discovery System provides a comprehensive API for discovering causal relationships in datasets. The API supports three levels of functionality, from basic causal discovery to advanced scalable processing.

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd terragon-causal-discovery

# Install dependencies
pip install -r requirements.txt

# Basic usage
python -c "
from src.algorithms.scalable_causal import ScalableCausalDiscoveryModel
import pandas as pd
import numpy as np

# Generate sample data
data = pd.DataFrame({
    'X1': np.random.randn(100),
    'X2': np.random.randn(100),
    'X3': np.random.randn(100)
})

# Create and run model
model = ScalableCausalDiscoveryModel()
result = model.fit_discover(data)
print(f'Discovered {result.metadata[\"n_edges\"]} causal relationships')
"
```

## Core API Classes

### 1. CausalDiscoveryModel (Base Class)

The foundation class for all causal discovery algorithms.

#### Methods

##### `fit(data: pd.DataFrame) -> CausalDiscoveryModel`
Fit the model to training data.

**Parameters:**
- `data`: DataFrame with samples as rows, variables as columns

**Returns:** Self (for method chaining)

**Example:**
```python
from src.algorithms.base import SimpleLinearCausalModel

model = SimpleLinearCausalModel(threshold=0.3)
model.fit(data)
```

##### `discover(data: Optional[pd.DataFrame] = None) -> CausalResult`
Discover causal relationships.

**Parameters:**
- `data`: Optional new data. Uses fitted data if None.

**Returns:** CausalResult object containing discovered relationships

**Example:**
```python
result = model.discover()
adjacency_matrix = result.adjacency_matrix
confidence_scores = result.confidence_scores
```

##### `fit_discover(data: pd.DataFrame) -> CausalResult`
Convenience method that combines fit and discover.

**Parameters:**
- `data`: Input dataset

**Returns:** CausalResult object

**Example:**
```python
result = model.fit_discover(data)
```

### 2. SimpleLinearCausalModel (Generation 1)

Basic causal discovery using correlation-based methods.

#### Constructor
```python
SimpleLinearCausalModel(threshold: float = 0.3)
```

**Parameters:**
- `threshold`: Correlation threshold for causal edge detection (0.0-1.0)

#### Example Usage
```python
from src.algorithms.base import SimpleLinearCausalModel

# Create model with custom threshold
model = SimpleLinearCausalModel(threshold=0.5)

# Discover causal relationships
result = model.fit_discover(data)

# Access results
print(f"Method used: {result.method_used}")
print(f"Number of edges: {result.metadata['n_edges']}")
print(f"Variables: {result.metadata['variable_names']}")
```

### 3. RobustCausalDiscoveryModel (Generation 2)

Production-ready model with comprehensive validation, security, and error handling.

#### Constructor
```python
RobustCausalDiscoveryModel(
    base_model: Optional[CausalDiscoveryModel] = None,
    threshold: float = 0.3,
    enable_security: bool = True,
    strict_validation: bool = True,
    max_retries: int = 3,
    circuit_breaker_threshold: int = 5,
    user_id: Optional[str] = None
)
```

**Parameters:**
- `base_model`: Underlying model to wrap (defaults to SimpleLinearCausalModel)
- `threshold`: Correlation threshold for causal detection
- `enable_security`: Whether to enable security validation
- `strict_validation`: Whether to treat warnings as errors
- `max_retries`: Maximum number of retries on failures
- `circuit_breaker_threshold`: Number of failures before opening circuit breaker
- `user_id`: User identifier for audit logging

#### Enhanced Features

##### Security Validation
```python
model = RobustCausalDiscoveryModel(
    enable_security=True,
    user_id="researcher_001"
)

result = model.fit_discover(data)

# Check security assessment
print(f"Security risk level: {result.security_result.risk_level}")
print(f"Security issues: {result.security_result.issues}")
```

##### Quality Scoring
```python
result = model.fit_discover(data)

# Quality assessment
print(f"Quality score: {result.quality_score:.3f}")
print(f"Processing time: {result.processing_time:.3f}s")
print(f"Validation passed: {result.validation_result.is_valid}")
```

##### Health Monitoring
```python
# Get model health status
health_status = model.get_health_status()
print(f"Overall health: {health_status['overall_health']}")
print(f"Circuit breaker state: {health_status['circuit_breaker_state']}")
print(f"Success rate: {health_status['recent_success_rate']:.1%}")

# Get model information
model_info = model.get_model_info()
print(f"Processing history: {model_info['processing_history_length']} operations")
```

### 4. ScalableCausalDiscoveryModel (Generation 3)

High-performance model with advanced optimization, caching, and auto-scaling.

#### Constructor
```python
ScalableCausalDiscoveryModel(
    base_model: Optional[CausalDiscoveryModel] = None,
    enable_parallelization: bool = True,
    enable_caching: bool = True,
    enable_auto_scaling: bool = True,
    max_workers: Optional[int] = None,
    cache_size: int = 1000,
    batch_size: int = 100,
    optimization_level: str = "balanced"  # "speed", "memory", "balanced"
)
```

**Parameters:**
- `enable_parallelization`: Enable parallel processing
- `enable_caching`: Enable adaptive caching
- `enable_auto_scaling`: Enable auto-scaling
- `max_workers`: Maximum number of worker processes/threads
- `cache_size`: Maximum cache size
- `batch_size`: Batch size for processing
- `optimization_level`: Optimization strategy

#### Advanced Features

##### Performance Optimization
```python
# Speed-optimized configuration
speed_model = ScalableCausalDiscoveryModel(
    optimization_level="speed",
    max_workers=8,
    enable_caching=True
)

# Memory-optimized configuration
memory_model = ScalableCausalDiscoveryModel(
    optimization_level="memory",
    max_workers=2,
    batch_size=50
)

# Balanced configuration (default)
balanced_model = ScalableCausalDiscoveryModel(
    optimization_level="balanced"
)
```

##### Dataset Optimization
```python
model = ScalableCausalDiscoveryModel()

# Get optimization recommendations for your dataset
recommendations = model.optimize_for_dataset(data)
print(f"Recommended settings: {recommendations}")

# Apply recommendations manually or create optimized model
optimized_model = ScalableCausalDiscoveryModel(**recommendations)
```

##### Scalability Monitoring
```python
result = model.fit_discover(data)

# Get scalability report
report = model.get_scalability_report()
print(f"Operations completed: {report['operations_count']}")
print(f"Average processing time: {report['average_processing_time']:.3f}s")
print(f"Cache hit rate: {report['cache_hit_rate']:.1%}")
print(f"Parallelization efficiency: {report['parallelization_efficiency']:.1%}")
```

## Result Objects

### CausalResult

Basic result object containing discovered causal relationships.

#### Attributes
- `adjacency_matrix`: NxN numpy array (1 = causal edge, 0 = no edge)
- `confidence_scores`: NxN numpy array with confidence values
- `method_used`: String identifier of the algorithm used
- `metadata`: Dictionary with algorithm-specific information

#### Example Usage
```python
result = model.fit_discover(data)

# Access adjacency matrix
adj_matrix = result.adjacency_matrix
print(f"Shape: {adj_matrix.shape}")

# Find causal relationships
n_vars = adj_matrix.shape[0]
variable_names = result.metadata['variable_names']

for i in range(n_vars):
    for j in range(n_vars):
        if adj_matrix[i, j] == 1:
            confidence = result.confidence_scores[i, j]
            print(f"{variable_names[i]} -> {variable_names[j]} "
                  f"(confidence: {confidence:.3f})")
```

### RobustCausalResult

Enhanced result object with validation and security information.

#### Additional Attributes
- `validation_result`: ValidationResult object with data quality information
- `security_result`: SecurityResult object with security assessment
- `quality_score`: Overall quality score (0.0-1.0)
- `processing_time`: Processing time in seconds
- `warnings_raised`: List of warnings during processing

#### Example Usage
```python
result = model.fit_discover(data)

# Quality assessment
print(f"Quality score: {result.quality_score:.3f}")
print(f"Processing time: {result.processing_time:.3f}s")

# Validation details
if result.validation_result.is_valid:
    print("Data validation passed")
else:
    print(f"Validation errors: {result.validation_result.errors}")

# Security assessment
security = result.security_result
print(f"Security risk: {security.risk_level}")
if security.issues:
    print(f"Security issues: {security.issues}")
```

## Utility Classes

### DataProcessor

Comprehensive data processing and validation utilities.

#### Methods

##### `generate_synthetic_data(n_samples, n_variables, **kwargs) -> pd.DataFrame`
Generate synthetic data for testing.

```python
from src.utils.data_processing import DataProcessor

processor = DataProcessor()
data = processor.generate_synthetic_data(
    n_samples=1000,
    n_variables=5,
    noise_level=0.1,
    random_state=42
)
```

##### `clean_data(data: pd.DataFrame) -> pd.DataFrame`
Clean and preprocess data.

```python
cleaned_data = processor.clean_data(data)
```

##### `validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]`
Validate data quality.

```python
is_valid, issues = processor.validate_data(data)
if not is_valid:
    print(f"Data issues: {issues}")
```

### CausalMetrics

Evaluation metrics for causal discovery results.

```python
from src.utils.metrics import CausalMetrics

metrics = CausalMetrics()

# Compare discovered graph with ground truth
precision = metrics.precision(discovered_matrix, true_matrix)
recall = metrics.recall(discovered_matrix, true_matrix)
f1_score = metrics.f1_score(discovered_matrix, true_matrix)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")
```

## Error Handling

### Exception Types

#### `CausalDiscoveryError`
Base exception for causal discovery operations.

#### `DataValidationError`
Raised when input data validation fails.

#### `AlgorithmError`
Raised when causal discovery algorithm fails.

#### `ConvergenceError`
Raised when algorithm fails to converge.

### Example Error Handling
```python
from src.utils.error_handling import CausalDiscoveryError, DataValidationError

try:
    result = model.fit_discover(data)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
except CausalDiscoveryError as e:
    print(f"Causal discovery failed: {e}")
    # Check error context
    if hasattr(e, 'context'):
        print(f"Error context: {e.context}")
```

## Configuration

### Environment Variables

- `CAUSAL_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `CAUSAL_ENABLE_SECURITY`: Enable security features (true/false)
- `CAUSAL_MAX_WORKERS`: Default maximum number of workers
- `CAUSAL_CACHE_SIZE`: Default cache size
- `CAUSAL_CONFIG_PATH`: Path to configuration file

### Configuration File Example

```yaml
# production_config.yaml
performance:
  max_workers: 8
  cache_size: 2000
  optimization_level: "balanced"

security:
  enable_audit_logging: true
  data_encryption: true

monitoring:
  health_check_interval: 30
  metrics_collection: true
```

## Best Practices

### 1. Data Preparation
```python
# Always validate data before processing
processor = DataProcessor()
is_valid, issues = processor.validate_data(data)

if not is_valid:
    print(f"Data issues found: {issues}")
    # Fix issues before proceeding

# Clean data
cleaned_data = processor.clean_data(data)
```

### 2. Model Selection
```python
# For exploration and small datasets
simple_model = SimpleLinearCausalModel()

# For production with reliability requirements
robust_model = RobustCausalDiscoveryModel(
    enable_security=True,
    user_id="production_user"
)

# For large-scale processing
scalable_model = ScalableCausalDiscoveryModel(
    optimization_level="speed",
    max_workers=8
)
```

### 3. Performance Optimization
```python
# Get dataset-specific recommendations
model = ScalableCausalDiscoveryModel()
recommendations = model.optimize_for_dataset(data)

# Apply recommendations
optimized_model = ScalableCausalDiscoveryModel(**recommendations)

# Monitor performance
result = optimized_model.fit_discover(data)
report = optimized_model.get_scalability_report()
```

### 4. Error Recovery
```python
# Use robust model for automatic error recovery
model = RobustCausalDiscoveryModel(
    max_retries=3,
    circuit_breaker_threshold=5
)

try:
    result = model.fit_discover(data)
except Exception as e:
    # Check health status
    health = model.get_health_status()
    if health['circuit_breaker_state'] == 'OPEN':
        # Circuit breaker is open, wait and retry later
        print("Circuit breaker open, retrying later...")
```

## Performance Guidelines

### Dataset Size Recommendations

| Dataset Size | Recommended Model | Configuration |
|-------------|------------------|---------------|
| < 1K samples | SimpleLinearCausalModel | Default settings |
| 1K-10K samples | RobustCausalDiscoveryModel | Default settings |
| 10K-100K samples | ScalableCausalDiscoveryModel | optimization_level="balanced" |
| > 100K samples | ScalableCausalDiscoveryModel | optimization_level="speed" |

### Memory Optimization
```python
# For memory-constrained environments
memory_model = ScalableCausalDiscoveryModel(
    optimization_level="memory",
    batch_size=100,
    max_workers=2
)
```

### Speed Optimization
```python
# For maximum performance
speed_model = ScalableCausalDiscoveryModel(
    optimization_level="speed",
    max_workers=8,
    enable_caching=True
)
```

---

For additional examples and use cases, see the `examples/` directory in the repository.