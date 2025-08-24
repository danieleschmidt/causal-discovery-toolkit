FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY production_config.yaml ./

# Create non-root user
RUN useradd -m -u 1000 causal && chown -R causal:causal /app
USER causal

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CAUSAL_CONFIG_PATH=/app/production_config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.algorithms.base; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.algorithms.scalable_causal"]
