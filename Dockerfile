# Production-ready Docker image for Causal Discovery Toolkit
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.ai>"
LABEL description="Causal Discovery Toolkit - Production Ready"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import causal_discovery_toolkit; print('OK')" || exit 1

# Expose port (if running as web service)
EXPOSE 8000

# Default command
CMD ["python", "-m", "causal_discovery_toolkit.cli", "--help"]