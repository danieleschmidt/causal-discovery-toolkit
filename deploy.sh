#!/bin/bash
set -e

echo "🚀 Starting Causal Discovery Production Deployment"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Create necessary directories
mkdir -p logs data monitoring/grafana/{dashboards,datasources}

# Build and start services
echo "📦 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

echo "🔧 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Verify deployment
echo "🔍 Verifying deployment..."
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up (healthy)"; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Causal Discovery API: http://localhost:8080"
    echo "   Prometheus Monitoring: http://localhost:9090"
    echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo ""
    echo "📊 View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "🛑 Stop services: docker-compose -f docker-compose.prod.yml down"
else
    echo "❌ Deployment verification failed"
    echo "📋 Service status:"
    docker-compose -f docker-compose.prod.yml ps
    exit 1
fi
