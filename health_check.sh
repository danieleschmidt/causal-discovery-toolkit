#!/bin/bash
set -e

echo "🏥 Health Check - Causal Discovery System"
echo "========================================"

# Check container health
echo "📦 Container Status:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "🔍 Service Health Checks:"

# API Health Check
if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ API Service: Healthy"
else
    echo "❌ API Service: Unhealthy"
fi

# Prometheus Health Check
if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus: Healthy"
else
    echo "❌ Prometheus: Unhealthy"
fi

# Grafana Health Check
if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana: Healthy"
else
    echo "❌ Grafana: Unhealthy"
fi

echo ""
echo "📊 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo "📋 Recent Logs (last 10 lines):"
docker-compose -f docker-compose.prod.yml logs --tail=10 causal-discovery
