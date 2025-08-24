#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE=$1
RESTORE_DIR="/tmp/causal_restore_$(date +%s)"

echo "🔄 Restoring from backup: $BACKUP_FILE"

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Find backup directory
BACKUP_CONTENT=$(find "$RESTORE_DIR" -name "backup_*" -type d | head -1)

if [ -z "$BACKUP_CONTENT" ]; then
    echo "❌ Invalid backup file"
    exit 1
fi

# Stop current services
echo "🛑 Stopping current services..."
docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

# Restore files
echo "📂 Restoring configuration files..."
cp "$BACKUP_CONTENT/production_config.yaml" . 2>/dev/null || true
cp "$BACKUP_CONTENT/Dockerfile" . 2>/dev/null || true
cp "$BACKUP_CONTENT/docker-compose.prod.yml" . 2>/dev/null || true
cp -r "$BACKUP_CONTENT/monitoring" . 2>/dev/null || true

# Restore data
echo "💾 Restoring data..."
cp -r "$BACKUP_CONTENT/data" . 2>/dev/null || echo "No data to restore"

# Start services
echo "🚀 Starting restored services..."
docker-compose -f docker-compose.prod.yml up -d

# Cleanup
rm -rf "$RESTORE_DIR"

echo "✅ Restore completed successfully"
