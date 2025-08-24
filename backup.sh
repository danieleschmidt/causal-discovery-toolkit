#!/bin/bash
set -e

BACKUP_DIR="/root/causal_discovery_backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$DATE"

echo "💾 Creating backup: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup configuration
cp -r production_config.yaml "$BACKUP_PATH/"
cp -r monitoring/ "$BACKUP_PATH/" 2>/dev/null || true

# Backup data and logs
cp -r data/ "$BACKUP_PATH/" 2>/dev/null || true
cp -r logs/ "$BACKUP_PATH/" 2>/dev/null || true

# Backup Docker configurations
cp Dockerfile docker-compose.prod.yml "$BACKUP_PATH/"

# Create metadata
cat > "$BACKUP_PATH/backup_info.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "services": $(docker-compose -f docker-compose.prod.yml ps --format json 2>/dev/null || echo '[]')
}
EOF

# Compress backup
cd "$BACKUP_DIR"
tar -czf "backup_$DATE.tar.gz" "backup_$DATE"
rm -rf "backup_$DATE"

echo "✅ Backup created: $BACKUP_DIR/backup_$DATE.tar.gz"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete
