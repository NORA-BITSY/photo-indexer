#!/bin/bash
# Backup script for photo-indexer: photo_index.db + staging/ directory
# Usage: ./scripts/backup.sh [BACKUP_DIR]
set -euo pipefail

BACKUP_DIR="${1:-/srv/photo-indexer/backups}"
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/photo-indexer-backup-${DATE}.tar.gz"

mkdir -p "$BACKUP_DIR"

echo "=== Photo Indexer Backup ==="
echo "Date: $(date)"
echo "Backup dir: $BACKUP_DIR"
echo ""

# Check required files exist
DB_PATH="${PHOTO_INDEX_DB:-./photo_index.db}"
if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: Database not found at $DB_PATH"
    exit 1
fi

STAGING_LOCATION="${STAGING_LOCATION:-./staging}"

# Create backup archive
echo "Creating backup archive..."
tar -czf "$BACKUP_FILE" \
    "$DB_PATH" \
    "$STAGING_LOCATION" \
    .env \
    docker-compose.yml \
    sql/enrich_schema.sql \
    2>/dev/null

echo "Backup created: $BACKUP_FILE"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo ""

# Retention: keep last 7 daily backups
echo "Cleaning old backups (keeping last 7)..."
ls -t "${BACKUP_DIR}"/photo-indexer-backup-*.tar.gz 2>/dev/null | tail -n +8 | xargs -r rm -f
echo "Remaining backups:"
ls -lh "${BACKUP_DIR}"/photo-indexer-backup-*.tar.gz 2>/dev/null || echo "  (none)"

echo ""
echo "=== Backup complete ==="
