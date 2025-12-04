#!/bin/bash
set -euo pipefail

BACKUP_DIR="./data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

echo "[BACKUP] Starting Tessrax System Backup: $TIMESTAMP"

# 1. Postgres Dump
echo "  > Dumping Database..."
docker exec tessrax_db pg_dumpall -c -U tessrax > "$BACKUP_DIR/db_${TIMESTAMP}.sql"

# 2. Vault Snapshot
echo "  > Snapshotting Vault..."
# Requires Vault Token to be set in environment
# docker exec -e VAULT_TOKEN="$ROOT_TOKEN" tessrax_vault vault operator raft snapshot save "/vault/file/snapshot_${TIMESTAMP}.snap"
# cp "./data/vault/file/snapshot_${TIMESTAMP}.snap" "$BACKUP_DIR/"

# 3. Ledger Archive
echo "  > Archiving Merkle Ledger..."
tar -czf "$BACKUP_DIR/ledger_${TIMESTAMP}.tar.gz" ./data/ledger

# 4. Retention Policy (Keep last 7 days)
echo "  > Cleaning old backups..."
find "$BACKUP_DIR" -name "*.sql" -type f -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +7 -delete

echo "[SUCCESS] Backup Complete: $BACKUP_DIR"
