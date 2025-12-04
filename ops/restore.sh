#!/bin/bash
# USAGE: ./ops/restore.sh <timestamp>
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./restore.sh <YYYYMMDD_HHMMSS>"
    exit 1
fi

TIMESTAMP=$1
BACKUP_DIR="./data/backups"

echo "[RESTORE] Restoring System State to $TIMESTAMP..."
echo "WARNING: This will overwrite current data."
read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi

# 1. Stop Services
docker-compose stop api worker

# 2. Restore DB
echo "  > Restoring Database..."
cat "$BACKUP_DIR/db_${TIMESTAMP}.sql" | docker exec -i tessrax_db psql -U tessrax postgres

# 3. Restore Ledger
echo "  > Restoring Ledger..."
rm -rf ./data/ledger/*
tar -xzf "$BACKUP_DIR/ledger_${TIMESTAMP}.tar.gz" -C ./

# 4. Restart
docker-compose start

echo "[SUCCESS] System Restored."
