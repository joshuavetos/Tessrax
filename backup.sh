#!/bin/bash
set -e

DATE=$(date +"%Y-%m-%d_%H-%M")
BACKUP_DIR="/backups/$DATE"
mkdir -p "$BACKUP_DIR"

echo "[+] Backing up Postgres..."
pg_dump -U tessrax_user tessrax_db > "$BACKUP_DIR/db.sql"

echo "[+] Archiving ledger files..."
tar -czf "$BACKUP_DIR/ledger.tar.gz" /app/ledger

echo "[+] Writing manifest..."
sha256sum "$BACKUP_DIR/db.sql" "$BACKUP_DIR/ledger.tar.gz" > "$BACKUP_DIR/manifest.sha256"

echo "[+] Enforcing retention..."
find /backups -type d -mtime +14 -exec rm -rf {} \;

echo "[✓] Backup complete: $BACKUP_DIR"
