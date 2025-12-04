#!/bin/bash
set -e

RESTORE_DIR="$1"

if [ -z "$RESTORE_DIR" ]; then
  echo "Usage: restore.sh <backup_dir>"
  exit 1
fi

echo "[+] Validating manifest..."
sha256sum -c "$RESTORE_DIR/manifest.sha256"

echo "[+] Restoring Postgres..."
psql -U tessrax_user tessrax_db < "$RESTORE_DIR/db.sql"

echo "[+] Restoring ledger..."
tar -xzf "$RESTORE_DIR/ledger.tar.gz" -C /

echo "[✓] Restore complete and validated."
