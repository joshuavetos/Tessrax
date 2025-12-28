#!/bin/bash
set -e

SRC="/app/ledger"
DEST="/mnt/offline/ledger"

echo "[+] Creating offline snapshot..."
rsync -av --delete "$SRC/" "$DEST/"

echo "[+] Hashing snapshot..."
find "$DEST" -type f -exec sha256sum {} \; > "$DEST/SNAPSHOT.sha256"

echo "[✓] Air-gapped replication complete."
