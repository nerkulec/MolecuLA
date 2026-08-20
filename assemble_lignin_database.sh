#!/usr/bin/env bash
# Reassemble the LFS-chunked SQLite database and verify it byte-for-byte.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO_DIR/data"
DATABASE="$DATA_DIR/lignin_solubility.db"
CHECKSUM_FILE="$DATA_DIR/lignin_solubility.db.sha256"
PARTS=("$DATABASE".part-*)

[[ -f "${PARTS[0]}" ]] || { echo "Missing database chunks: $DATABASE.part-*" >&2; exit 1; }
[[ -f "$CHECKSUM_FILE" ]] || { echo "Missing checksum: $CHECKSUM_FILE" >&2; exit 1; }

if [[ -f "$DATABASE" ]] && (cd "$DATA_DIR" && sha256sum --check --status "$(basename "$CHECKSUM_FILE")"); then
  echo "Database already assembled and verified: $DATABASE"
  exit 0
fi

TEMP_DATABASE="$DATABASE.assembling.$$"
trap 'rm -f "$TEMP_DATABASE"' EXIT
cat "${PARTS[@]}" > "$TEMP_DATABASE"
EXPECTED="$(cut -d ' ' -f 1 "$CHECKSUM_FILE")"
ACTUAL="$(sha256sum "$TEMP_DATABASE" | cut -d ' ' -f 1)"
[[ "$ACTUAL" == "$EXPECTED" ]] || {
  echo "Database checksum mismatch: expected $EXPECTED, got $ACTUAL" >&2
  exit 1
}
mv "$TEMP_DATABASE" "$DATABASE"
trap - EXIT
echo "Assembled and verified: $DATABASE"
