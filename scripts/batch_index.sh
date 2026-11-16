#!/usr/bin/env bash
# batch_index.sh — Index photos in batches to manage memory/disk
#
# Usage:
#   ./scripts/batch_index.sh [OPTIONS] <staging_path>
#
# Options:
#   --batch-size N       Files per batch (default: 500)
#   --skip-vlm           Skip VLM enrichment (default: true)
#   --exts "ext1,ext2"   Comma-separated extensions (default: HEIC,JPG,PNG,MOV,MP4)
#   --dry-run            Show batches without indexing
#   --resume             Skip already-indexed files (checks DB)
#
# Examples:
#   ./scripts/batch_index.sh ~/photo-indexer/staging/icloud
#   ./scripts/batch_index.sh --batch-size 200 --exts "HEIC,JPG" ~/photo-indexer/staging/icloud/j.patrickwhite_at_me.com/2026/10
#   ./scripts/batch_index.sh --resume --batch-size 300 ~/photo-indexer/staging/icloud

set -euo pipefail

BATCH_SIZE=500
SKIP_VLM=true
EXTS="HEIC,JPG,JPEG,PNG,MOV,MP4,mp4,mov,heic,jpg,jpeg,png"
DRY_RUN=false
RESUME=false
DB_PATH=""

usage() {
  cat <<'EOF'
Usage: batch_index.sh [OPTIONS] <staging_path>

Options:
  --batch-size N       Files per batch (default: 500)
  --skip-vlm           Skip VLM enrichment (default: true)
  --exts "ext1,ext2"   Comma-separated extensions (default: HEIC,JPG,PNG,MOV,MP4)
  --dry-run            Show batches without indexing
  --resume             Skip already-indexed files (checks DB)
  --db PATH            SQLite DB path (default: PHOTO_INDEX_DB or ./photo_index.db)
  --help               Show this help
EOF
}

# Parse args
STAGING_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --skip-vlm) SKIP_VLM=true; shift ;;
    --no-skip-vlm) SKIP_VLM=false; shift ;;
    --exts) EXTS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --resume) RESUME=true; shift ;;
    --db) DB_PATH="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    -*) echo "Unknown option: $1"; usage; exit 1 ;;
    *) STAGING_PATH="$1"; shift ;;
  esac
done

if [[ -z "$STAGING_PATH" ]]; then
  echo "ERROR: <staging_path> required"
  usage
  exit 1
fi

STAGING_PATH="$(realpath "$STAGING_PATH")"
if [[ ! -d "$STAGING_PATH" ]]; then
  echo "ERROR: $STAGING_PATH not found"
  exit 1
fi

# Build find command for extensions
IFS=',' read -ra EXT_ARRAY <<< "$EXTS"
FIND_ARGS=()
for ext in "${EXT_ARRAY[@]}"; do
  FIND_ARGS+=(-o -iname "*.${ext}")
done
FIND_ARGS=("${FIND_ARGS[@]:1}")  # remove first -o

# Find all image/video files
echo "Scanning $STAGING_PATH for files..."
mapfile -t ALL_FILES < <(find "$STAGING_PATH" -type f \( "${FIND_ARGS[@]}" \) | sort)

TOTAL=${#ALL_FILES[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "No files found with extensions: $EXTS"
  exit 0
fi

echo "Found $TOTAL files"

# Resume: filter out already-indexed files
if [[ "$RESUME" == "true" ]]; then
  echo "Checking DB for already-indexed files..."
  DB_CMD="python3 -m photo_indexer.cli list --format=json"
  [[ -n "$DB_PATH" ]] && DB_CMD+=" --db $DB_PATH"
  INDEXED_PATHS=$($DB_CMD | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d in data:
    print(d['path'])
" | sort -u)

  # Filter out indexed files
  FILTERED=()
  for f in "${ALL_FILES[@]}"; do
    if ! grep -qx "$f" <<< "$INDEXED_PATHS"; then
      FILTERED+=("$f")
    fi
  done
  ALL_FILES=("${FILTERED[@]}")
  TOTAL=${#ALL_FILES[@]}
  echo "After resume filter: $TOTAL files remaining"
fi

if [[ $TOTAL -eq 0 ]]; then
  echo "All files already indexed"
  exit 0
fi

# Build CLI command
CLI_CMD="python3 -m photo_indexer.cli index"
[[ -n "$DB_PATH" ]] && CLI_CMD+=" --db $DB_PATH"
[[ "$SKIP_VLM" == "true" ]] && CLI_CMD+=" --skip-vlm"

# Process in batches
BATCH_NUM=0
for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
  BATCH_NUM=$((BATCH_NUM + 1))
  END=$((i + BATCH_SIZE))
  [[ $END -gt $TOTAL ]] && END=$TOTAL
  BATCH_FILES=("${ALL_FILES[@]:i:END-i}")

  echo ""
  echo "=== Batch $BATCH_NUM: ${#BATCH_FILES[@]} files (total $END/$TOTAL) ==="

  if [[ "$DRY_RUN" == "true" ]]; then
    for f in "${BATCH_FILES[@]}"; do
      echo "  $f"
    done
  else
    # Run index command with batch files
    $CLI_CMD "${BATCH_FILES[@]}"
    if [[ $? -ne 0 ]]; then
      echo "ERROR: Batch $BATCH_NUM failed, stopping"
      exit 1
    fi
  fi
done

echo ""
echo "=== Done: $TOTAL files indexed in $BATCH_NUM batches ==="