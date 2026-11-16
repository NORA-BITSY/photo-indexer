#!/usr/bin/env bash
# sync_icloud_to_gcp.sh — Download iCloud Photos on macOS, rsync to GCP VM, trigger index
#
# Prerequisites (run once on Mac):
#   pip install osxphotos icloudpd
#   ssh gcp "mkdir -p ~/photo-indexer/staging/icloud"
#
# Usage:
#   ./sync_icloud_to_gcp.sh                 # download recent 1000, sync, index
#   ./sync_icloud_to_gcp.sh --all           # download all, sync, index
#   ./sync_icloud_to_gcp.sh --dry-run       # show what would happen
#   ./sync_icloud_to_gcp.sh --no-index      # skip remote index step
#
# Config via env:
#   GCP_HOST=gcp              # SSH host from ~/.ssh/config
#   GCP_REMOTE_PATH=~/photo-indexer
#   ICLOUD_DOWNLOAD_DIR=~/photo-indexer/staging/icloud
#   RSYNC_OPTS="-avz --progress"
#   INDEX_OPTS="--skip-vlm"   # or empty to run VLM

set -euo pipefail

GCP_HOST="${GCP_HOST:-gcp}"
GCP_REMOTE_PATH="${GCP_REMOTE_PATH:-~/photo-indexer}"
ICLOUD_DOWNLOAD_DIR="${ICLOUD_DOWNLOAD_DIR:-$HOME/photo-indexer/staging/icloud}"
RSYNC_OPTS="${RSYNC_OPTS:--avz --progress --delete}"
INDEX_OPTS="${INDEX_OPTS:---skip-vlm}"
DOWNLOAD_ALL="${DOWNLOAD_ALL:-false}"
DRY_RUN="${DRY_RUN:-false}"
NO_INDEX="${NO_INDEX:-false}"

usage() {
  cat <<'EOF'
Usage: sync_icloud_to_gcp.sh [options]
  --all           Download all iCloud photos (not just recent)
  --dry-run       Show commands without executing
  --no-index      Skip remote indexing step
  --recent N      Download N most recent (default: 1000)
  --help          Show this help
EOF
}

RECENT=1000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DOWNLOAD_ALL=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --no-index) NO_INDEX=true; shift ;;
    --recent) RECENT="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$*"
  fi
}

# 1. Ensure local download dir exists
run "mkdir -p \"$ICLOUD_DOWNLOAD_DIR\""

# 2. Download from iCloud
if command -v osxphotos >/dev/null 2>&1; then
  echo "Using osxphotos..."
  if [[ "$DOWNLOAD_ALL" == "true" ]]; then
    run "osxphotos export --download-missing \"$ICLOUD_DOWNLOAD_DIR\""
  else
    run "osxphotos export --download-missing --last-import-days 30 \"$ICLOUD_DOWNLOAD_DIR\""
  fi
elif command -v icloudpd >/dev/null 2>&1; then
  echo "Using icloudpd..."
  if [[ "$DOWNLOAD_ALL" == "true" ]]; then
    run "icloudpd --directory \"$ICLOUD_DOWNLOAD_DIR\""
  else
    run "icloudpd --directory \"$ICLOUD_DOWNLOAD_DIR\" --recent $RECENT"
  fi
else
  echo "ERROR: Neither osxphotos nor icloudpd found. Install one:"
  echo "  pip install osxphotos icloudpd"
  exit 1
fi

# 3. Rsync to GCP
echo "Syncing to $GCP_HOST:$GCP_REMOTE_PATH/staging/icloud/ ..."
run "rsync $RSYNC_OPTS \"$ICLOUD_DOWNLOAD_DIR/\" \"$GCP_HOST:$GCP_REMOTE_PATH/staging/icloud/\""

# 4. Trigger remote index (optional)
if [[ "$NO_INDEX" != "true" ]]; then
  echo "Triggering remote index on $GCP_HOST..."
  run "ssh \"$GCP_HOST\" \"cd $GCP_REMOTE_PATH && python3 -m photo_indexer.cli index \\\"$GCP_REMOTE_PATH/staging/icloud\\\" $INDEX_OPTS\""
else
  echo "Skipping remote index (--no-index)"
fi

echo "Done."