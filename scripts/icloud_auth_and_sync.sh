#!/usr/bin/env bash
# icloud_auth_and_sync.sh — Authenticate iCloud on macOS, reuse session on GCP VM
#
# Workflow:
#   1. Run ON MAC: ./icloud_auth_and_sync.sh auth     # interactive 2FA, saves session
#   2. Run ON MAC: ./icloud_auth_and_sync.sh push-session  # copy ~/.icloudpd to GCP
#   3. Run ON GCP: ./icloud_auth_and_sync.sh download   # download directly to GCP
#   4. Run ON GCP: ./icloud_auth_and_sync.sh index      # index downloaded files
#
# Prerequisites:
#   pip install icloudpd   # on BOTH macOS and GCP VM
#   ssh gcp "mkdir -p ~/photo-indexer/staging/icloud"

set -euo pipefail

GCP_HOST="${GCP_HOST:-gcp}"
GCP_REMOTE_PATH="${GCP_REMOTE_PATH:-~/photo-indexer}"
ICLOUD_DIR="${ICLOUD_DIR:-$HOME/photo-indexer/staging/icloud}"
ICLOUDPD_SESSION_DIR="${ICLOUDPD_SESSION_DIR:-$HOME/.icloudpd}"

usage() {
  cat <<'EOF'
Usage: icloud_auth_and_sync.sh <command>

Commands:
  auth           (macOS) Interactive iCloud auth with 2FA, saves session to ~/.icloudpd/
  push-session   (macOS) Copy ~/.icloudpd/ to GCP VM
  download       (GCP)  Download iCloud photos to GCP using saved session
  index          (GCP)  Index downloaded files in photo-indexer
  full           (macOS) auth + push-session, then ssh to GCP for download + index

Env:
  GCP_HOST=gcp                    # SSH host from ~/.ssh/config
  GCP_REMOTE_PATH=~/photo-indexer # Remote path on GCP
  ICLOUD_DIR=~/photo-indexer/staging/icloud
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi

run_local() {
  echo "[local] $*"
  eval "$*"
}

run_remote() {
  echo "[remote] $*"
  ssh "$GCP_HOST" "eval \"$*\""
}

case "$cmd" in
  auth)
    echo "=== Authenticating iCloud on macOS (interactive 2FA) ==="
    mkdir -p "$ICLOUD_DIR"
    run_local "icloudpd --directory \"$ICLOUD_DIR\" --recent 10"
    echo "Session saved to $ICLOUDPD_SESSION_DIR"
    ;;

  push-session)
    echo "=== Pushing iCloud session to GCP ==="
    run_local "rsync -avz --delete \"$ICLOUDPD_SESSION_DIR/\" \"$GCP_HOST:$ICLOUDPD_SESSION_DIR/\""
    echo "Session pushed to $GCP_HOST:$ICLOUDPD_SESSION_DIR"
    ;;

  download)
    echo "=== Downloading iCloud photos on GCP ==="
    run_remote "mkdir -p \"$ICLOUD_DIR\""
    run_remote "icloudpd --directory \"$ICLOUD_DIR\" --recent 1000"
    ;;

  index)
    echo "=== Indexing on GCP ==="
    run_remote "cd \"$GCP_REMOTE_PATH\" && python3 -m photo_indexer.cli index \"$GCP_REMOTE_PATH/staging/icloud\" --skip-vlm"
    ;;

  full)
    echo "=== Full workflow: auth -> push -> download -> index ==="
    "$0" auth
    "$0" push-session
    ssh "$GCP_HOST" "bash -s" <<'REMOTE_EOF'
      set -euo pipefail
      GCP_REMOTE_PATH="${GCP_REMOTE_PATH:-~/photo-indexer}"
      ICLOUD_DIR="${ICLOUD_DIR:-~/photo-indexer/staging/icloud}"
      echo "Downloading on GCP..."
      icloudpd --directory "$ICLOUD_DIR" --recent 1000
      echo "Indexing..."
      cd "$GCP_REMOTE_PATH" && python3 -m photo_indexer.cli index "$GCP_REMOTE_PATH/staging/icloud" --skip-vlm
REMOTE_EOF
    ;;

  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac