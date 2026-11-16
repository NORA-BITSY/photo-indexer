#!/usr/bin/env bash
# icloud_auth_vm.sh — Run THIS on the GCP VM in a terminal for interactive iCloud auth
#
# Usage on GCP VM:
#   cd ~/photo-indexer
#   ./scripts/icloud_auth_vm.sh                           # interactive: prompts for each account
#   ./scripts/icloud_auth_vm.sh --both                    # auth both known accounts interactively
#   ./scripts/icloud_auth_vm.sh --all                     # download ALL photos (not just recent)
#   ./scripts/icloud_auth_vm.sh --both --all              # both accounts, all photos
#   ICLOUD_USERS="j.patrickwhite@me.com,paraliel@icloud.com" ./scripts/icloud_auth_vm.sh --both
#   ICLOUD_USER=j.patrickwhite@me.com ICLOUD_PASS=xxx ICLOUD_USER2=paraliel@icloud.com ICLOUD_PASS2=yyy ./scripts/icloud_auth_vm.sh --both
#
# Each account gets its own subdirectory: staging/icloud/<email_sanitized>/

set -euo pipefail

ICLOUD_BASE_DIR="${ICLOUD_BASE_DIR:-$HOME/photo-indexer/staging/icloud}"
RECENT="${RECENT:-1000}"
SIZE="${SIZE:-original}"
ALL_PHOTOS="${ALL_PHOTOS:-false}"
BOTH=false

# Known accounts (can be overridden by env)
ACCOUNT1="${ACCOUNT1:-j.patrickwhite@me.com}"
ACCOUNT2="${ACCOUNT2:-paraliel@icloud.com}"

# Parse args
for arg in "$@"; do
  case "$arg" in
    --both) BOTH=true ;;
    --all) ALL_PHOTOS=true ;;
  esac
done

mkdir -p "$ICLOUD_BASE_DIR"

export PATH="/home/g8emiller245/.local/bin:$PATH"

# Build icloudpd args
ICLOUDPD_ARGS=(
  --size "$SIZE"
  --log-level info
)
if [[ "$ALL_PHOTOS" == "true" ]]; then
  # Don't use --recent, download all
  echo "=== Downloading ALL photos (this will take a while) ==="
else
  ICLOUDPD_ARGS+=(--recent "$RECENT")
  echo "=== Downloading recent $RECENT photos ==="
fi

auth_account() {
  local user="$1"
  local password="$2"
  local dir="$ICLOUD_BASE_DIR/${user//@/_at_}"
  mkdir -p "$dir"

  if [[ -z "$password" ]]; then
    read -rsp "iCloud password for $user: " password
    echo
  fi

  echo "Authenticating $user..."
  icloudpd \
    --directory "$dir" \
    --username "$user" \
    --password "$password" \
    "${ICLOUDPD_ARGS[@]}"

  echo "  ✓ $user -> $dir"
}

# Single account mode (interactive or arg)
if [[ "$BOTH" != "true" ]]; then
  if [[ $# -eq 1 ]]; then
    auth_account "$1" ""
  else
    read -rp "Apple ID (email): " user
    auth_account "$user" ""
  fi
  exit 0
fi

# --both mode: authenticate both accounts
echo "=== Authenticating both accounts ==="

# Account 1
pass1="${ICLOUD_PASS:-}"
if [[ -z "$pass1" && -n "${ICLOUD_USER_PASS:-}" ]]; then
  pass1="${ICLOUD_USER_PASS#*:}"
fi
auth_account "$ACCOUNT1" "$pass1"

# Account 2
pass2="${ICLOUD_PASS2:-}"
if [[ -z "$pass2" && -n "${ICLOUD_USER2_PASS:-}" ]]; then
  pass2="${ICLOUD_USER2_PASS#*:}"
fi
auth_account "$ACCOUNT2" "$pass2"

echo ""
echo "=== Both accounts authenticated ==="
echo "Account 1: $ACCOUNT1 -> $ICLOUD_BASE_DIR/${ACCOUNT1//@/_at_}"
echo "Account 2: $ACCOUNT2 -> $ICLOUD_BASE_DIR/${ACCOUNT2//@/_at_}"
echo ""
echo "Session saved to ~/.pyicloud (shared cookie jar)"
echo ""
echo "For future non-interactive runs:"
echo "  export PATH=/home/g8emiller245/.local/bin:\$PATH"
echo "  icloudpd --directory \"$ICLOUD_BASE_DIR/${ACCOUNT1//@/_at_}\" ${ALL_PHOTOS:+\"\${ICLOUDPD_ARGS[@]}\"}"
echo "  icloudpd --directory \"$ICLOUD_BASE_DIR/${ACCOUNT2//@/_at_}\" ${ALL_PHOTOS:+\"\${ICLOUDPD_ARGS[@]}\"}"