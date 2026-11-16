#!/bin/sh
# Safely write/update XMP sidecars for selected enrichment fields (non-destructive).
# Originals are never modified. Sidecars are written as <media-file>.xmp by default.
#
# Usage:
#   ./scripts/write_xmp_sidecar.sh <media-file> [--title "..."] [--description "..."] [--keywords "a,b,c"] [--person "Name"] [--force]
#
# Safety behavior:
# - Refuses RAW/DNG/video sidecar writes unless --force (too risky/format-specific for blind writes).
# - Creates a .bak copy of existing sidecar before updating (unless first-time create).
# - Records nothing to DB directly; caller can log outcome to enrich.asset_sidecar.

set -eu

usage() {
  cat >&2 <<'EOF'
Usage: write_xmp_sidecar.sh <media-file> [options]
  --title <text>        Set XMP-dc:Title
  --description <text>  Set XMP-dc:Description / ImageDescription
  --keywords <csv>      Add keywords/subjects (comma-separated; repeatable)
  --person <name>       Add person-in-image keyword/name tag (repeatable)
  --sidecar <path>      Explicit sidecar path (default: <media-file>.xmp)
  --force               Allow writing sidecars for RAW/video targets (use with care)
  --dry-run             Show planned exiftool command without writing
EOF
  exit 2
}

if ! command -v exiftool >/dev/null 2>&1; then
  echo "Error: exiftool not found." >&2
  exit 1
fi

[ $# -ge 1 ] || usage
MEDIA="$1"; shift
[ -f "$MEDIA" ] || { echo "Error: media file not found: $MEDIA" >&2; exit 1; }

TITLE=""
DESC=""
SIDECAR="${MEDIA}.xmp"
FORCE="0"
DRY_RUN="0"
KW_ARGS=""
PERSON_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --title) TITLE="${2:-}"; shift 2 ;;
    --description) DESC="${2:-}"; shift 2 ;;
    --keywords)
      # split comma-separated values into repeated -XMP-dc:Subject+= tags
      old_ifs="$IFS"; IFS=','; set -- $2; IFS="$old_ifs"
      for kw in "$@"; do
        # trim leading/trailing spaces (basic)
        kw_trim=$(printf '%s' "$kw" | sed 's/^ *//;s/ *$//')
        [ -n "$kw_trim" ] && KW_ARGS="$KW_ARGS -XMP-dc:Subject+=$kw_trim"
      done
      shift 2
      ;;
    --person)
      p="${2:-}"
      [ -n "$p" ] && PERSON_ARGS="$PERSON_ARGS -XMP-iptcExt:PersonInImage+=$p -XMP-mwg-rs:RegionPersonDisplayName+=$p"
      shift 2
      ;;
    --sidecar) SIDECAR="${2:-}"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

ext=$(printf '%s' "$MEDIA" | awk -F. '{print tolower($NF)}')
case "$ext" in
  dng|cr2|nef|arw|raf|rw2|orf|mp4|mov|m4v|avi|mkv|3gp)
    if [ "$FORCE" != "1" ]; then
      echo "Refusing to write sidecar for '$ext' without --force (status would be skipped_unsafe)." >&2
      exit 3
    fi
    ;;
esac

if [ -z "$TITLE" ] && [ -z "$DESC" ] && [ -z "$KW_ARGS" ] && [ -z "$PERSON_ARGS" ]; then
  echo "No fields provided to write. Nothing to do." >&2
  exit 0
fi

# Build exiftool args; always target sidecar only and preserve originals.
set -- exiftool -overwrite_original -o "$SIDECAR"

[ -n "$TITLE" ] && set -- "$@" "-XMP-dc:Title=$TITLE"
if [ -n "$DESC" ]; then
  set -- "$@" "-XMP-dc:Description=$DESC" "-IFD0:ImageDescription=$DESC"
fi

# Append keyword/person args (word-splitting intentional for simple CSV/name tokens)
if [ -n "$KW_ARGS" ]; then
  # shellcheck disable=SC2086
  set -- "$@" $KW_ARGS
fi
if [ -n "$PERSON_ARGS" ]; then
  # shellcheck disable=SC2086
  set -- "$@" $PERSON_ARGS
fi

set -- "$@" "$MEDIA"

if [ -f "$SIDECAR" ] && [ "$DRY_RUN" != "1" ]; then
  cp -f "$SIDECAR" "${SIDECAR}.bak"
fi

if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY RUN:'; printf ' %q' "$@" 2>/dev/null || printf ' %s' "$@"; printf '\n'
  exit 0
fi

echo "Writing sidecar: $SIDECAR"
"$@"
echo "Sidecar write completed (status candidate: written)."