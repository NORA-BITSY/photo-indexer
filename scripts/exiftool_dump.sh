#!/bin/sh
# Extract full metadata from image/video files into JSON using ExifTool.
# Usage:
#   ./scripts/exiftool_dump.sh [SOURCE_DIR] [OUTPUT_JSONL_OR_DIR]
# Examples:
#   ./scripts/exiftool_dump.sh /data/library /data/metadata/library.exiftool.jsonl
#   ./scripts/exiftool_dump.sh /data/staging/icloud /data/metadata/icloud
#
# Notes:
# - Preserves originals (read-only). Produces complete tag dump (-a -G1 -s).
# - JSONL mode (output ends with .jsonl/.ndjson): one JSON object per file line.
# - Directory mode: one .json file per media file under OUTPUT_DIR.
# - Intended to feed enrich.asset_metadata_raw.raw_json.

set -eu

SOURCE_DIR="${1:-/data/library}"
OUTPUT_PATH="${2:-/data/metadata/exiftool.jsonl}"

if ! command -v exiftool >/dev/null 2>&1; then
  echo "Error: exiftool not found. Run inside exiftool-dump container or install exiftool." >&2
  exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")" 2>/dev/null || true

case "$OUTPUT_PATH" in
  *.jsonl|*.ndjson)
    echo "Writing ExifTool JSONL dump: $OUTPUT_PATH"
    # -r recursive, -a duplicate tags, -G1 group names, -s short tag names
    # -json emits an array per invocation; exiftool emits one object per file in array.
    # Convert array objects to JSONL via awk-compatible streaming is non-trivial in pure sh,
    # so we emit per-file JSON objects by invoking exiftool per file (safe for correctness).
    : > "$OUTPUT_PATH"
    find "$SOURCE_DIR" -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.heif' \
         -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' -o -iname '*.dng' -o -iname '*.cr2' \
         -o -iname '*.nef' -o -iname '*.arw' -o -iname '*.raf' -o -iname '*.rw2' -o -iname '*.orf' \
         -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.mkv' \
         -o -iname '*.3gp' -o -iname '*.gif' \) \
      -print | while IFS= read -r f; do
        exiftool -json -a -G1 -s -- "$f" 2>/dev/null | sed -e '1s/^\[//' -e '$s/\]$//' | tr -d '\n' >> "$OUTPUT_PATH"
        printf '\n' >> "$OUTPUT_PATH"
      done
    echo "Done. Lines: $(wc -l < "$OUTPUT_PATH" | tr -d ' ')"
    ;;
  *)
    mkdir -p "$OUTPUT_PATH"
    echo "Writing per-file ExifTool JSON to directory: $OUTPUT_PATH"
    find "$SOURCE_DIR" -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.heif' \
         -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' -o -iname '*.dng' -o -iname '*.cr2' \
         -o -iname '*.nef' -o -iname '*.arw' -o -iname '*.raf' -o -iname '*.rw2' -o -iname '*.orf' \
         -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.mkv' \
         -o -iname '*.3gp' -o -iname '*.gif' \) \
      -print | while IFS= read -r f; do
        rel="${f#$SOURCE_DIR/}"
        safe_name=$(printf '%s' "$rel" | tr '/ ' '__')
        out="$OUTPUT_PATH/${safe_name}.json"
        exiftool -json -a -G1 -s -- "$f" > "$out" 2>/dev/null || printf '[{"SourceFile":"%s","Error":"exiftool_failed"}]\n' "$f" > "$out"
      done
    echo "Done."
    ;;
esac
