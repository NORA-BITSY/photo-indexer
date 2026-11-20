# Photo Indexer (OSS/self-hosted)

Local-first stack to export photos from iOS/Android/cloud sources, normalize them into an Immich-backed library, extract full metadata, and run local VLM enrichment (captions, objects/scenes, OCR, faces/embeddings where supported) with non-destructive outputs.

## Defaults
- **Hub:** Immich (server + ML + PostgreSQL + Redis)
- **Storage:** Linux home server/NAS, Docker Compose, separate paths for uploads/library/staging/postgres/metadata
- **AI:** Local-only via Ollama (`VLM_MODEL=llava` by default), CPU fallback OK (GPU optional)
- **Sources (priority):** iPhone/iCloud (`icloudpd`) + Google Photos **Takeout** + Android Samsung/Pixel (byte-perfect/original transfer) + local folders
- **Enrichment outputs:** DB tables in `enrich.*` and optional XMP sidecars/JSON dumps; originals remain intact

## Created artifacts
- `docker-compose.yml` — Immich + Redis + Postgres + Immich ML + Ollama + `immich-analyze` worker + optional `exiftool-dump` job profile
- `.env.example` — configuration template (copy to `.env`)
- `sql/enrich_schema.sql` — companion schema for raw metadata, normalized core fields, VLM results, faces, and sidecar tracking
- `scripts/exiftool_dump.sh` — full ExifTool metadata extraction to JSON/JSONL
- `scripts/write_xmp_sidecar.sh` — safe, non-destructive XMP sidecar writer for selected enrichment fields
- `scripts/backup.sh` — backup database, staging, config with 7-day retention
- `scripts/batch_index.sh` — scheduled indexing runner for systemd timer
- `scripts/photo-indexer-index.service` / `.timer` — systemd units for automated indexing
- `scripts/icloud_auth_and_sync.sh` / `icloud_auth_vm.sh` / `sync_icloud_to_gcp.sh` — iCloud sync helpers

## Quick start
1. **Prepare storage paths** (adjust in `.env`):
   ```bash
   sudo mkdir -p /srv/photo-indexer/{immich/upload,library,staging/{icloud,takeout,android,local},postgres,metadata}
   ```
2. **Configure env:**
   ```bash
   cp .env.example .env
   # edit .env (set DB_PASSWORD, paths, TZ, etc.)
   ```
3. **Start core services:**
   ```bash
   docker compose up -d database redis immich-server immich-machine-learning ollama
   ```
4. **Pull local VLM model (local-only):**
   ```bash
   docker compose exec ollama ollama pull ${VLM_MODEL:-llava}
   ```
5. **Open Immich:** `http://<server>:2283` and create an API key for enrichment workers (`IMMICH_API_KEY` in `.env`).
6. **Start enrichment worker:**
   ```bash
   docker compose up -d immich-analyze
   ```

> Note: `sql/enrich_schema.sql` is mounted into Postgres init dir and applies on first DB initialization. For an existing DB, run it manually against the Immich database.

## Source ingestion (recommended)
- **iCloud:** `icloud-photos-downloader/icloud_photos_downloader` (`icloudpd`) into `STAGING_LOCATION/icloud/`. Handle 2FA/session expiry/ADP constraints.
- **Google Photos:** Prefer **Google Takeout** (metadata/JSON sidecars included) unpacked into `STAGING_LOCATION/takeout/`. Avoid relying solely on Google Photos API for originals/location fidelity.
- **Android (Samsung/Pixel):** Prefer byte-perfect/original transfer (USB/MTP/DCIM, vendor tools, or direct copies) into `STAGING_LOCATION/android/` to avoid cloud recompression/metadata loss.
- **Local folders:** place/link under `STAGING_LOCATION/local/` or mount into `LIBRARY_LOCATION`.

Then add as Immich External Library and/or upload/import per Immich docs. Keep originals read-only where possible.

## Metadata extraction & enrichment
- **Immich:** parses common EXIF/media metadata into its DB.
- **Full dump (all tags):** run ExifTool job container/profile and scripts, store JSON in `enrich.asset_metadata_raw`:
  ```bash
  docker compose --profile jobs run --rm exiftool-dump \
    /scripts/exiftool_dump.sh /data/library /data/metadata/library.exiftool.jsonl
  ```
- **VLM enrichment:** `immich-analyze` + Ollama generate detailed captions/objects/scenes/OCR/faces signals (model/support dependent); persist to `enrich.asset_vlm` / `enrich.asset_faces` via your worker/integration.
- **Sidecars (non-destructive):**
  ```bash
  ./scripts/write_xmp_sidecar.sh /path/to/photo.jpg --description "..." --keywords "person,beach,sunset"
  ```
  Log outcomes in `enrich.asset_sidecar` (`written|skipped_unsafe|error`).

## Face recognition (CLI)

Manage known persons and match detected faces:

```bash
# Create a known person with face embedding (JSON array of 512 floats)
photo-indexer person create --name "John Doe" --embedding-file embedding.json

# List all known persons
photo-indexer person list

# Get person details
photo-indexer person get --name "John Doe"

# Match faces in a specific asset to known persons
photo-indexer match-faces --asset-id <asset-uuid> --threshold 0.6

# Match faces in ALL assets and auto-link
photo-indexer match-faces --all --threshold 0.6

# Show all faces matched to a person
photo-indexer person-faces --name "John Doe"
```

Face embeddings are 512-dimensional vectors from InsightFace (`buffalo_l` model). Matching uses cosine similarity with configurable threshold (default 0.6).

## Tagging system (CLI)

Add, remove, and query tags on assets:

```bash
# Add tags to an asset
photo-indexer tag add --asset-id <uuid> --tag "sunset" --tag "beach"

# Remove a tag
photo-indexer tag remove --asset-id <uuid> --tag "beach"

# List tags on an asset
photo-indexer tag list --asset-id <uuid>

# Find assets by tag
photo-indexer tags find --tag "sunset" --limit 50

# Bulk import tags from TSV file (asset_id<TAB>tag1,tag2,tag3)
photo-indexer tag-import --file tags.tsv
```

Tags are stored in `enrich_tags` table with source tracking (`manual`, `file`, etc.).

## Prometheus metrics & monitoring

Enable metrics server for scraping:

```bash
# Start CLI with metrics on port 9090
photo-indexer --metrics-port 9090 index /path/to/photos
```

Exposed metrics (prefix `photo_indexer_`):
- **Counters:** `indexed_total`, `vlm_analyzed_total`, `sidecar_written_total`, `faces_detected_total`, `errors_total`
- **Gauges:** `assets_indexed`, `db_size_bytes`
- **Histograms:** `index_duration_seconds`, `vlm_analyze_duration_seconds`

Monitoring stack includes:
- **Grafana dashboard** (`monitoring/grafana/photo-indexer.json`) — assets, rate, latency, errors, DB size
- **Alertmanager rules** (`monitoring/alerts/rules.yml`) — failure rate, DB growth, no activity
- **Logrotate config** (`monitoring/logrotate/photo-indexer`) — daily rotation, 7-day retention

Install Prometheus client: `pip install photo-indexer[monitoring]`

## Backup & automation

```bash
# Manual backup (7-day retention by default)
./scripts/backup.sh [/custom/backup/dir]

# Scheduled indexing via systemd (install service + timer)
sudo cp scripts/photo-indexer-index.service /etc/systemd/system/
sudo cp scripts/photo-indexer-index.timer /etc/systemd/system/
sudo systemctl enable --now photo-indexer-index.timer
```

## iCloud sync helpers

```bash
# Authenticate icloudpd (interactive, handles 2FA)
./scripts/icloud_auth_and_sync.sh

# Or use VM-based auth for headless servers
./scripts/icloud_auth_vm.sh

# Sync from iCloud to GCP bucket
./scripts/sync_icloud_to_gcp.sh
```

## Caveats
- Verify/align `docker-compose.yml` with the **current Immich release** (DB image/vector extensions/env vars/ML config evolve). Pin image digests in production.
- `immich-analyze` image/tag/API compatibility can change; verify source and supported env vars.
- CPU-only VLM inference is slow on large libraries; tune model (`moondream` smaller/faster) and batch sizes.
- "All metadata" is best-effort across sources; combine Immich + ExifTool full dumps + source Takeout/iCloud sidecars.
- Keep originals immutable; write only sidecars/derived artifacts; maintain **3-2-1 backups** for originals and Postgres.