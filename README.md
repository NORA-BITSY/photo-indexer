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

## Caveats
- Verify/align `docker-compose.yml` with the **current Immich release** (DB image/vector extensions/env vars/ML config evolve). Pin image digests in production.
- `immich-analyze` image/tag/API compatibility can change; verify source and supported env vars.
- CPU-only VLM inference is slow on large libraries; tune model (`moondream` smaller/faster) and batch sizes.
- "All metadata" is best-effort across sources; combine Immich + ExifTool full dumps + source Takeout/iCloud sidecars.
- Keep originals immutable; write only sidecars/derived artifacts; maintain **3-2-1 backups** for originals and Postgres.
