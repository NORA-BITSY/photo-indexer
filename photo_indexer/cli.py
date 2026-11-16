"""CLI core: click-based CLI with index/analyze/enrich/export/sidecar commands."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import click

from .database import PhotoIndexDB
from .metadata import (
    find_images,
    parse_exif,
    run_exiftool,
    sha256_file,
    write_xmp_sidecar,
)
from .models import Asset, AssetStatus, deterministic_asset_id
from .ner import extract_entities, merge_entities
from .taxonomies import ALL_TAXONOMIES, list_taxonomies
from .vlm import VLMClient


# ---------------------------------------------------------------------------
# Helper: default paths from env
# ---------------------------------------------------------------------------
def _default_db_path() -> Path:
    import os
    return Path(os.environ.get("PHOTO_INDEX_DB", "./photo_index.db"))


def _default_staging() -> Path:
    import os
    return Path(os.environ.get("STAGING_LOCATION", "./staging"))


def _default_library() -> Path:
    import os
    return Path(os.environ.get("LIBRARY_LOCATION", "./library"))


def _default_vlm_model() -> str:
    import os
    return os.environ.get("VLM_MODEL", "llava")


def _default_vlm_url() -> str:
    import os
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------
@click.group()
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to SQLite index database (default: PHOTO_INDEX_DB or ./photo_index.db)",
)
@click.pass_context
def main(ctx: click.Context, db_path: Optional[Path]) -> None:
    """Photo Indexer — self-hosted, local-first photo indexing CLI.

    Index, enrich, and export photos with local VLM + NER + taxonomies.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path or _default_db_path()
    ctx.obj["db"] = PhotoIndexDB(ctx.obj["db_path"])


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------
@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recursively scan directories (default: yes)",
)
@click.option(
    "--skip-vlm",
    is_flag=True,
    default=False,
    help="Skip VLM analysis during indexing",
)
@click.option(
    "--vlm-model",
    default=None,
    help="VLM model to use (default: VLM_MODEL env or llava)",
)
@click.option(
    "--vlm-url",
    default=None,
    help="Ollama base URL (default: OLLAMA_BASE_URL or http://localhost:11434)",
)
@click.option(
    "--write-sidecar/--no-write-sidecar",
    default=False,
    help="Write XMP sidecar during indexing (default: no)",
)
@click.pass_context
def index(
    ctx: click.Context,
    paths: List[Path],
    recursive: bool,
    skip_vlm: bool,
    vlm_model: Optional[str],
    vlm_url: Optional[str],
    write_sidecar: bool,
) -> None:
    """Index image files from PATH(s): extract Exif metadata, hash, optionally run VLM + NER."""

    db: PhotoIndexDB = ctx.obj["db"]
    model = vlm_model or _default_vlm_model()
    url = vlm_url or _default_vlm_url()

    vlm: Optional[VLMClient] = None
    if not skip_vlm:
        vlm = VLMClient(base_url=url, model=model)
        if not vlm.is_available():
            click.echo(
                f"WARNING: Ollama not reachable at {url}. VLM analysis will be skipped.",
                err=True,
            )
            vlm = None

    total = 0
    indexed = 0

    for base in paths:
        base = base.resolve()
        images = find_images(base, recursive=recursive)
        click.echo(f"Found {len(images)} image(s) under {base}")

        for img_path in images:
            total += 1
            rel = str(img_path)
            click.echo(f"  [{total}] {rel}", nl=False)

            # Hash + Exif
            digest = sha256_file(img_path)
            raw_exif = run_exiftool(img_path)
            if not raw_exif and not shutil.which("exiftool"):
                click.echo(" [exiftool not found, metadata skipped]", nl=False)
            exif = parse_exif(raw_exif) if raw_exif else None

            # Build asset
            asset = Asset(
                id=deterministic_asset_id(rel, digest),
                path=rel,
                filename=img_path.name,
                content_sha256=digest,
                file_size=img_path.stat().st_size if img_path.exists() else None,
                mime_type=exif.mime_type if exif else None,
                exif=exif,
                status=AssetStatus.INDEXED,
            )

            # VLM analysis
            if vlm is not None:
                click.echo(" ... vlm", nl=False)
                try:
                    result = vlm.analyze_image(img_path, model=model)
                    asset.vlm = result
                    if result.caption or result.detailed_description:
                        text = (result.caption or "") + " " + (result.detailed_description or "")
                        entities = extract_entities(text)
                        asset.ner_entities = merge_entities(asset.ner_entities, entities)
                except Exception as exc:  # noqa: BLE001
                    click.echo(f" [vlm error: {exc}]", nl=False)
                    asset.status = AssetStatus.ERROR
                    asset.error_message = str(exc)

            # Sidecar
            if write_sidecar:
                status = write_xmp_sidecar(
                    img_path,
                    description=asset.vlm.caption if asset.vlm else None,
                    keywords=[e.label for e in asset.ner_entities if e.label],
                )
                asset.sidecar_status = status

            asset.status = AssetStatus.ENRICHED if asset.status != AssetStatus.ERROR else AssetStatus.ERROR
            db.upsert(asset)
            indexed += 1
            click.echo(" done")

    click.echo(f"\nIndexed {indexed}/{total} assets into {ctx.obj['db_path']}")


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Analyze a specific asset by ID",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    default=False,
    help="Re-analyze all indexed assets",
)
@click.option(
    "--vlm-model",
    default=None,
    help="VLM model to use",
)
@click.option(
    "--vlm-url",
    default=None,
    help="Ollama base URL",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    asset_id: Optional[str],
    analyze_all: bool,
    vlm_model: Optional[str],
    vlm_url: Optional[str],
) -> None:
    """Run VLM analysis on indexed assets (re-run or targeted)."""

    db: PhotoIndexDB = ctx.obj["db"]
    model = vlm_model or _default_vlm_model()
    url = vlm_url or _default_vlm_url()

    vlm = VLMClient(base_url=url, model=model)
    if not vlm.is_available():
        click.echo(f"ERROR: Ollama not reachable at {url}", err=True)
        sys.exit(1)

    if analyze_all:
        assets = db.list_all(limit=10000)
        click.echo(f"Re-analyzing {len(assets)} assets...")
        for a in assets:
            img_path = Path(a.path)
            if not img_path.exists():
                click.echo(f"  SKIP (missing): {a.path}")
                continue
            result = vlm.analyze_image(img_path, model=model)
            a.vlm = result
            if result.caption or result.detailed_description:
                text = (result.caption or "") + " " + (result.detailed_description or "")
                entities = extract_entities(text)
                a.ner_entities = merge_entities(a.ner_entities, entities)
            a.status = AssetStatus.ANALYZED
            db.upsert(a)
            click.echo(f"  OK: {a.filename}")
    elif asset_id:
        a = db.get(asset_id)
        if not a:
            click.echo(f"ERROR: asset {asset_id} not found", err=True)
            sys.exit(1)
        img_path = Path(a.path)
        if not img_path.exists():
            click.echo(f"ERROR: image file missing: {a.path}", err=True)
            sys.exit(1)
        result = vlm.analyze_image(img_path, model=model)
        a.vlm = result
        if result.caption or result.detailed_description:
            text = (result.caption or "") + " " + (result.detailed_description or "")
            entities = extract_entities(text)
            a.ner_entities = merge_entities(a.ner_entities, entities)
        a.status = AssetStatus.ANALYZED
        db.upsert(a)
        click.echo(f"OK: {a.filename}")
    else:
        click.echo("Specify --asset-id or --all", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# enrich command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Enrich a specific asset by ID",
)
@click.option(
    "--all",
    "enrich_all",
    is_flag=True,
    default=False,
    help="Enrich all indexed assets",
)
@click.option(
    "--taxonomy",
    "taxonomies",
    multiple=True,
    type=click.Choice(list_taxonomies()),
    help="Taxonomy to use for NER (repeatable, default: all)",
)
@click.pass_context
def enrich(
    ctx: click.Context,
    asset_id: Optional[str],
    enrich_all: bool,
    taxonomies: tuple,
) -> None:
    """Run NER entity extraction against VLM text using taxonomies."""

    db: PhotoIndexDB = ctx.obj["db"]
    tax_list = list(taxonomies) if taxonomies else None

    if enrich_all:
        assets = db.list_all(limit=10000)
        click.echo(f"Enriching {len(assets)} assets...")
        for a in assets:
            text_parts = []
            if a.vlm:
                if a.vlm.caption:
                    text_parts.append(a.vlm.caption)
                if a.vlm.detailed_description:
                    text_parts.append(a.vlm.detailed_description)
            if not text_parts:
                continue
            text = " ".join(text_parts)
            entities = extract_entities(text, taxonomy_names=tax_list)
            a.ner_entities = merge_entities(a.ner_entities, entities)
            a.status = AssetStatus.ENRICHED
            db.upsert(a)
            click.echo(f"  OK: {a.filename} ({len(entities)} entities)")
    elif asset_id:
        a = db.get(asset_id)
        if not a:
            click.echo(f"ERROR: asset {asset_id} not found", err=True)
            sys.exit(1)
        text_parts = []
        if a.vlm:
            if a.vlm.caption:
                text_parts.append(a.vlm.caption)
            if a.vlm.detailed_description:
                text_parts.append(a.vlm.detailed_description)
        if not text_parts:
            click.echo("No VLM text to enrich from. Run 'analyze' first.", err=True)
            sys.exit(1)
        text = " ".join(text_parts)
        entities = extract_entities(text, taxonomy_names=tax_list)
        a.ner_entities = merge_entities(a.ner_entities, entities)
        a.status = AssetStatus.ENRICHED
        db.upsert(a)
        click.echo(f"OK: {a.filename} ({len(entities)} entities)")
    else:
        click.echo("Specify --asset-id or --all", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sidecar command
# ---------------------------------------------------------------------------
@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--title",
    default=None,
    help="Set XMP-dc:Title",
)
@click.option(
    "--description",
    default=None,
    help="Set XMP-dc:Description / ImageDescription",
)
@click.option(
    "--keywords",
    default=None,
    help="Add keywords/subjects (comma-separated)",
)
@click.option(
    "--person",
    "persons",
    multiple=True,
    help="Add person-in-image keyword (repeatable)",
)
@click.option(
    "--sidecar",
    "sidecar_path",
    default=None,
    help="Explicit sidecar path (default: <media-file>.xmp)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing sidecars and include RAW/video",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Do not write, just report what would be done",
)
@click.pass_context
def sidecar(
    ctx: click.Context,
    paths: List[Path],
    title: Optional[str],
    description: Optional[str],
    keywords: Optional[str],
    persons: tuple,
    sidecar_path: Optional[str],
    force: bool,
    dry_run: bool,
) -> None:
    """Write XMP sidecar files for PATH(s)."""

    from .metadata import write_xmp_sidecar

    db: PhotoIndexDB = ctx.obj["db"]
    total = 0
    written = 0
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    for base in paths:
        base = base.resolve()
        for img_path in find_images(base, recursive=True):
            total += 1
            a = None
            # Try to find existing asset to pull caption/keywords
            for existing in db.list_all(limit=10000):
                if existing.path == str(img_path):
                    a = existing
                    break
            # CLI options override DB values
            desc = description if description is not None else (a.vlm.caption if (a and a.vlm) else None)
            kw = kw_list if kw_list is not None else ([e.label for e in a.ner_entities if e.label] if a else None)
            # Build keyword list with persons
            if persons:
                if kw is None:
                    kw = []
                kw.extend(persons)
            # sidecar_path is passed directly to write_xmp_sidecar
            status = write_xmp_sidecar(
                img_path,
                description=desc,
                keywords=kw,
                force=force,
                dry_run=dry_run,
                sidecar_path=sidecar_path,
            )
            from .models import SidecarStatus
            if status == SidecarStatus.WRITTEN:
                written += 1
            click.echo(f"  {img_path}: {status.value}")
            if a:
                a.sidecar_status = status
                db.upsert(a)
    click.echo(f"\nSidecars: {written}/{total} written")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------
@main.command(name="list")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "--limit",
    default=100,
    help="Max results (default: 100)",
)
@click.option(
    "--offset",
    default=0,
    help="Offset for pagination (default: 0)",
)
@click.pass_context
def list_assets(ctx: click.Context, fmt: str, limit: int, offset: int) -> None:
    """List indexed assets."""

    db: PhotoIndexDB = ctx.obj["db"]
    assets = db.list_all(limit=limit, offset=offset)

    if fmt == "json":
        data = []
        for a in assets:
            data.append({
                "id": str(a.id),
                "path": a.path,
                "filename": a.filename,
                "status": a.status.value if hasattr(a.status, "value") else a.status,
                "sha256": a.content_sha256,
                "vlm_caption": a.vlm.caption if a.vlm else None,
                "entities": [e.label for e in a.ner_entities if e.label],
                "sidecar": a.sidecar_status.value if hasattr(a.sidecar_status, "value") else a.sidecar_status if a.sidecar_status else None,
            })
        click.echo(json.dumps(data, indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["id", "path", "filename", "status", "sha256", "caption", "entities", "sidecar"])
        for a in assets:
            status = a.status.value if hasattr(a.status, "value") else a.status
            sidecar = a.sidecar_status.value if hasattr(a.sidecar_status, "value") else a.sidecar_status if a.sidecar_status else ""
            writer.writerow([
                str(a.id), a.path, a.filename, status,
                a.content_sha256 or "",
                (a.vlm.caption or "") if a.vlm else "",
                ";".join(e.label for e in a.ner_entities if e.label),
                sidecar,
            ])
    else:
        click.echo(f"{'ID':<38} {'Filename':<40} {'Status':<12} {'Caption'}")
        click.echo("-" * 100)
        for a in assets:
            status = a.status.value if hasattr(a.status, "value") else a.status
            cap = (a.vlm.caption or "")[:60] if a.vlm else ""
            click.echo(f"{str(a.id):<38} {a.filename:<40} {status:<12} {cap}")


# ---------------------------------------------------------------------------
# taxonomies command
# ---------------------------------------------------------------------------
@main.command()
def taxonomies() -> None:
    """List available taxonomies and their entries."""

    for name in list_taxonomies():
        tax = ALL_TAXONOMIES[name]
        click.echo(f"\n{name} (v{tax.version}) — {tax.description}")
        for entry in tax.entries:
            syn = f" ({', '.join(entry.synonyms)})" if entry.synonyms else ""
            click.echo(f"  - {entry.label}{syn}")


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--check-ollama",
    is_flag=True,
    default=False,
    help="Check Ollama availability and list models",
)
@click.pass_context
def config(ctx: click.Context, check_ollama: bool) -> None:
    """Show configuration and environment."""


    click.echo("Photo Indexer Configuration")
    click.echo("=" * 40)
    click.echo(f"DB path:        {ctx.obj['db_path']}")
    click.echo(f"VLM model:      {_default_vlm_model()}")
    click.echo(f"VLM URL:        {_default_vlm_url()}")
    click.echo(f"Staging:        {_default_staging()}")
    click.echo(f"Library:        {_default_library()}")
    click.echo(f"DB records:     {ctx.obj['db'].count()}")
    click.echo(f"Taxonomies:     {', '.join(list_taxonomies())}")

    if check_ollama:
        vlm = VLMClient(base_url=_default_vlm_url(), model=_default_vlm_model())
        if vlm.is_available():
            models = vlm.list_models()
            click.echo(f"\nOllama: reachable at {_default_vlm_url()}")
            click.echo(f"Models: {', '.join(models) if models else '(none)'}")
        else:
            click.echo(f"\nOllama: NOT reachable at {_default_vlm_url()}", err=True)


if __name__ == "__main__":
    main()
