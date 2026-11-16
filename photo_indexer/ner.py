"""NER/NLP entity extraction from VLM text output using built-in taxonomies."""

import re
from typing import List, Optional

from .models import EntityType, NEREntity, Taxonomy
from .taxonomies import get_taxonomy, list_taxonomies


# Mapping from taxonomy name to EntityType
_TAXONOMY_TO_ENTITY_TYPE = {
    "faces": EntityType.PERSON,
    "animals": EntityType.ANIMAL,
    "pets": EntityType.PET,
    "places": EntityType.PLACE,
    "locations": EntityType.LOCATION,
    "objects": EntityType.OBJECT,
}


def _extract_mentions(text: str) -> List[str]:
    """Split VLM text into candidate noun-phrase mentions.

    Simple heuristic: split on sentence boundaries and punctuation,
    then filter short tokens. This keeps the dependency surface small
    (no spaCy required) while remaining effective for short VLM captions.
    """
    if not text:
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences/clauses
    clauses = re.split(r"[.!?;:\n]|,\s*", text)
    mentions: List[str] = []
    for clause in clauses:
        clause = clause.strip(" \"'()[]-")
        if not clause:
            continue
        # Further split long clauses on conjunctions
        for part in re.split(r"\s+(and|or|with|in|at|on|near)\s+", clause, flags=re.IGNORECASE):
            part = part.strip(" \"'()[]-")
            if len(part) >= 2:
                mentions.append(part)
    return mentions


def extract_entities(
    text: str,
    taxonomy_names: Optional[List[str]] = None,
    source: str = "vlm",
) -> List[NEREntity]:
    """Extract named entities from VLM output text against taxonomies.

    Args:
        text: VLM caption/description text to scan.
        taxonomy_names: Which taxonomies to match against. Defaults to all.
        source: Source tag for the entities (default "vlm").

    Returns:
        List of NEREntity objects with matched labels and types.
    """
    if taxonomy_names is None:
        taxonomy_names = list_taxonomies()

    mentions = _extract_mentions(text)
    entities: List[NEREntity] = []
    seen: set[tuple[str, str]] = set()

    for mention in mentions:
        for tax_name in taxonomy_names:
            taxonomy = get_taxonomy(tax_name)
            if taxonomy is None:
                continue
            entry = taxonomy.get_entry(mention)
            if entry is None:
                # Try substring/synonym match
                entry = _fuzzy_match(mention, taxonomy)
            if entry is None:
                continue

            key = (entry.label, tax_name)
            if key in seen:
                continue
            seen.add(key)

            entity_type = _TAXONOMY_TO_ENTITY_TYPE.get(tax_name, EntityType.OBJECT)
            entities.append(
                NEREntity(
                    text=mention,
                    entity_type=entity_type,
                    label=entry.label,
                    confidence=entry.confidence_threshold,
                    source=source,
                )
            )

    return entities


def _fuzzy_match(text: str, taxonomy: Taxonomy) -> Optional[Taxonomy]:
    """Fuzzy-match text against taxonomy entries by substring/synonym."""
    text_lower = text.lower()
    for entry in taxonomy.entries:
        if entry.label.lower() in text_lower or text_lower in entry.label.lower():
            return entry
        for syn in entry.synonyms:
            if syn.lower() in text_lower:
                return entry
    return None


def merge_entities(existing: List[NEREntity], new: List[NEREntity]) -> List[NEREntity]:
    """Merge two entity lists, deduplicating by (text, entity_type, label)."""
    seen = {(e.text.lower(), e.entity_type, e.label) for e in existing}
    merged = list(existing)
    for e in new:
        key = (e.text.lower(), e.entity_type, e.label)
        if key not in seen:
            merged.append(e)
            seen.add(key)
    return merged
