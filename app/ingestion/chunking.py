"""
Structure-aware chunking for CVE and ATT&CK records.

Design, per the earlier architecture discussion:
- The natural unit IS the record (one CVE, one technique) — no blind
  character-count slicing of well-structured data.
- If a record's text exceeds CHUNK_SIZE_TOKENS, it gets split further with a
  recursive splitter (paragraph -> sentence -> hard cut) so nothing reaches
  the embedding model oversized/truncated.
- Every resulting chunk carries the FULL metadata of its parent record, so
  Qdrant can pre-filter on CVE ID, CVSS score, ATT&CK tactic, etc. regardless
  of whether the record was split into 1 chunk or 5.

Token counting: BGE-M3 uses an XLM-RoBERTa tokenizer. We approximate here
with a chars-per-token heuristic (~4 chars/token for English technical text)
rather than pulling the real tokenizer, since downloading the BGE-M3 tokenizer
isn't possible from this sandbox's network allowlist. Swap in the real
tokenizer once you're running locally — see `_count_tokens` docstring.
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from app.ingestion.attack_loader import AttackTechnique
from app.ingestion.cve_loader import CVERecord
from transformers import AutoTokenizer

_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_type: str          # "cve" | "attack_technique"
    parent_id: str            # CVE ID or technique ID, shared across split siblings
    metadata: dict[str, Any] = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    
    return len(_tok.encode(text))


def _split_oversized_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Recursive splitter: try paragraph breaks first, then sentence breaks,
    then a hard character cut as a last resort. Keeps overlap so context
    isn't lost at a chunk boundary.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # No paragraph breaks to work with — fall back to sentences
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current = ""
    for unit in paragraphs:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if _count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Carry overlap from the tail of the previous chunk into the next
            overlap_chars = overlap_tokens * 4
            current = (current[-overlap_chars:] + "\n\n" + unit).strip() if current else unit
            # If a single unit alone still exceeds max_tokens, hard-cut it
            while _count_tokens(current) > max_tokens:
                cut_chars = max_tokens * 4
                chunks.append(current[:cut_chars])
                current = current[cut_chars - overlap_tokens * 4:]
    if current:
        chunks.append(current)
    return chunks


def chunk_cve_record(record: CVERecord) -> list[Chunk]:
    base_metadata = {
        "cve_id": record.cve_id,
        "cvss_score": record.cvss_score,
        "cvss_severity": record.cvss_severity,
        "published_date": record.published_date,
        "affected_products": record.affected_products[:20],
    }
    text_pieces = _split_oversized_text(
        record.to_document_text(), CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
    )
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text=piece,
            source_type="cve",
            parent_id=record.cve_id,
            metadata={**base_metadata, "part": i + 1, "total_parts": len(text_pieces)},
        )
        for i, piece in enumerate(text_pieces)
    ]


def chunk_attack_technique(technique: AttackTechnique) -> list[Chunk]:
    base_metadata = {
        "technique_id": technique.technique_id,
        "technique_name": technique.name,
        "tactics": technique.tactics,
        "platforms": technique.platforms,
        "is_subtechnique": technique.is_subtechnique,
        "parent_technique_id": technique.parent_technique_id,
    }
    text_pieces = _split_oversized_text(
        technique.to_document_text(), CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
    )
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text=piece,
            source_type="attack_technique",
            parent_id=technique.technique_id,
            metadata={**base_metadata, "part": i + 1, "total_parts": len(text_pieces)},
        )
        for i, piece in enumerate(text_pieces)
    ]


if __name__ == "__main__":
    from app.ingestion.attack_loader import fetch_attack_bundle, parse_attack_bundle

    bundle = fetch_attack_bundle(save=False)
    techniques = parse_attack_bundle(bundle)

    # Find the longest technique description to prove the splitter actually engages
    longest = max(techniques, key=lambda t: len(t.description))
    chunks = chunk_attack_technique(longest)

    print(f"Technique {longest.technique_id} ({longest.name}): "
          f"{_count_tokens(longest.to_document_text())} approx tokens -> {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"  part {c.metadata['part']}/{c.metadata['total_parts']}: "
              f"{_count_tokens(c.text)} approx tokens, metadata keys={list(c.metadata.keys())}")

    # Sanity check on a short technique too — should stay as exactly 1 chunk
    shortest = min(techniques, key=lambda t: len(t.description))
    short_chunks = chunk_attack_technique(shortest)
    print(f"\nTechnique {shortest.technique_id} ({shortest.name}): "
          f"{_count_tokens(shortest.to_document_text())} approx tokens -> {len(short_chunks)} chunk(s)")
