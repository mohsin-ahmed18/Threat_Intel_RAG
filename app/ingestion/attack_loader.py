"""
Loader for MITRE ATT&CK data.

ATT&CK is distributed as a STIX 2.1 bundle: a single JSON file containing many
typed "objects" (attack-pattern, course-of-action, intrusion-set, relationship, ...)
all mixed together in one flat list. We pull out attack-pattern objects (= techniques)
and resolve their relationships so each technique record is self-contained and
ready to chunk — no relationship graph left implicit.
"""
import json
import requests
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import ATTACK_STIX_URL, RAW_ATTACK_DIR


@dataclass
class AttackTechnique:
    technique_id: str          # e.g. "T1055"
    name: str                  # e.g. "Process Injection"
    description: str
    tactics: list[str] = field(default_factory=list)   # e.g. ["defense-evasion", "privilege-escalation"]
    platforms: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    is_subtechnique: bool = False
    parent_technique_id: Optional[str] = None

    def to_document_text(self) -> str:
        """Flatten into the text block that actually gets embedded."""
        lines = [
            f"Technique {self.technique_id}: {self.name}",
            f"Tactics: {', '.join(self.tactics) or 'unspecified'}",
            f"Platforms: {', '.join(self.platforms) or 'unspecified'}",
            "",
            self.description,
        ]
        if self.mitigations:
            lines += ["", "Mitigations:", *[f"- {m}" for m in self.mitigations]]
        return "\n".join(lines)


def fetch_attack_bundle(save: bool = True) -> dict:
    """Download the current enterprise-attack STIX bundle from MITRE's official repo."""
    resp = requests.get(ATTACK_STIX_URL, timeout=60)
    resp.raise_for_status()
    bundle = resp.json()
    if save:
        RAW_ATTACK_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_ATTACK_DIR / "enterprise-attack.json"
        out_path.write_text(json.dumps(bundle))
    return bundle


def _get_external_id(obj: dict) -> Optional[str]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def parse_attack_bundle(bundle: dict) -> list[AttackTechnique]:
    """
    Parse a raw STIX bundle into flat, self-contained AttackTechnique records.
    Resolves 'relationship' objects to attach mitigations to their techniques,
    since in raw STIX that link is a separate object, not a nested field.
    """
    objects = bundle["objects"]

    techniques_by_stix_id = {}
    mitigations_by_stix_id = {}
    relationships = []

    for obj in objects:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        obj_type = obj.get("type")
        if obj_type == "attack-pattern":
            techniques_by_stix_id[obj["id"]] = obj
        elif obj_type == "course-of-action":
            mitigations_by_stix_id[obj["id"]] = obj
        elif obj_type == "relationship":
            relationships.append(obj)

    # Map technique -> list of mitigation names via "mitigates" relationships
    mitigations_for_technique: dict[str, list[str]] = {}
    for rel in relationships:
        if rel.get("relationship_type") == "mitigates":
            target = rel.get("target_ref")
            source = rel.get("source_ref")
            if target in techniques_by_stix_id and source in mitigations_by_stix_id:
                mitigations_for_technique.setdefault(target, []).append(
                    mitigations_by_stix_id[source].get("name", "")
                )

    results = []
    for stix_id, obj in techniques_by_stix_id.items():
        technique_id = _get_external_id(obj)
        if not technique_id:
            continue

        tactics = [
            phase["phase_name"]
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]

        results.append(
            AttackTechnique(
                technique_id=technique_id,
                name=obj.get("name", ""),
                description=obj.get("description", ""),
                tactics=tactics,
                platforms=obj.get("x_mitre_platforms", []),
                mitigations=mitigations_for_technique.get(stix_id, []),
                is_subtechnique=obj.get("x_mitre_is_subtechnique", False),
                parent_technique_id=(
                    technique_id.split(".")[0] if "." in technique_id else None
                ),
            )
        )
    return results


if __name__ == "__main__":
    bundle = fetch_attack_bundle()
    techniques = parse_attack_bundle(bundle)
    print(f"Parsed {len(techniques)} techniques")
    print("\n--- Example record ---\n")
    print(techniques[0].to_document_text())
