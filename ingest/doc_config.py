"""
Static configuration for the three Kestrel policy documents.

We hardcode document-level metadata (id, title, version, effective date)
rather than trying to regex it out of free text on every ingest run. These
facts are set once when a new policy version is issued, not something that
changes per-chunk, and hardcoding them means a single stray sentence in the
PDF can never silently corrupt every chunk's version/date metadata.

If a new policy version ships, bump the values here and re-run ingestion
(the stable per-clause IDs in build_collections.py mean the old chunks are
overwritten in place rather than duplicated).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentConfig:
    department: str          # "hr" | "it" | "finance" -- also the Chroma collection key
    document_id: str         # e.g. "KSPL-HR-POL-004"
    title: str                # e.g. "HR Policy Handbook"
    version: str              # e.g. "4.1"
    effective_date: str       # ISO format, e.g. "2026-04-01"
    source_pdf: str           # filename expected in data/


DOCUMENTS = {
    "hr": DocumentConfig(
        department="hr",
        document_id="KSPL-HR-POL-004",
        title="HR Policy Handbook",
        version="4.1",
        effective_date="2026-04-01",
        source_pdf="Kestrel_HR_Policy_Handbook_v4_1.pdf",
    ),
    "it": DocumentConfig(
        department="it",
        document_id="KSPL-IT-POL-002",
        title="IT Services & Information Security Policy",
        version="2.3",
        effective_date="2026-01-01",
        source_pdf="Kestrel_IT_Services_and_Security_Policy_v2_3.pdf",
    ),
    "finance": DocumentConfig(
        department="finance",
        document_id="KSPL-FIN-POL-003",
        title="Travel, Expense & Relocation Policy",
        version="3.0",
        effective_date="2026-07-01",
        source_pdf="Kestrel_Travel_Expense_and_Relocation_Policy_v3_0.pdf",
    ),
}

COLLECTION_NAMES = {
    "hr": "hr_policy",
    "it": "it_policy",
    "finance": "finance_policy",
}
