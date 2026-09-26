"""
Phase 1: Read, clean and split the three policy PDFs into clause-level chunks.

Design notes (see README "Design decisions" for the full rationale):

  * We rely on the document's own running header ("Kestrel Systems Pvt. Ltd.
    <Title> v<version>" followed by "<DOC-ID> | Effective <date> | Internal
    Page <N>") as BOTH the thing we strip AND the ground-truth source of the
    page number for everything that follows it. This is more reliable than
    trusting pypdf's own page boundaries, because it survives however the
    PDF's text stream happens to be laid out.

  * Splitting is done on the clause pattern "N.M " at the start of a line,
    never on a fixed character count. A clause and anything under it (a
    table, a bullet list) stays in the same chunk because nothing about a
    table row matches "N.M " at line-start, so it's simply absorbed into
    the buffer for the clause above it.

  * Only genuinely long clauses (rare in these documents) get a further
    character-based split, and even then every sub-chunk carries the full
    original metadata plus a part suffix.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingest.doc_config import DOCUMENTS, DocumentConfig

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Matches the two-line running header wherever it occurs (it is sometimes
# glued directly onto the previous page's last word with no whitespace,
# which is why we don't anchor on line-start here).
HEADER_RE = re.compile(
    r"Kestrel Systems Pvt\.\s*Ltd\.\s+[^\n]*?v\d+\.\d+\s*\n"
    r"KSPL-[A-Z]+-POL-\d+\s*\|\s*Effective\s+[^|]*?\|\s*Internal Page\s*(\d+)\s*\n?",
)

CLAUSE_RE = re.compile(r"^(\d+)\.(\d+)\s+(.*)$")     # e.g. "6.6 Recovery: if the employee..."
SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")          # e.g. "6. Resignation and Exit"

MAX_CHUNK_CHARS = 1800   # generous; only unusually long clauses get sub-split
CHUNK_OVERLAP = 150

_sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass
class Chunk:
    chunk_id: str
    department: str
    document_id: str
    document_title: str
    version: str
    effective_date: str
    section_number: str
    section_heading: str
    page: int
    text: str

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("chunk_id")
        return d


def extract_text_from_pdf(path: str) -> str:
    """Real-world entry point: extract text from an actual PDF file with pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _tag_lines_with_pages(raw_text: str) -> List[Tuple[int, str]]:
    """
    Strip the running header, and return a flat list of (page_number, line)
    tuples for every non-blank line in the document body, using the page
    number embedded in each header occurrence as ground truth.
    """
    pieces: List[str] = []
    last_end = 0
    page_for_next_span = 1  # page 1's header is always the first thing in the stream

    for match in HEADER_RE.finditer(raw_text):
        before = raw_text[last_end:match.start()]
        pieces.append(f"<<<PAGE:{page_for_next_span}>>>\n{before}")
        # The header states the page number of the content that FOLLOWS it.
        page_for_next_span = int(match.group(1))
        last_end = match.end()

    pieces.append(f"<<<PAGE:{page_for_next_span}>>>\n{raw_text[last_end:]}")

    reconstructed = "\n".join(pieces)

    tagged: List[Tuple[int, str]] = []
    current_page = 1
    marker_re = re.compile(r"^<<<PAGE:(\d+)>>>$")
    for raw_line in reconstructed.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        m = marker_re.match(line)
        if m:
            current_page = int(m.group(1))
            continue
        tagged.append((current_page, line))
    return tagged


def _split_into_clauses(tagged_lines: List[Tuple[int, str]], cfg: DocumentConfig) -> List[Chunk]:
    chunks: List[Chunk] = []

    current_section_heading = "Document Header"
    current_clause_num = "0.0"
    current_clause_page = tagged_lines[0][0] if tagged_lines else 1
    buffer: List[str] = []

    def flush():
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{cfg.document_id}-{current_clause_num}",
                department=cfg.department,
                document_id=cfg.document_id,
                document_title=cfg.title,
                version=cfg.version,
                effective_date=cfg.effective_date,
                section_number=current_clause_num,
                section_heading=current_section_heading,
                page=current_clause_page,
                text=text,
            )
        )

    for page, line in tagged_lines:
        clause_m = CLAUSE_RE.match(line)
        section_m = SECTION_RE.match(line) if not clause_m else None

        if clause_m:
            flush()
            buffer = []
            current_clause_num = f"{clause_m.group(1)}.{clause_m.group(2)}"
            current_clause_page = page
            buffer.append(line)
        elif section_m:
            # A bare section header ("6. Resignation and Exit") updates the
            # heading used for clauses that follow, but is not itself a
            # clause boundary -- fold it into the front-matter/preceding
            # buffer rather than starting a new chunk.
            current_section_heading = section_m.group(2).strip()
        else:
            buffer.append(line)

    flush()
    return chunks


def _sub_split_long_chunks(chunks: List[Chunk]) -> List[Chunk]:
    out: List[Chunk] = []
    for c in chunks:
        if len(c.text) <= MAX_CHUNK_CHARS:
            out.append(c)
            continue
        parts = _sub_splitter.split_text(c.text)
        for i, part in enumerate(parts, start=1):
            out.append(
                Chunk(
                    chunk_id=f"{c.chunk_id}-part{i}",
                    department=c.department,
                    document_id=c.document_id,
                    document_title=c.document_title,
                    version=c.version,
                    effective_date=c.effective_date,
                    section_number=c.section_number,
                    section_heading=c.section_heading,
                    page=c.page,
                    text=part,
                )
            )
    return out


def clean_and_split(raw_text: str, department: str) -> List[Chunk]:
    cfg = DOCUMENTS[department]
    tagged = _tag_lines_with_pages(raw_text)
    chunks = _split_into_clauses(tagged, cfg)
    chunks = _sub_split_long_chunks(chunks)
    return chunks


def load_raw_text(department: str) -> str:
    """
    Load raw extracted text for a department. Prefers a real PDF in data/
    (production path); falls back to the offline fixture in data/raw/ used
    for testing the parser without needing the actual PDF bytes on disk.
    """
    cfg = DOCUMENTS[department]
    pdf_path = os.path.join(PDF_DIR, cfg.source_pdf)
    if os.path.exists(pdf_path):
        return extract_text_from_pdf(pdf_path)

    fixture_path = os.path.join(RAW_DIR, f"{department}_raw.txt")
    if os.path.exists(fixture_path):
        with open(fixture_path, "r", encoding="utf-8") as f:
            return f.read()

    raise FileNotFoundError(
        f"Neither {pdf_path} nor {fixture_path} exists. Put the real PDF in "
        f"data/, named {cfg.source_pdf}."
    )


def clean_all() -> dict:
    return {dept: clean_and_split(load_raw_text(dept), dept) for dept in DOCUMENTS}


if __name__ == "__main__":
    all_chunks = clean_all()

    print("=" * 70)
    print("PHASE 1 CHECKPOINT REPORT")
    print("=" * 70)

    for dept, chunks in all_chunks.items():
        print(f"\n[{dept}] {DOCUMENTS[dept].title} v{DOCUMENTS[dept].version}")
        print(f"  chunk count: {len(chunks)}")

        # header/footer leakage check: "Internal Page" and the doc-id-pipe
        # pattern only ever occur inside the running header we strip, never
        # in genuine clause prose, so they're an unambiguous leak signal.
        # (A bare "Kestrel Systems Pvt. Ltd." is NOT used here, since clause
        # 1.1 legitimately mentions the company name in its own sentence.)
        leaked = [c for c in chunks if "Internal Page" in c.text or HEADER_RE.search(c.text)]
        print(f"  chunks with leaked header/footer text: {len(leaked)}")
        if leaked:
            for c in leaked[:3]:
                print(f"    LEAK in {c.chunk_id}: {c.text[:80]!r}")

        print(f"  first 3 section numbers: {[c.section_number for c in chunks[:3]]}")
        print(f"  last 3 section numbers:  {[c.section_number for c in chunks[-3:]]}")

    print("\n--- Finance 6.4 spot check (temporary accommodation, the key conflict clause) ---")
    fin = {c.section_number: c for c in all_chunks["finance"]}
    c64 = fin.get("6.4")
    if c64:
        print(f"  section_number   : {c64.section_number}")
        print(f"  section_heading  : {c64.section_heading}")
        print(f"  version          : {c64.version}")
        print(f"  effective_date   : {c64.effective_date}")
        print(f"  page             : {c64.page}")
        print(f"  text             : {c64.text}")
    else:
        print("  MISSING -- clause 6.4 not found!")

    print("\n--- HR 4.4 spot check (the clause 6.4 conflicts with) ---")
    hr = {c.section_number: c for c in all_chunks["hr"]}
    c44 = hr.get("4.4")
    if c44:
        print(f"  text: {c44.text}")
