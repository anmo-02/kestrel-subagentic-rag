"""
Offline regression tests for Phase 1 (clause splitting) and Phase 2 (vector
collections). These run with no OpenAI key and no network access -- they use
the FakeEmbeddings stand-in to verify the plumbing (upsert, idempotency,
collection separation, metadata) independent of embedding quality.
"""

from ingest.clean import clean_all
from ingest.build_collections import build_all_collections, get_client
from ingest.doc_config import COLLECTION_NAMES
from tests.fake_embeddings import FakeEmbeddings

EXPECTED_SECTIONS = {
    "hr": [
        "0.0", "1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3", "2.4", "2.5",
        "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5",
        "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "6.4", "6.5",
        "7.1", "7.2", "7.3", "7.4",
    ],
    "it": [
        "0.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3",
        "3.4", "4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "5.3", "5.4", "6.1",
        "6.2", "6.3", "7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9.1", "9.2",
        "9.3", "9.4",
    ],
    "finance": [
        "0.0", "1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "2.4", "3.1", "3.2",
        "3.3", "3.4", "4.1", "4.2", "4.3", "5.1", "5.2", "5.3", "5.4", "6.1",
        "6.2", "6.3", "6.4", "6.5", "6.6", "7.1", "7.2", "8.1", "8.2", "8.3",
        "9.1", "9.2",
    ],
}


def test_clause_counts_and_no_gaps_or_duplicates():
    all_chunks = clean_all()
    for dept, expected in EXPECTED_SECTIONS.items():
        got = [c.section_number for c in all_chunks[dept]]
        assert got == expected, f"{dept}: expected {expected}, got {got}"


def test_no_header_footer_leakage():
    all_chunks = clean_all()
    for dept, chunks in all_chunks.items():
        for c in chunks:
            assert "Internal Page" not in c.text, f"leak in {dept} {c.chunk_id}"


def test_every_chunk_has_full_metadata():
    all_chunks = clean_all()
    for dept, chunks in all_chunks.items():
        for c in chunks:
            md = c.metadata()
            for field in ("department", "document_id", "document_title", "version", "effective_date", "section_number", "section_heading", "page"):
                assert md[field] not in (None, ""), f"{dept} {c.chunk_id} missing {field}"


def test_finance_6_4_matches_checkpoint_spec():
    """The Phase 1 checkpoint in the assignment names this exact clause."""
    all_chunks = clean_all()
    fin = {c.section_number: c for c in all_chunks["finance"]}
    c = fin["6.4"]
    assert c.version == "3.0"
    assert c.effective_date == "2026-07-01"
    assert c.page == 2
    assert "10 nights" in c.text


def test_ingestion_is_idempotent(tmp_path):
    persist_dir = str(tmp_path / "chroma_test")
    build_all_collections(persist_directory=persist_dir, embeddings=FakeEmbeddings())
    client = get_client(persist_dir)
    sizes_first = {name: client.get_collection(name).count() for name in COLLECTION_NAMES.values()}

    build_all_collections(persist_directory=persist_dir, embeddings=FakeEmbeddings())
    sizes_second = {name: client.get_collection(name).count() for name in COLLECTION_NAMES.values()}

    assert sizes_first == sizes_second
    assert all(n > 0 for n in sizes_first.values())


def test_collections_are_isolated_by_department(tmp_path):
    persist_dir = str(tmp_path / "chroma_test2")
    build_all_collections(persist_directory=persist_dir, embeddings=FakeEmbeddings())
    client = get_client(persist_dir)
    for dept, coll_name in COLLECTION_NAMES.items():
        coll = client.get_collection(coll_name)
        data = coll.get()
        for md in data["metadatas"]:
            assert md["department"] == dept


def test_query_plumbing_returns_correctly_shaped_results(tmp_path):
    persist_dir = str(tmp_path / "chroma_test3")
    build_all_collections(persist_directory=persist_dir, embeddings=FakeEmbeddings())
    client = get_client(persist_dir)
    coll = client.get_collection(COLLECTION_NAMES["hr"])
    fake = FakeEmbeddings()

    res = coll.query(query_embeddings=[fake.embed_query("earned leave")], n_results=4)
    assert len(res["ids"][0]) == 4
    assert all(m["department"] == "hr" for m in res["metadatas"][0])
    assert all("section_number" in m for m in res["metadatas"][0])
