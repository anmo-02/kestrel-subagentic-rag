"""
Phase 2: Embed each document's clause chunks into its own persistent Chroma
collection, using text-embedding-3-small (OpenAIEmbeddings) for all three.

Ingestion is idempotent: chunk IDs are built from document_id + section_number
(see ingest/clean.py's Chunk.chunk_id), so re-running this script replaces
existing vectors in place via Chroma's upsert() instead of duplicating them.
Run it once; the collections persist to disk under chroma_db/.
"""

from __future__ import annotations

import os
from typing import List, Optional

import chromadb

from ingest.clean import Chunk, clean_all
from ingest.doc_config import COLLECTION_NAMES

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_client(persist_directory: str = PERSIST_DIR):
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def upsert_chunks(client, collection_name: str, chunks: List[Chunk], embeddings) -> int:
    """Embed and upsert one department's chunks. Returns the count upserted."""
    if not chunks:
        return 0
    collection = client.get_or_create_collection(name=collection_name)
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata() for c in chunks]
    vectors = embeddings.embed_documents(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    return len(chunks)


def build_all_collections(persist_directory: str = PERSIST_DIR, embeddings: Optional[object] = None) -> dict:
    """
    Build (or refresh) all three collections.

    `embeddings` defaults to real OpenAI text-embedding-3-small. Tests pass a
    deterministic FakeEmbeddings instead, so ingestion/upsert/query plumbing
    can be verified without an API key -- see tests/fake_embeddings.py.
    """
    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client = get_client(persist_directory)
    all_chunks = clean_all()
    counts = {}
    for dept, chunks in all_chunks.items():
        collection_name = COLLECTION_NAMES[dept]
        counts[collection_name] = upsert_chunks(client, collection_name, chunks, embeddings)
    return counts


if __name__ == "__main__":
    counts = build_all_collections()
    print("Ingested chunk counts per collection (chroma_db/):")
    for name, n in counts.items():
        print(f"  {name}: {n}")
