"""
Deterministic, fully offline stand-in for OpenAIEmbeddings.

Used ONLY by the test suite, so ingestion/upsert/retrieval plumbing can be
verified without an OpenAI API key or network access. The vectors carry no
real semantic meaning (they're a hash of the text) -- never point the actual
app at this class.
"""

import hashlib
from typing import List

DIM = 32


def _vector_for(text: str) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b / 127.5) - 1.0 for b in digest[:DIM]]


class FakeEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_vector_for(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return _vector_for(text)
