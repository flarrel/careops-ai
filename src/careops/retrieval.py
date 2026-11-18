"""Rank policy passages and expose the shared evidence-packet contract."""

from dataclasses import dataclass

from careops.contracts import validate_evidence_packet
from careops.ingest import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def ranked(chunks, scores, top_k, *, positive_only=False):
    eligible = [
        (chunk, float(score))
        for chunk, score in zip(chunks, scores, strict=True)
        if not positive_only or score > 0
    ]
    return [
        Hit(chunk, score)
        for chunk, score in sorted(eligible, key=lambda pair: (-pair[1], pair[0].chunk_id))[:top_k]
    ]


class LexicalIndex:
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    @property
    def payload_bytes(self):
        matrix = self.matrix
        return sum(array.nbytes for array in (
            matrix.data, matrix.indices, matrix.indptr, self.vectorizer.idf_,
        ))

    def search(self, query, top_k):
        vector = self.vectorizer.transform([query])
        scores = (self.matrix @ vector.T).toarray().ravel()
        return ranked(self.chunks, scores, top_k, positive_only=True)


class DenseIndex:
    def __init__(self, chunks, encoder):
        self.chunks = chunks
        self.encoder = encoder
        self.matrix = encoder.passages([chunk.text for chunk in chunks])

    @property
    def payload_bytes(self):
        return self.matrix.nbytes

    def search(self, query, top_k):
        return ranked(self.chunks, self.matrix @ self.encoder.query(query), top_k)


class RerankedIndex:
    """Fuse lexical and semantic candidate ranks, then keep one passage per page."""

    def __init__(self, lexical, dense):
        self.lexical = lexical
        self.dense = dense

    @property
    def payload_bytes(self):
        return self.lexical.payload_bytes + self.dense.payload_bytes

    def search(self, query, top_k):
        pool_size = max(20, 4 * top_k)
        pages = {}
        for index in (self.lexical, self.dense):
            seen = set()
            for rank, hit in enumerate(index.search(query, pool_size), start=1):
                key = (hit.chunk.source_id, hit.chunk.page)
                if key in seen:
                    continue
                seen.add(key)
                representative, score, best_rank = pages.get(key, (hit, 0.0, rank))
                if rank < best_rank:
                    representative, best_rank = hit, rank
                pages[key] = (representative, score + 1 / (60 + rank), best_rank)
        return ranked(
            [hit.chunk for hit, _, _ in pages.values()],
            [score for _, score, _ in pages.values()],
            top_k,
        )


def retrieve(index, query, top_k=5):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    packet = {
        "query": query,
        "items": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "source_id": hit.chunk.source_id,
                "text": hit.chunk.text,
                "score": hit.score,
                "location": hit.chunk.location,
            }
            for hit in index.search(query, top_k)
        ],
    }
    validate_evidence_packet(packet)
    return packet
