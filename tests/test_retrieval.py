import unittest

from careops.corpus import normalize_page
from careops.ingest import MAX_WORDS, OVERLAP_WORDS, chunk_page
from careops.retrieval import Hit, RerankedIndex, ranked, retrieve
from careops.retrieval_eval import score_hits


class StaticIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k):
        return self.hits[:top_k]


class RetrievalTests(unittest.TestCase):
    def test_chunk_offsets_overlap_and_article_boundaries_are_stable(self):
        raw = "  Pasal 10  \n" + " ".join(f"kata{n}" for n in range(110))
        raw += "\n Pasal 11A\n" + "kewajiban " * 15
        chunks = list(chunk_page("example", 8, raw))
        self.assertEqual(chunks, list(chunk_page("example", 8, raw)))
        self.assertEqual([chunk.article for chunk in chunks], ["Pasal 10", "Pasal 10", "Pasal 11A"])
        self.assertEqual(len(chunks[0].text.split()), MAX_WORDS)
        self.assertEqual(
            chunks[0].text.split()[-OVERLAP_WORDS:], chunks[1].text.split()[:OVERLAP_WORDS]
        )
        page = normalize_page(raw)
        for chunk in chunks:
            self.assertEqual(chunk.text, page[chunk.start:chunk.end])
        self.assertEqual(list(chunk_page("example", 9, "only footer")), [])

    def test_rank_fusion_deduplicates_page_and_packet_keeps_source_span(self):
        chunks = [*chunk_page("a", 1, "peraturan " * 125),
                  *chunk_page("b", 2, "pembayaran " * 15)]
        first, second, third = chunks
        lexical = StaticIndex([Hit(first, 0.8), Hit(second, 0.7), Hit(third, 0.4)])
        dense = StaticIndex([Hit(third, 0.9), Hit(first, 0.8)])
        packet = retrieve(RerankedIndex(lexical, dense), "pertanyaan", 3)
        self.assertEqual(len(packet["items"]), 2)
        self.assertEqual({item["source_id"] for item in packet["items"]}, {"a", "b"})
        self.assertTrue(all(item["location"].startswith("PDF page") for item in packet["items"]))
        self.assertTrue(all(0 < item["score"] < 1 for item in packet["items"]))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieve(lexical, " ")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            retrieve(lexical, "pertanyaan", 0)

    def test_hits_distinguish_page_recall_from_source_match(self):
        chunks = [*chunk_page("a", 1, "peraturan " * 15),
                  *chunk_page("a", 2, "pembayaran " * 15)]
        results = ranked(chunks, [0.8, 0.7], 5)
        expected = [{"source_id": "a", "pdf_page": 2}, {"source_id": "b", "pdf_page": 3}]
        self.assertEqual(score_hits(results, expected), {
            "page_recall": 0.5, "reciprocal_rank": 0.5, "source_hit": True,
        })
        self.assertEqual(ranked(chunks, [0, 0], 5, positive_only=True), [])
