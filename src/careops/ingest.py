"""Turn checksum-pinned PDFs into deterministic, citeable passages."""

import re
from dataclasses import dataclass
from hashlib import sha256

from careops.corpus import iter_pdf_pages, normalize_page

MIN_PAGE_CHARS = 100
MAX_WORDS = 100
OVERLAP_WORDS = 20
ARTICLE_HEADING = re.compile(r"(?m)^[ \t]*(Pasal[ \t]+[0-9]+[A-Z]?)[ \t]*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    page: int
    start: int
    end: int
    text: str
    article: str | None

    @property
    def location(self):
        place = f"PDF page {self.page}, chars {self.start}:{self.end}"
        return f"{place}, {self.article}" if self.article else place


def chunk_page(source_id, page, raw):
    text = normalize_page(raw)
    if len(text) < MIN_PAGE_CHARS:
        return

    sections = [(0, None)]
    for match in ARTICLE_HEADING.finditer(raw):
        prefix = normalize_page(raw[:match.start()])
        start = len(prefix) + bool(prefix)
        if start == 0:
            sections[0] = (0, normalize_page(match[1]))
        else:
            sections.append((start, normalize_page(match[1])))
    sections.append((len(text), None))

    for (section_start, article), (section_end, _) in zip(sections, sections[1:]):
        words = list(re.finditer(r"\S+", text[section_start:section_end]))
        for first in range(0, len(words), MAX_WORDS - OVERLAP_WORDS):
            last = min(first + MAX_WORDS, len(words))
            start = section_start + words[first].start()
            end = section_start + words[last - 1].end()
            passage = text[start:end]
            fingerprint = sha256(passage.encode("utf-8")).hexdigest()[:12]
            yield Chunk(
                f"{source_id}:p{page}:{start}-{end}:{fingerprint}",
                source_id, page, start, end, passage, article,
            )
            if last == len(words):
                break


def load_chunks(sources, raw_directory):
    chunks = []
    for row in sorted(sources, key=lambda item: item["source"]["source_id"]):
        source_id = row["source"]["source_id"]
        for page, raw in iter_pdf_pages(row, raw_directory):
            chunks.extend(chunk_page(source_id, page, raw))
    if not chunks:
        raise ValueError("No searchable text in the selected PDFs")
    return chunks
