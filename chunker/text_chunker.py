"""
text_chunker.py — Text splitting strategies for RAG pipelines

Strategies:
  - Fixed-size with overlap
  - Recursive (split on paragraphs → sentences → words)
  - Sentence-aware (split on sentence boundaries)
  - Token-aware (count tokens, not characters)
"""
import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    text: str
    start: int       # character offset in source
    end: int
    chunk_id: int
    metadata: dict   # source filename, page, etc.


class FixedSizeChunker:
    """Split text into fixed-size chunks with overlap.

    Simple and fast. Works well when text structure is uniform.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start=start,
                    end=end,
                    chunk_id=chunk_id,
                    metadata=metadata,
                ))
                chunk_id += 1
            start = end - self.overlap if end < len(text) else end

        return chunks


class RecursiveChunker:
    """Split on natural boundaries: paragraphs → sentences → words → chars.

    Best general-purpose chunker. Preserves document structure.
    """

    # Separators tried in order (most natural → least)
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        raw_chunks = list(self._split_recursive(text, self.SEPARATORS))
        return self._merge_chunks(raw_chunks, metadata)

    def _split_recursive(self, text: str, separators: list[str]) -> Iterator[str]:
        if not text.strip():
            return
        if len(text) <= self.chunk_size or not separators:
            yield text
            return

        sep = separators[0]
        remaining = separators[1:]

        parts = text.split(sep) if sep else list(text)
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    # current fits — recurse to split further if still too big
                    yield from self._split_recursive(current, remaining)
                current = part

        if current:
            yield from self._split_recursive(current, remaining)

    def _merge_chunks(self, raw_chunks: list[str], metadata: dict) -> list[Chunk]:
        chunks = []
        pos = 0
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                pos += len(raw_chunks[i]) + 1
                continue
            chunks.append(Chunk(
                text=text,
                start=pos,
                end=pos + len(text),
                chunk_id=len(chunks),
                metadata=metadata,
            ))
            pos += len(text)
        return chunks


class SentenceChunker:
    """Split on sentence boundaries, group into chunks of ~N sentences."""

    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self.SENTENCE_RE.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        step = max(1, self.sentences_per_chunk - self.overlap_sentences)
        pos = 0

        for i in range(0, len(sentences), step):
            group = sentences[i : i + self.sentences_per_chunk]
            chunk_text = " ".join(group)
            chunks.append(Chunk(
                text=chunk_text,
                start=pos,
                end=pos + len(chunk_text),
                chunk_id=len(chunks),
                metadata=metadata,
            ))
            pos += sum(len(s) + 1 for s in group)

        return chunks


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances language models
    by giving them access to external knowledge. Instead of relying solely on training
    data, RAG retrieves relevant documents at query time and uses them as context.

    The process has three main steps. First, documents are split into chunks and
    embedded into vectors. Second, when a query arrives, it is embedded and similar
    chunks are retrieved. Third, the retrieved chunks are passed to the LLM as context.

    This approach has several advantages. The knowledge base can be updated without
    retraining. Responses can cite sources. Hallucinations are reduced significantly.
    """

    print("=== Fixed-size chunker ===")
    for chunk in FixedSizeChunker(chunk_size=200, overlap=30).split(sample):
        print(f"  [{chunk.chunk_id}] {chunk.text[:80]}...")

    print("\n=== Recursive chunker ===")
    for chunk in RecursiveChunker(chunk_size=200, overlap=20).split(sample):
        print(f"  [{chunk.chunk_id}] {chunk.text[:80]}...")

    print("\n=== Sentence chunker ===")
    for chunk in SentenceChunker(sentences_per_chunk=3).split(sample):
        print(f"  [{chunk.chunk_id}] {chunk.text[:80]}...")
