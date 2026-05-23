"""
pipeline.py — End-to-end RAG pipeline (no LangChain)

Steps:
  1. Load documents
  2. Chunk text
  3. Embed + index chunks
  4. Retrieve relevant chunks for a query
  5. Generate answer with GPT
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from chunker.text_chunker import RecursiveChunker, Chunk
from retriever.vector_retriever import HybridRetriever, SearchResult

client = OpenAI()


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    model: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Cite the relevant parts of the context in your answer."
    )


class RAGPipeline:
    """Complete RAG pipeline — index documents, answer questions."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.retriever = HybridRetriever()
        self._indexed = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "document") -> int:
        """Add raw text to the index. Returns number of chunks created."""
        chunks = self.chunker.split(text, metadata={"source": source})
        self.retriever.add(
            texts=[c.text for c in chunks],
            metadata=[c.metadata for c in chunks],
        )
        self._indexed = True
        return len(chunks)

    def add_file(self, filepath: str) -> int:
        """Read a text file and add it to the index."""
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        return self.add_text(text, source=os.path.basename(filepath))

    def add_texts(self, texts: list[str], sources: list[str] | None = None) -> int:
        """Add multiple texts at once."""
        total = 0
        for i, text in enumerate(texts):
            source = (sources or [])[i] if sources and i < len(sources) else f"doc-{i}"
            total += self.add_text(text, source=source)
        return total

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[SearchResult]:
        """Retrieve most relevant chunks for a query."""
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call add_text() first.")
        return self.retriever.search(query, top_k=self.config.top_k)

    # ── Generation ────────────────────────────────────────────────────────────

    def _build_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source}, score: {r.score:.3f})\n{r.text}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, stream: bool = False) -> str:
        """Retrieve context and generate an answer."""
        results = self.retrieve(question)

        if not results:
            return "No relevant documents found to answer this question."

        context = self._build_context(results)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        if stream:
            answer = ""
            with client.chat.completions.stream(
                model=self.config.model,
                messages=messages,
            ) as s:
                for text in s.text_stream:
                    print(text, end="", flush=True)
                    answer += text
            print()
            return answer
        else:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""

    def query_with_sources(self, question: str) -> dict:
        """Return answer + retrieved sources."""
        results = self.retrieve(question)
        context = self._build_context(results)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"text": r.text[:200], "source": r.metadata.get("source"), "score": r.score}
                for r in results
            ],
            "usage": response.usage.model_dump() if response.usage else {},
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = RAGPipeline(RAGConfig(chunk_size=300, top_k=3))

    # Index sample documents
    docs = [
        ("Python is a high-level, interpreted programming language known for its simplicity. "
         "It was created by Guido van Rossum and released in 1991. Python emphasizes code "
         "readability and supports multiple programming paradigms including procedural, "
         "object-oriented, and functional programming."),

        ("FastAPI is a modern Python web framework for building APIs. It is based on standard "
         "Python type hints and is one of the fastest Python frameworks available. FastAPI "
         "automatically generates OpenAPI documentation. It supports async/await natively "
         "and uses Pydantic for data validation."),

        ("PostgreSQL is a powerful, open-source relational database management system. "
         "It supports complex queries, foreign keys, triggers, views, and stored procedures. "
         "PostgreSQL is ACID-compliant and supports JSON natively through JSONB type. "
         "It has excellent support for full-text search and geospatial data via PostGIS."),

        ("RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving relevant documents "
         "before generating a response. The process involves embedding documents into vectors, "
         "storing them in a vector database, and at query time retrieving the most similar "
         "documents to use as context for the language model."),
    ]

    print("Indexing documents...")
    total = rag.add_texts([d for d in docs], sources=["python.txt", "fastapi.txt", "postgres.txt", "rag.txt"])
    print(f"Indexed {total} chunks\n")

    questions = [
        "What is FastAPI and what are its key features?",
        "How does RAG work?",
        "What database supports JSONB?",
    ]

    for q in questions:
        print(f"Q: {q}")
        result = rag.query_with_sources(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
        print()
