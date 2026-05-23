# rag-from-scratch

> Build a complete RAG (Retrieval-Augmented Generation) pipeline without LangChain — pure Python, OpenAI embeddings, FAISS, BM25 hybrid retrieval.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![OpenAI](https://img.shields.io/badge/openai-1.x-green) ![License](https://img.shields.io/badge/license-MIT-green)

```
Q: What is FastAPI and what are its key features?
A: FastAPI is a modern Python web framework for building APIs based on standard
   Python type hints. Key features: async/await support, auto OpenAPI docs,
   Pydantic validation, one of the fastest Python frameworks available.
Sources: ['fastapi.txt'] (score: 0.847)
```

## Architecture

```
Documents → Chunker → Embedder → FAISS Index
                                      ↑
Query → Embedder ──── Dense Search ──┤  → RRF Fusion → LLM → Answer
      → Tokenizer ─── BM25 Search ───┘
```

## Files

```
chunker/
└── text_chunker.py    # FixedSize, Recursive, Sentence chunkers
retriever/
└── vector_retriever.py  # DenseRetriever (FAISS) + BM25Retriever + HybridRetriever (RRF)
pipeline.py            # End-to-end RAG pipeline
```

## Quick Start

```bash
git clone https://github.com/bhupendra05/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python pipeline.py
```

## Usage

```python
from pipeline import RAGPipeline, RAGConfig

rag = RAGPipeline(RAGConfig(chunk_size=512, top_k=5))

# Index documents
rag.add_file("my_docs/readme.txt")
rag.add_text("FastAPI is a modern Python web framework...", source="fastapi")

# Query
answer = rag.query("What are the key features of FastAPI?")
print(answer)

# Query with sources
result = rag.query_with_sources("How do I add authentication?")
print(result["answer"])
print(result["sources"])  # which chunks were used
```

## Chunking Strategies

```python
from chunker.text_chunker import RecursiveChunker, SentenceChunker, FixedSizeChunker

# Best general-purpose (tries paragraphs → sentences → words)
chunks = RecursiveChunker(chunk_size=512, overlap=64).split(text)

# For well-structured text with clear sentences
chunks = SentenceChunker(sentences_per_chunk=5, overlap_sentences=1).split(text)

# Simple, fast, uniform
chunks = FixedSizeChunker(chunk_size=512, overlap=64).split(text)
```

## Hybrid Retrieval (Dense + BM25)

```python
from retriever.vector_retriever import HybridRetriever, DenseRetriever, BM25Retriever

# Dense-only (semantic similarity via OpenAI embeddings + FAISS)
retriever = DenseRetriever()
retriever.add(texts, metadata)
results = retriever.search("web framework", top_k=5)

# Sparse-only (BM25 keyword matching — no API calls)
retriever = BM25Retriever()
retriever.add(texts)
results = retriever.search("FastAPI web framework")

# Hybrid (RRF fusion — usually best)
retriever = HybridRetriever(dense_weight=0.6)
retriever.add(texts, metadata)
results = retriever.search("FastAPI web framework")
```

## Why Build RAG from Scratch?

| Aspect | This repo | LangChain |
|--------|-----------|-----------|
| Complexity | Simple, readable | Abstracted, hard to debug |
| Dependencies | 3 packages | 50+ packages |
| Customization | Easy to modify | Requires understanding internals |
| Learning value | High | Low |
| Production control | Full | Limited |

## License

MIT © bhupendra05
