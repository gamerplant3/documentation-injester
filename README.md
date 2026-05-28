# Enterprise documentation ingester & RAG pipeline

A local RAG pipeline designed to ingest complex technical manuals (e.g., PDFs of large, dense design guides, developer guides, any other dense documentation...) and deliver accurate, hallucination-resistant answers.

Combines semantic chunking with a 2-stage retrieval (embed + rerank) architecture. Using Cohere's latest V2 API.


### Architecture

```
[Text Source] ──► [Semantic Chunking] ──► [Step 1: Vector Search] ──► [Step 2: Cohere Rerank] ──► [Command-R-Plus LLM]

```

* Semantic Chunking: Splits text dynamically by analyzing shifts in sentence embeddings rather than using arbitrary/hardcoded character counts. I used `rerank-v3.5` to decouple initial vector similarity matching from actual semantic answering relevance, shrinking the final prompt window and lowering inference overhead.
* Two-Stage Retrieval: Filters document matrices quickly down to 20 candidates using `embed-english-v3.0`, then uses `rerank-v3.5` to pick the absolute best 4 contexts for the LLM.
* Safeguards: Includes batching logic (max 96 strings per call) and manual throttling since I'm using the free API (rate limits and token ceilings). I managed the sliding-window token multiplication behavior of LangChain's chunker using delayed pacing for extraction.


### Setup

```bash
uv init
uv add cohere numpy pypdf langchain-experimental python-dotenv

```

In `.env` :

```text
COHERE_API_KEY=key_here

```

### Usage

1. Save the PDF (or text file) of your target documentation in the project directory.
2. Run with:

```bash
uv run rag.py user_guide.pdf

```

### Example Session


