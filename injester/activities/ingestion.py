import time
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from pypdf import PdfReader
from temporalio import activity
from injester.cohere_client import get_cohere_client
from injester.models import (
    COHERE_EMBED_MODEL,
    EMBED_BATCH_DELAY_SECONDS,
    EMBED_BATCH_SIZE,
    IndexRef,
    IngestRequest,
    utc_now_iso,
)
from injester.storage import index_store


class CohereEmbeddingWrapper:
    """Minimal wrapper to make Cohere Embed v3 compatible with LangChain's chunker."""

    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        all_embeddings = []
        activity.logger.info(
            "Batching %s sentences for embedding generation", len(texts)
        )
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            res = self.client.embed(
                texts=batch,
                model=COHERE_EMBED_MODEL,
                input_type="search_document",
            )
            all_embeddings.extend(res.embeddings.float)
            time.sleep(EMBED_BATCH_DELAY_SECONDS)
        return all_embeddings


@activity.defn
def extract_pdf(request: IngestRequest) -> str:
    """Extracts raw text from a PDF and stores it on disk."""
    activity.logger.info("Parsing %s", request.pdf_path)
    reader = PdfReader(request.pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        raise ValueError(
            f"No text extracted from PDF '{request.pdf_path}'. "
            "The file may be empty, scanned, or image-only."
        )
    index_store.save_work_text(request.doc_id, text)
    return request.doc_id


@activity.defn
def semantic_chunk(doc_id: str) -> str:
    """Splits text dynamically based on changes in semantic meaning."""
    activity.logger.info("Performing semantic chunking for %s", doc_id)
    text = index_store.load_work_text(doc_id)
    co = get_cohere_client()
    embedding_model = CohereEmbeddingWrapper(co)
    chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    docs = chunker.create_documents([text])
    chunks = [doc.page_content for doc in docs]
    activity.logger.info("Created %s semantic chunks for %s", len(chunks), doc_id)
    index_store.save_work_chunks(doc_id, chunks)
    return doc_id


@activity.defn
def embed_chunks(doc_id: str) -> str:
    """Generates matrix embeddings for dense retrieval using batched Cohere calls."""
    activity.logger.info("Embedding semantic chunks for %s", doc_id)
    chunks = index_store.load_work_chunks(doc_id)
    co = get_cohere_client()
    all_embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        res = co.embed(
            texts=batch,
            model=COHERE_EMBED_MODEL,
            input_type="search_document",
        )
        all_embeddings.extend(res.embeddings.float)
        time.sleep(EMBED_BATCH_DELAY_SECONDS)
    index_store.save_work_embeddings(doc_id, np.array(all_embeddings))
    return doc_id


@activity.defn
def persist_index(request: IngestRequest) -> IndexRef:
    """Moves work artifacts into the durable index store and updates the manifest."""
    chunks = index_store.load_work_chunks(request.doc_id)
    created_at = utc_now_iso()
    index_store.finalize_index(
        request.doc_id,
        request.pdf_path,
        len(chunks),
        created_at,
    )
    activity.logger.info(
        "Persisted index for %s (%s chunks)", request.doc_id, len(chunks)
    )
    return IndexRef(
        doc_id=request.doc_id,
        chunk_count=len(chunks),
        source_path=request.pdf_path,
        created_at=created_at,
    )
