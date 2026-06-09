import numpy as np
from temporalio import activity
from injester.cohere_client import get_cohere_client
from injester.models import (
    COHERE_CHAT_MODEL,
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
    RETRIEVAL_FINAL_TOP,
    RETRIEVAL_TOP_K,
    QueryRequest,
    QueryResult,
)
from injester.storage import index_store


@activity.defn
def retrieve_and_rerank(request: QueryRequest) -> list[str]:
    """Two-stage retrieval: vector search followed by Cohere rerank."""
    chunks = index_store.load_index_chunks(request.doc_id)
    chunk_embeddings = index_store.load_index_embeddings(request.doc_id)
    co = get_cohere_client()
    query_res = co.embed(
        texts=[request.query],
        model=COHERE_EMBED_MODEL,
        input_type="search_query",
    )
    query_vec = np.array(query_res.embeddings.float[0])
    scores = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(scores)[::-1][:RETRIEVAL_TOP_K]
    candidate_chunks = [chunks[idx] for idx in top_indices]
    activity.logger.info(
        "Reranking top %s candidates down to %s for %s",
        len(candidate_chunks),
        RETRIEVAL_FINAL_TOP,
        request.doc_id,
    )
    rerank_res = co.rerank(
        query=request.query,
        documents=candidate_chunks,
        model=COHERE_RERANK_MODEL,
        top_n=RETRIEVAL_FINAL_TOP,
    )
    return [candidate_chunks[hit.index] for hit in rerank_res.results]


@activity.defn
def generate_answer(request: QueryRequest, contexts: list[str]) -> QueryResult:
    """Generates an answer from retrieved contexts using Cohere chat."""
    system_prompt = (
        "Answer the user's question accurately using ONLY the context provided below. If the context does "
        "not contain the answer, say 'I cannot find that in the loaded documentation.'\n\n"
        f"CONTEXT:\n" + "\n---\n".join(contexts)
    )
    co = get_cohere_client()
    response = co.chat(
        model=COHERE_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query},
        ],
    )
    answer = response.message.content[0].text
    return QueryResult(
        doc_id=request.doc_id,
        answer=answer,
        contexts_used=len(contexts),
    )
