import os
import sys
import time # using free api token, so throttle to avoid rate limits
from dotenv import load_dotenv
load_dotenv()

import cohere
import numpy as np
from pypdf import PdfReader
from langchain_experimental.text_splitter import SemanticChunker


# Initialize Cohere Client
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    print("Error: COHERE_API_KEY environment variable not set.")
    sys.exit(1)

co = cohere.ClientV2(api_key=api_key)

def load_pdf(file_path):
    """Extracts raw text from a PDF file."""
    print(f"Parsing {file_path}...")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

class CohereEmbeddingWrapper:
    """A minimal wrapper to make Cohere Embed v3 compatible with LangChain's chunker."""
    def __init__(self, client):
        self.client = client
        
    def embed_documents(self, texts):
        # Cohere V2 max batch size is 96
        batch_size = 96
        all_embeddings = []
        
        print(f" -> Batching {len(texts)} sentences for embedding generation...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = self.client.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            all_embeddings.extend(res.embeddings.float)
            time.sleep(1.0)
            
        return all_embeddings

def semantic_chunk_text(text):
    """Splits text dynamically based on changes in semantic meaning."""
    print("Performing semantic chunking...")
    embedding_model = CohereEmbeddingWrapper(co)
    
    # Using percentile threshold: splits text when differences cross the 95th percentile
    chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    docs = chunker.create_documents([text])
    
    chunks = [doc.page_content for doc in docs]
    print(f"Created {len(chunks)} smart semantic chunks.")
    return chunks

def build_vector_store(chunks):
    """Generates matrix embeddings for dense retrieval using batches."""
    print("Embedding final semantic chunks with Cohere Embed v3...")
    batch_size = 96
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        res = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        all_embeddings.extend(res.embeddings.float)
        time.sleep(1.0)
        
    return np.array(all_embeddings)

def retrieve_and_rerank(query, chunks, chunk_embeddings, top_k=20, final_top=4):
    """Two-Stage Retrieval: Matrix dot-product vector search followed by Cohere Rerank."""
    # 1. Vector Search
    query_res = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query")
    query_vec = np.array(query_res.embeddings.float[0])
    
    # cosine similarity via dot product (assuming normalized vectors)
    scores = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]
    candidate_chunks = [chunks[idx] for idx in top_indices]
    
    # 2. Cohere Rerank
    print(f"Reranking top {len(candidate_chunks)} candidates down to {final_top} critical contexts...")
    rerank_res = co.rerank(
        query=query,
        documents=candidate_chunks,
        model="rerank-v3.5",
        top_n=final_top
    )
    
    best_contexts = []
    for hit in rerank_res.results:
        best_contexts.append(candidate_chunks[hit.index])
        
    return best_contexts

def answer_query(query, contexts):
    """Context for the LLM's reply."""
    system_prompt = (
        "Answer the user's question accurately using ONLY the context provided below. If the context does "
        "not contain the answer, say 'I cannot find that in the loaded documentation.'\n\n"
        f"CONTEXT:\n" + "\n---\n".join(contexts)
    )
    
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.message.content[0].text

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run rag.py <path_to_file_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    raw_text = load_pdf(pdf_path)
    chunks = semantic_chunk_text(raw_text)
    embeddings = build_vector_store(chunks)
    
    print("\n🚀 RAG Pipeline Ready! Ask anything (type 'exit' to quit):\n")
    while True:
        query = input("🤖 Question: ")
        if query.lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue
            
        contexts = retrieve_and_rerank(query, chunks, embeddings)
        answer = answer_query(query, contexts)
        print(f"\n💡 Answer:\n{answer}\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()