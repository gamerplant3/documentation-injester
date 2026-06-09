import os
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel
from temporalio.common import RetryPolicy

TASK_QUEUE = "injester-task-queue"
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")

# Cohere model names
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_RERANK_MODEL = "rerank-v3.5"
COHERE_CHAT_MODEL = "command-r-plus-08-2024"

# Retrieval limits
RETRIEVAL_TOP_K = 20
RETRIEVAL_FINAL_TOP = 4

# Cohere V2 free-tier pacing: max strings per embed call and delay between batches.
EMBED_BATCH_SIZE = 96
EMBED_BATCH_DELAY_SECONDS = 1.0

COHERE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=8,
)


class IngestRequest(BaseModel):
    doc_id: str
    pdf_path: str


class IndexRef(BaseModel):
    doc_id: str
    chunk_count: int
    source_path: str
    created_at: str


class QueryRequest(BaseModel):
    doc_id: str
    query: str


class QueryResult(BaseModel):
    doc_id: str
    answer: str
    contexts_used: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
