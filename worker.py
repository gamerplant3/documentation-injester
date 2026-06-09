import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from injester.activities.ingestion import (
    embed_chunks,
    extract_pdf,
    persist_index,
    semantic_chunk,
)
from injester.activities.retrieval import generate_answer, retrieve_and_rerank
from injester.models import TASK_QUEUE, TEMPORAL_HOST
from injester.workflows.ingest_workflow import IngestDocumentWorkflow
from injester.workflows.query_workflow import QueryDocumentWorkflow


async def main() -> None:
    client = await Client.connect(TEMPORAL_HOST, data_converter=pydantic_data_converter)
    with ThreadPoolExecutor(max_workers=4) as activity_executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[IngestDocumentWorkflow, QueryDocumentWorkflow],
            activities=[
                extract_pdf,
                semantic_chunk,
                embed_chunks,
                persist_index,
                retrieve_and_rerank,
                generate_answer,
            ],
            activity_executor=activity_executor,
        )
        print(f"Worker started on task queue '{TASK_QUEUE}' ({TEMPORAL_HOST})")
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
