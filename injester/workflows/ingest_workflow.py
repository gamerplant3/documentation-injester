from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from injester.activities.ingestion import (
        embed_chunks,
        extract_pdf,
        persist_index,
        semantic_chunk,
    )
    from injester.models import COHERE_RETRY, IndexRef, IngestRequest


@workflow.defn
class IngestDocumentWorkflow:
    @workflow.run
    async def run(self, request: IngestRequest) -> IndexRef:
        doc_id = await workflow.execute_activity(
            extract_pdf,
            request,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=COHERE_RETRY,
        )
        doc_id = await workflow.execute_activity(
            semantic_chunk,
            doc_id,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=COHERE_RETRY,
        )
        doc_id = await workflow.execute_activity(
            embed_chunks,
            doc_id,
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=COHERE_RETRY,
        )
        return await workflow.execute_activity(
            persist_index,
            request,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=COHERE_RETRY,
        )
