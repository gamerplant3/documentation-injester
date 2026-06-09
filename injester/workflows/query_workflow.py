from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from injester.activities.retrieval import generate_answer, retrieve_and_rerank
    from injester.models import COHERE_RETRY, QueryRequest, QueryResult


@workflow.defn
class QueryDocumentWorkflow:
    @workflow.run
    async def run(self, request: QueryRequest) -> QueryResult:
        contexts = await workflow.execute_activity(
            retrieve_and_rerank,
            request,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=COHERE_RETRY,
        )
        return await workflow.execute_activity(
            generate_answer,
            args=[request, contexts],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=COHERE_RETRY,
        )
