import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from injester.models import TASK_QUEUE, TEMPORAL_HOST, IngestRequest, QueryRequest
from injester.storage import index_store
from injester.workflows.ingest_workflow import IngestDocumentWorkflow
from injester.workflows.query_workflow import QueryDocumentWorkflow

QUESTION_PROMPT = "ASK: \033[2m[or type exit to quit]\033[0m "


async def connect_client() -> Client:
    return await Client.connect(TEMPORAL_HOST, data_converter=pydantic_data_converter)


def doc_id_from_path(pdf_path: str) -> str:
    return Path(pdf_path).stem


async def run_ingest(pdf_path: str, doc_id: str | None) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)
    resolved_doc_id = doc_id or doc_id_from_path(pdf_path)
    client = await connect_client()
    request = IngestRequest(doc_id=resolved_doc_id, pdf_path=str(path.resolve()))
    print(f"Starting ingest workflow for '{resolved_doc_id}'...")
    result = await client.execute_workflow(
        IngestDocumentWorkflow.run,
        request,
        id=f"ingest-{resolved_doc_id}-{uuid.uuid4()}",
        task_queue=TASK_QUEUE,
    )
    print(
        f"Ingest complete: doc_id={result.doc_id}, chunks={result.chunk_count}, "
        f"created_at={result.created_at}"
    )


async def run_ask(client: Client, doc_id: str, query: str) -> None:
    request = QueryRequest(doc_id=doc_id, query=query)
    print(f"Querying '{doc_id}'...")
    result = await client.execute_workflow(
        QueryDocumentWorkflow.run,
        request,
        id=f"query-{doc_id}-{uuid.uuid4()}",
        task_queue=TASK_QUEUE,
    )
    print(f"\nAnswer ({result.contexts_used} contexts):\n{result.answer}\n")


async def run_ask_interactive(doc_id: str) -> None:
    if not index_store.index_exists(doc_id):
        print(f"Error: no index found for doc_id '{doc_id}'. Run ingest first.")
        sys.exit(1)
    client = await connect_client()
    print(f"\nRAG ready for '{doc_id}'.\n")
    while True:
        query = input(QUESTION_PROMPT)
        if query.lower() in ("exit", "quit"):
            break
        if not query.strip():
            continue
        await run_ask(client, doc_id, query)


def run_delete(doc_id: str) -> None:
    if not index_store.delete_index(doc_id):
        print(f"Error: no index found for doc_id '{doc_id}'.")
        sys.exit(1)
    print(f"Deleted index for '{doc_id}'.")


def run_list() -> None:
    indexes = index_store.list_indexes()
    if not indexes:
        print("No ingested documents yet.")
        return
    print("Ingested documents:")
    for entry in indexes:
        print(
            f"  - {entry['doc_id']}: {entry['chunk_count']} chunks "
            f"from {entry['source_path']} ({entry['created_at']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Documentation injester Temporal client"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a PDF into the durable index"
    )
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")
    ingest_parser.add_argument(
        "--doc-id", help="Document ID (defaults to PDF filename stem)"
    )
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question against an ingested document"
    )
    ask_parser.add_argument("query", nargs="?", help="Question to ask")
    ask_parser.add_argument("--doc", required=True, dest="doc_id", help="Document ID")
    ask_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter an interactive question loop",
    )
    delete_parser = subparsers.add_parser(
        "delete", help="Delete an ingested document index"
    )
    delete_parser.add_argument(
        "--doc", required=True, dest="doc_id", help="Document ID"
    )
    subparsers.add_parser("list", help="List ingested documents")
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        await run_ingest(args.pdf_path, args.doc_id)
    elif args.command == "ask":
        if not index_store.index_exists(args.doc_id):
            print(
                f"Error: no index found for doc_id '{args.doc_id}'. Run ingest first."
            )
            sys.exit(1)
        if args.interactive:
            await run_ask_interactive(args.doc_id)
        elif args.query:
            client = await connect_client()
            await run_ask(client, args.doc_id, args.query)
        else:
            print("Error: provide a query or use --interactive")
            sys.exit(1)
    elif args.command == "delete":
        run_delete(args.doc_id)
    elif args.command == "list":
        run_list()


if __name__ == "__main__":
    asyncio.run(main())
