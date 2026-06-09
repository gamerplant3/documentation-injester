# Documentation ingester & RAG pipeline

Updated to move orchestration to Temporal (exploration). I maintained the RAG pipeline for ingesting docs with semantic chunking and answering questions with two-stage retrieval (embed + rerank) using Cohere. The new addition is running it as a Temporal workflow with disk-persisted indexes.

### Architecture

```
[CLI client] --> [Temporal dev server] --> [Python worker] --> [Cohere API]
                      |                          |
                      |           extract -> chunk -> embed -> persist
                      |                          |
                      |              retrieve -> rerank -> answer
                      v                          v
               [Temporal Web UI]       [storage/index/{doc_id}]
```

### Known limitation

Ingest embeds text twice: Langchain's `SemanticChunker` embeds sentences to find chunk boundaries, then `embed_chunks` embeds final chunks again for the search index. The first-pass embeddings are not reused. Left as-is intentionally for chunk quality. Tradeoff: higher Cohere API usage on free tier.

### Features
- Semantic chunking splits text dynamically by embedding shifts (95th percentile breakpoints), not fixed character counts.
- PDF ingestion via `IngestDocumentWorkflow` (`uv run client.py ingest <pdf>`), will go through the following steps: extract -> semantic chunk -> batched embed -> persist to `storage/index/{doc_id}/`.
- Querying via `QueryDocumentWorkflow` (`uv run client.py ask "<question>" --doc <doc_id>`) will load persisted chunks and embeddings without re-embedding.
- Two-stage retrieval: vector search (top 20) via `embed-english-v3.0`, then `rerank-v3.5` to top 4 contexts for `command-r-plus-08-2024`. This decouples the initial vector similarity matching from actual semantic answering relevance, shrinking the final prompt window and lowering inference overhead.
- Cohere free-tier pacing: max 96 strings per embed call with a 1.0s delay between batches.
- Claim-check storage: large text, chunks, and embeddings live on disk under `storage/`; workflows pass only `doc_id` refs.

### Setup

```powershell
uv sync
```

Install the Temporal CLI (Windows):

```powershell
winget install -e --id Temporal.TemporalCLI
```

Close and reopen the terminal, then verify:

```powershell
temporal --version
```

In `.env`:

```text
COHERE_API_KEY=key_here
```

### Usage

Three terminals (PowerShell):

```powershell
# Terminal 1 - Temporal dev server (Web UI at http://localhost:8233)
temporal server start-dev

# Terminal 2 - worker
uv run worker.py

# Terminal 3 - ingest and query
uv run client.py ingest "C:\path\to\USER_GUIDE.pdf"
uv run client.py list
uv run client.py ask "What is this document about?" --doc USER_GUIDE
uv run client.py ask --doc USER_GUIDE --interactive
uv run client.py delete --doc USER_GUIDE
```

`delete` removes the persisted index for a document from `storage/` (chunks, embeddings, and manifest entry). It does not delete the source PDF.

The `doc_id` defaults to the PDF filename without `.pdf` (e.g. `USER_GUIDE.pdf` becomes `USER_GUIDE`). Use `client.py list` to see all docs.

Interactive mode accepts multiple questions in a row. Type `exit` or `quit` to leave the loop. Ctrl+C also stops it, but `exit`/`quit` is the intended clean exit.

Re-querying reuses the on-disk index; embeddings are not regenerated unless ingest runs again.

### Troubleshooting

**`'dict' object has no attribute 'doc_id'`** - Temporal can pass activity payloads as plain dicts instead of typed models; `retrieve_and_rerank` expected `QueryRequest`. Fixed with Pydantic models and `pydantic_data_converter` on `Client.connect` (client and worker).

### Example Session

```
PS C:\Users\[   ]\Documents\Local\documentation-injester> uv run client.py ingest "C:\Users\[   ]\...path...\USER_GUIDE.pdf"
Starting ingest workflow for 'USER_GUIDE'...
Ingest complete: doc_id=USER_GUIDE, chunks=12, created_at=2026-06-09T17:58:13Z
PS C:\Users\[   ]\Documents\Local\documentation-injester> uv run client.py ask "What is this document about?" --doc USER_GUIDE                                                           
Querying 'USER_GUIDE'...

Answer (4 contexts):
This document is the user guide for a web application called the Design Automation Controller. It provides instructions and information for users on how to set up, configure, and use the application to automate various tasks related to Design Automation and file processing.

The user guide covers various topics, including initial setup, user access control, uploading Design Automation bundles, project syncing, automatic project enablement, creating automations, trigger options, filter options, action options, running one-off actions, and viewing run history and status. It provides step-by-step instructions and explanations for each of these tasks, ensuring users can effectively utilize the application's features.

PS C:\Users\[   ]\Documents\Local\documentation-injester> uv run client.py ask --doc USER_GUIDE --interactive                                                                            

RAG ready for 'USER_GUIDE'.

ASK: [or type exit to quit] briefly list the filter options for the DAC
Querying 'USER_GUIDE'...

Answer (4 contexts):
The Design Automation Controller (DAC) offers the following filter options when creating an automation:

1. Include / Exclude Hubs: Include or exclude files based on the hubs they belong to.
2. Include / Exclude Projects: Include or exclude files based on the projects they are associated with.
3. Include / Exclude Folder Names: Include or exclude files based on the folder names or subfolders.
4. Include / Exclude Specific Folder: Include or exclude files based on specific folder URNs.  
5. Include / Exclude File Names: Include or exclude files with specific names.
6. Include / Exclude File Extensions: Include or exclude files with specific file extensions.  

ASK: [or type exit to quit] how do i create an automation?
Querying 'USER_GUIDE'...

Answer (4 contexts):
To create an automation, follow these steps:

1. Go to the 'Automations' page.
2. Click 'Create New Automation'.
3. Add one or more triggers:
   - Choose a trigger type from the dropdown.
   - Select the specific trigger from the list.
   - Set any additional trigger-specific settings as required.
   - Press 'Save Changes'.
4. Add one or more filters:
   - Choose a filter type from the dropdown.
   - Select the desired filter from the list.
   - Adjust any additional filter-specific settings.
   - Press 'Save Changes'.
5. Add one or more actions:
   - Choose an action type from the dropdown.
   - Select the action you want to perform from the list.
   - Configure any additional action-specific settings.
   - Press 'Save Changes'.
6. Edit the automation name if needed.
7. Press 'Create New Automation' to finalize the process.

You have various trigger options, such as Schedule, On File Created, On File Modified, On Model Publish (Cloud Workshared), and Manual Trigger. The filters include options like Include/Exclude Hubs, Projects, Folder Names, Specific Folders, File Names, and File Extensions. For actions, you can choose from Publish Revit Models, Test Action, Web API Action (HTTP Request), and more.

ASK: [or type exit to quit] what config should i use for fully automatic project mirroring?
Querying 'USER_GUIDE'...

Answer (4 contexts):
To set up fully automatic project mirroring with comprehensive synchronization, follow these steps:

1. Enable 'Auto-enable new projects' for the desired hubs: Go to "Settings," then "Accounts," and click on the specific hub (account) you want to configure. Toggle the "Auto-enable new projects" switch, and the setting will be saved automatically. Repeat this for all hubs you want to enable.

2. Enable global 'Auto-sync new projects folders and files': Navigate to "Settings" and click on "Sync Settings." Toggle the "Auto-sync new projects folders and files" checkbox.

3. Set the sync schedule: On the same "Sync Settings" page, adjust the "Sync Job Schedule" to your preferred interval, with a minimum of 10 minutes. The default is set to hourly.

This configuration ensures that new projects are automatically enabled, webhooks are registered, and all enabled projects are synced according to the schedule you set. Real-time webhook updates will also continue to function alongside the scheduled syncs, providing redundancy and catching any changes that webhooks might miss.

ASK: [or type exit to quit] exit
```
### Screenshots

Temporal UI

<img width="2040" height="1320" alt="screenie 1" src="https://github.com/user-attachments/assets/ae05981f-3031-4e90-9598-bf8db8c002b2" />

<img width="2040" height="1320" alt="screenie 2" src="https://github.com/user-attachments/assets/1e364698-56e6-48ed-b6ee-3d99ec50ac09" />
