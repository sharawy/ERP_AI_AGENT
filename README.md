# Agent‑Driven ERP System

This repository contains a modular, agent‑driven ERP prototype. It demonstrates how local LLMs and composable tools (SQL, vector RAG, analytics helpers) can provide a conversational interface over classic ERP workflows. It is not production‑ready, but organized for experimentation and extension.

## High‑Level Architecture

**Note:** The current implementation has `/orders` endpoints that bypass the router and go directly to the database. In a proper agent-driven design, only `/chat` should exist, with the Router Agent deciding all actions based on user messages.

!!!!!!!!!!!!!!!!!!!!!!!!!THIS SHOULD BE FIXED BEFORE SUBMISSION!!!!!!!!!!!!!!!!!!!!!!!!!

```
┌────────────┐    HTTP    ┌─────────────┐
│ Streamlit  │ ─────────▶ │  FastAPI     │
│   Front‑end│            │  Back‑end    │
└────────────┘            └─────┬───────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
        ┌────▼─────┐       ┌──────▼─────┐       ┌──────▼────┐
        │ Router   │       │ Sales      │       │ Finance   │
        │ Agent    │       │ Agent      │       │ Agent     │
        └────┬─────┘       └─────┬──────┘       └─────┬─────┘
             │                   │                    │
             │             ┌─────▼─────┐        ┌─────▼─────┐
             │             │ Inventory │        │ Analytics │
             │             │ Agent     │        │ Agent     │
             ▼             └───────────┘        └───────────┘
         ┌───────────┐
         │ Tools &   │  SQL + Vector RAG + ML‑like helpers
         │ Utilities │
         └────┬──────┘
              │
         ┌────▼──────┐
         │  SQLite    │  (project_data/erp.db)
         │  Database  │
         └────────────┘
```

- Streamlit UI: Minimal chat interface that talks to the API.
- FastAPI (`app.py`): Exposes `/chat`, `/orders` (GET/POST) and wires all agents.
- Router Agent (`agents/router_agent.py`): LLM‑based domain classification → delegates to domain agents; logs tool calls to DB.
- Domain Agents (`agents/*.py`):
  - Sales: customers, leads, orders, support tickets; RAG‑assisted procedures; lead scoring helper.
  - Finance: invoices, payments; policy RAG; approval thresholds for high‑value invoices.
  - Inventory: stock, suppliers, purchase orders; document RAG.
  - Analytics: saved reports, ad‑hoc analysis; text‑to‑SQL + simple chart specs.
- Tools (`tools/*.py`): `sql_tool.py`, `vector_rag_tool.py` (sentence‑transformers + ChromaDB, with fallbacks), analytics helpers, approval system, audit logging, conversation/entity memory.
- Database (`database.py`): Single SQLite connection, helper APIs. Default DB at `project_data/erp.db`.
- LLM Integration (`llm.py`): Uses LM Studio local server by default (`LM_STUDIO_URL`, default `http://localhost:1234/v1/chat/completions`). Functions are named `call_gemini*` for compatibility, but all calls go to LM Studio unless you change the URL.

## Project Structure (selected)

- `app.py` – FastAPI app and endpoints
- `streamlit_app.py` – Streamlit chat UI (uses `ERP_API_URL`)
- `agents/` – Router, Sales, Finance, Inventory, Analytics
- `tools/` – SQL, Vector RAG (Chroma), analytics (text‑to‑SQL + charts), approvals, audit log, memory
- `models/common.py` – Pydantic models for API
- `database.py` – SQLite helpers
- `project_data/erp.db` – Sample database
- `vector_db/` – Chroma persistent store (pre‑seeded collections)

## Database

- Default path: `data/erp.db` (override with `ERP_DB_PATH`).
- Contains minimal tables used by agents, plus optional `documents` and `saved_reports`.
- On import, `tools/saved_reports.py` attempts to initialize default saved reports if the table exists.
The system uses an SQLite database with the following key tables:
- `customers`: Stores customer information.
- `leads`: Tracks potential sales leads.
- `orders`: Manages customer orders.
- `invoices`: Handles billing and payments.
- `products`: Maintains product catalog and pricing.

Refer to `data/docs/db_schemas.txt` for the full schema.
## Vector RAG

- `tools/vector_rag_tool.py` uses sentence‑transformers (`all‑MiniLM‑L6‑v2`) and ChromaDB.
- If vector deps are missing, it prints a warning and falls back to a simple keyword search over files referenced in the `documents` table.
- Persistent store at `./vector_db` (included).

## API Endpoints

- `POST /chat` – Body: `{ "message": str, "conversation_id": Optional[int] }` → `{ "conversation_id": int, "response": str }`
- `GET /orders?limit=10` – Recent orders with customer names
- `POST /orders` – Create an order. Body: `{ "customer_id": int, "items": [{"product_id": int, "quantity": int}, ...] }`

## Environment Variables

- `LM_STUDIO_URL` – LM Studio chat completions endpoint. Default: `http://localhost:1234/v1/chat/completions`
- `ERP_DB_PATH` – Optional override for SQLite DB path
- `ERP_API_URL` – Used by Streamlit UI to reach the API (default `http://localhost:8000`)

## Prerequisites

- Python 3.10+
- LM Studio (with a chat model loaded) if using the chat/LLM features
- On first run of LM Studio: start the Local Server and load a model

## Setup & Run

### 1) Create a virtual environment and install deps

Windows (PowerShell):

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

macOS/Linux (bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Setup LLm Provider Gemini
Update GOOGLE_API_KEY in .env
Create a `.env` file in the project root with the following content:
```
GOOGLE_API_KEY=key
ERP_DB_PATH=data/erp.db
```
### 3) Run the FastAPI server

From the project root:

```bash
uvicorn app:app --reload
```

The API will be at `http://localhost:8000`.

Test quickly (PowerShell):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -Body (@{ message = 'show recent orders' } | ConvertTo-Json) -ContentType 'application/json'
```

### 4) Run the Streamlit UI (optional)

In a second terminal:

```bash
streamlit run streamlit_app.py
```

If your API runs on a non‑default URL, set before launching:

```bash
$env:ERP_API_URL='http://localhost:8000'   # PowerShell
export ERP_API_URL='http://localhost:8000' # macOS/Linux
```

Open `http://localhost:8501` and chat with the system.

## How It Works (brief)

- Router uses the LLM (`llm.py`) to classify the user request into Sales/Finance/Inventory/Analytics and routes to the corresponding agent.
- Agents use `tools/sql_tool.py` for DB access; some write helper records (e.g., leads, invoices, purchase orders) to demonstrate workflows. Finance enforces approval thresholds via `tools/approval_system.py`.
- RAG tools search Chroma collections or fall back to file keyword search. Agents optionally blend RAG context into answers.
- `tools/audit_logger.py` records tool calls. `tools/memory_manager.py` keeps short conversation context and simple entity memory in `customer_kv`.

## Troubleshooting

- LLM errors like “endpoint not found (404)” → Start LM Studio Local Server and ensure a model is loaded; verify `LM_STUDIO_URL`.
- Vector warnings → Install `sentence-transformers` and `chromadb` (already in `requirements.txt`), then rerun.
- Empty analytics/saved reports → Ensure DB is `project_data/erp.db`; default reports initialize only if the `saved_reports` table exists.
- Windows CRLF warnings from Git are harmless.

## License

For educational and demonstration purposes.
