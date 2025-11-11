# Project Guide

This project has two parts: a Python **backend** (FastAPI) that runs the RAG logic, and a React **frontend** (Vite) that shows the chat experience in the browser.

## Backend (FastAPI)

- Location: `app/` and `scripts/` folders.
- Key files:
  - `app/main.py` – starts the API server and exposes the `/ask` endpoint.
  - `app/rag/pipeline.py` – loads documents, builds embeddings, and answers questions with citations.
  - `app/settings.py` – reads configuration from `config/.env` (OpenAI key, data paths, etc.).
  - `scripts/ingest.py` – command-line script that prepares the Chroma vector store (run before using the chat).

### Backend Setup & Run

1. Create/activate your Python environment (example: `conda activate langchain`).
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy the sample env and fill in real values:

   ```powershell
   Copy-Item config/.env.example config/.env
   # edit config/.env so OPENAI_API_KEY=sk-...
   ```

4. Ingest docs (must be in repo root when running):

   ```powershell
   $env:PYTHONPATH = (Resolve-Path .)
   python -m scripts.ingest
   ```

5. Start the API:

   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

### Backend Common Fixes

- **`ModuleNotFoundError: app.deps`** → run commands from the repo root and set `PYTHONPATH` as above.
- **`OPENAI_API_KEY missing`** → confirm `config/.env` exists and the key is spelled exactly `OPENAI_API_KEY`.
- **Slow startup** → first run downloads models (BAAI/bge-m3, reranker, etc.); allow time or pre-download.

### Customising models & parameters

1. Edit `config/.env` to switch models or tuning knobs. Useful keys:
   - `EMBEDDING_MODEL` – HuggingFace embedding checkpoint (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
   - `RERANK_MODEL` – cross-encoder used for reranking (e.g. `BAAI/bge-reranker-large`).
   - `CHAT_MODEL` – OpenAI chat completion model (e.g. `gpt-4o`, `gpt-4o-mini`).
   - `RETRIEVER_K`, `RERANK_TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TORCH_DEVICE`.
2. If you change embeddings or chunking, delete the existing index so it rebuilds with the new settings:

   ```powershell
   Remove-Item chroma_db -Recurse -Force
   ```

3. Re-run ingestion to regenerate embeddings:

   ```powershell
   $env:PYTHONPATH = (Resolve-Path .)
   python -m scripts.ingest
   ```

4. Restart the FastAPI server (`uvicorn app.main:app --reload --port 8000`) so it loads the new configuration.
5. Refresh the frontend and retest queries. Watch the backend logs for download progress or errors when new checkpoints load.

## Frontend (Vite + React)

- Location: `frontend/` folder.
- Key files:
  - `frontend/src/App.tsx` – chat UI, submits questions, displays answers & citations, loads PDF preview.
  - `frontend/src/client.ts` – fetch helper that calls the backend `/ask` endpoint.
  - `frontend/src/styles.css` – layout and theming.
  - `frontend/.env` – optional; set `VITE_API_BASE=http://localhost:8000` if backend runs elsewhere.

### Frontend Setup & Run

1. Install node packages (run inside `frontend/`):

   ```powershell
   npm install
   ```

2. Start the dev server:

   ```powershell
   npm run dev
   ```

3. Open the printed URL (usually `http://localhost:5173`) in your browser. Ensure the backend is running so API calls succeed.

### Frontend Troubleshooting

- **`NetworkError when attempting to fetch resource`** → backend is not running or the URL in `VITE_API_BASE` is wrong.
- **Red underline for imports in VS Code** → install dependencies (`npm install`) so TypeScript type declarations are available.

## Testing the Flow

1. Run ingestion and the backend (`uvicorn`).
2. Start the frontend (`npm run dev`).
3. Visit the frontend URL, ask a question, and click a citation to view the highlighted PDF snippet.
