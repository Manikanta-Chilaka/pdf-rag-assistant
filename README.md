# PDF RAG Assistant

Ask questions about your own PDF and TXT documents. Upload a file, and the API answers questions using only the content of that file (Retrieval-Augmented Generation).

Built with **FastAPI**, **ChromaDB** and **NVIDIA AI models** (via the OpenAI-compatible API).

## How it works

```
Upload (PDF/TXT)
   -> read text (pypdf for PDFs)
   -> split into overlapping word chunks
   -> create embeddings (nvidia/nemotron-3-embed-1b)
   -> store in ChromaDB (cosine similarity)

Question
   -> embed the question
   -> find the 3 most similar chunks in ChromaDB
   -> send chunks + question to the LLM (nvidia/nemotron-3.5-bolt-e4b)
   -> return the answer
```

## Project structure

| File | What it does |
|------|--------------|
| `read_document.py` | Reads PDF/TXT files and splits text into chunks |
| `embeddings.py` | Turns text into embedding vectors (NVIDIA API) |
| `ingestion.py` | Saves chunks + embeddings into ChromaDB, lists and deletes documents |
| `retrieval.py` | Finds the most similar chunks for a question |
| `llm.py` | Generates the answer from the retrieved context |
| `rag.py` | Connects everything: question -> answer |
| `app.py` | FastAPI endpoints |
| `tests/` | Pytest tests (API, chunking, PDF loading) |

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload a `.pdf` or `.txt` file |
| GET | `/documents` | List uploaded documents and chunk counts |
| DELETE | `/documents/{filename}` | Delete a document and its chunks |
| POST | `/ask` | Ask a question: `{"question": "..."}` |

## Setup

```bash
git clone https://github.com/Manikanta-Chilaka/pdf-rag-assistant.git
cd pdf-rag-assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with your free API key from [build.nvidia.com](https://build.nvidia.com):

```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Run

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try the API in the browser:

1. `POST /upload` -> choose a PDF
2. `POST /ask` -> `{"question": "What is this document about?"}`

## Tests

```bash
pytest
```

Tests mock the AI calls, so they run without an API key.

## Tech stack

Python, FastAPI, ChromaDB, pypdf, NVIDIA NIM (OpenAI SDK), Pydantic, Pytest
