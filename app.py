from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from rag import ask_question
from fastapi import HTTPException, UploadFile, File
from openai import  APIConnectionError, RateLimitError, APIStatusError
from pathlib import Path
import shutil
from ingestion import ingest_document, get_all_documents, delete_document


class QuestionRequest(BaseModel):
    question:str = Field(min_length = 1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value

    
app = FastAPI(
    title = "Document RAG assistant is running",
    description = "A document based question answering API",
    version = "1.1.0"
)


@app.get("/")
def home():
    return {"message": "Document RAG Assistant is running"}

@app.post("/ask")
def ask(request:QuestionRequest):
    try:
        answer = ask_question(
            request.question
        )
        return {"answer":answer}
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded. Please try again later."
        )
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the AI service."
        )
    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an Error."
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Required embedding storage file was not found."
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to process your question please try again later!"
        )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
        status_code=400,
        detail="Filename is required"
    )

    if Path(file.filename).suffix.lower() not in [".txt", ".pdf"]:
        raise HTTPException(
        status_code=400,
        detail="Only TXT and PDF files are supported"
    )
    upload_dir = Path("documents")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / Path(file.filename).name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        file.file.close()
    try:
        text_chunks, embeddings = ingest_document(
        file_path=str(file_path)
        )
    except ValueError:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="No text could be read from this file"
        )
    return {
        "message": "Document uploaded and processed successfully",
        "filename": file_path.name,
        "total_chunks": len(text_chunks)
    }
@app.get("/documents")
def list_documents():
    documents = get_all_documents()

    return {
        "total_documents": len(documents),
        "documents": [
            {
                "filename": filename,
                "total_chunks": count
            }
            for filename, count in documents.items()
        ]
    }

@app.delete("/documents/{filename}")
def remove_document(filename:str):
    try:
        deleted = delete_document(filename)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid file path"
        )
    

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    return {
        "message": "Document deleted successfully",
        "filename": filename
    }