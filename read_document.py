from pathlib import Path
from pypdf import PdfReader


def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found {file_path}")

    if not path.is_file():
        raise IsADirectoryError(f"Excepted a file got a directory: {file_path}")


    if path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        text = path.read_text(encoding = "utf-8")

    if not text.strip():
        raise ValueError("Document is empty")
    return text

def split_into_chunks(text, chunk_size, overlap):
    if chunk_size < 1:
         raise ValueError("Value cannot be negative or Zero")

    if overlap < 0:
         raise ValueError("Value of overlap cannot be less than 0")

    if overlap >= chunk_size:
         raise ValueError("Both the values overlap and chunk_size cannot be same and overlap cannot be greater than chunksize")

    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size-overlap):
            chunks.append(words[i:i+chunk_size])
            if i+chunk_size>=len(words):
                break
    return chunks

def process_document(file_path, chunk_size, overlap):
    text = load_document(file_path)
    chunks = split_into_chunks(text, chunk_size, overlap)
    return chunks