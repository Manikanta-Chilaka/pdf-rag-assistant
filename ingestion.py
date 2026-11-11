from pathlib import Path
import chromadb
from read_document import process_document
from embeddings import embed_chunks

client = chromadb.PersistentClient(
    path="storage/chroma"
)

collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

def ingest_document(file_path, chunk_size=200, over_lap=30):
    chunks = process_document(file_path, chunk_size, over_lap)

    text_chunks = [" ".join(chunk) for chunk in chunks]

    embeddings = embed_chunks(text_chunks)

    ids = [
        f"{Path(file_path).name}_{i}"
        for i in range(len(text_chunks))
    ]
    metadatas = [{
        "source":Path(file_path).name,
        "chunk_index":i
        }
        for i in range(len(text_chunks))
    ] 
    collection.upsert(
        ids = ids,
        documents = text_chunks,
        embeddings = embeddings,
        metadatas = metadatas
    )

    return text_chunks, embeddings


def get_all_documents():
    results = collection.get(
        include=["metadatas"]
    )
    documents = {}

    for metadata in results["metadatas"]:
        filename = metadata["source"]

        if filename not in documents:
            documents[filename] = 0

        documents[filename] += 1

    return documents

def delete_document(filename):
    documents_dir = Path("documents").resolve()
    file_path = (documents_dir / filename).resolve()

    if not file_path.is_relative_to(documents_dir):
        raise ValueError("Invalid file path")

    results = collection.get(
        where={"source": filename},
        include=["metadatas"]
    )

    ids = results["ids"]
    file_exists = file_path.exists()

    if ids:
        collection.delete(ids=ids)

    if file_exists:
        file_path.unlink()

    return bool(ids) or file_exists

if __name__ == "__main__":
    print("Collection name:", collection.name)
    print("Total records:", collection.count())