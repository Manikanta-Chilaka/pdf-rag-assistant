import json
import math
import chromadb

client = chromadb.PersistentClient(
    path="storage/chroma"
)

collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

def search_similar_chunks(query_embedding, top_k):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results = top_k,
        include=["documents", "distances", "metadatas"]
    )

    retrieved_chunks = []
    
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distances": results["distances"][0][i],
            "metadatas": results["metadatas"][0][i]
        })
    print("Retrieved chunks:", results["documents"])
    print("Retrieved metadata:", results["metadatas"])
    return retrieved_chunks



def load_embeddings(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have same dimension")
    dot_product = sum(a*b for a,b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("The question is empty or invalid")
    return dot_product/(magnitude_a*magnitude_b)



def build_context(results):
    chunk_text = []
    for result in results:
        chunk_text.append(result["text"])
    context = "\n\n".join(chunk_text)
    return context



if __name__ == "__main__":
    from embeddings import generate_query_embedding
    question = "What is Python?"
    query_embedding = generate_query_embedding(question)
    results = search_similar_chunks(query_embedding, top_k=3)
    for result in results:
        print("\nID:", result["id"])
        print("Text:", result["text"])
        print("Distance:", result["distances"])
        print("Metadata:", result["metadatas"])