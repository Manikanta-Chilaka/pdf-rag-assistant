from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv("NVIDIA_API_KEY")
)

def embed_chunks(chunks):
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            input = chunk,
            model = "nvidia/nemotron-3-embed-1b",
            extra_body = {
                "input_type" : "passage"
            }
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
        # Add the embedding to the embeddings list
    return embeddings



def generate_query_embedding(question):
    response = client.embeddings.create(
        input = question,
        model = "nvidia/nemotron-3-embed-1b",
        extra_body = {
            "input_type":"query"
        }
    )
    return response.data[0].embedding
