from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv("NVIDIA_API_KEY")
)

def generate_answer(context, question):
    start = time.time()
    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-bolt-e4b",
        messages = [
            {
                "role":"system",
                "content": "You are a helpful assistant. Answer using the provided context."
            },
            {
                "role":"user",
                "content":f"""
                    Context:{context}
                Question:{question}
            """
            }
        ],
        temperature=0.2,
        max_tokens=512,
        stream=False
    )
    print("API response time:", round(time.time() - start, 2), "seconds")
    return response.choices[0].message.content
