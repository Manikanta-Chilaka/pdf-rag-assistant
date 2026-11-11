from embeddings import generate_query_embedding
from retrieval import search_similar_chunks, build_context
from llm import generate_answer


def ask_question(question):
    question_embedding = generate_query_embedding(question)
    similar_chunks = search_similar_chunks(question_embedding, 3)
    context = build_context(similar_chunks)
    answer = generate_answer(context, question)
    return answer

if __name__ == "__main__":
    question = "What is FastAPI used for?"

    answer = ask_question(question)

    print("Question:", question)
    print("Answer:", answer)