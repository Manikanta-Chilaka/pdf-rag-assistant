from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch
import httpx
from openai import RateLimitError,APIConnectionError

client = TestClient(app)

def test_home():
    response = client.get('/')

    assert response.status_code == 200
    assert response.json() == {
        "message":"Document RAG Assistant is running"
    }

@patch("app.ask_question")
def test_ask_question(mock_ask_question):
    mock_ask_question.return_value = "Python is used for artificial intelligence"                                       

    response = client.post(
        "/ask",
        json = {
            "question":"What is python used for?"
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer":"Python is used for artificial intelligence"
    }

    mock_ask_question.assert_called_once_with(
        "What is python used for?"
    )


@patch("app.ask_question")
def test_missing_embeddings_file(mock_ask_question):
    mock_ask_question.side_effect = FileNotFoundError

    response = client.post(
        "/ask",
        json={"question": "What is Python?"}
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Required embedding storage file was not found."
    }

@patch("app.ask_question")
def test_rate_limit_error(mock_ask_question):

    request = httpx.Request(
        "POST",
        "https://api.example.com"
    )

    response = httpx.Response(
        429,
        request=request
    )

    mock_ask_question.side_effect = RateLimitError(
        "Rate limit exceeded",
        response=response,
        body={"error": "rate limit"}
    )

    result = client.post(
        "/ask",
        json={"question": "What is Python?"}
    )

    assert result.status_code == 429

    assert result.json() == {
        "detail": "API rate limit exceeded. Please try again later."
    }


@patch("app.ask_question")
def test_api_connection_error(mock_ask_question):

    mock_ask_question.side_effect = APIConnectionError(
        request=httpx.Request(
            "POST",
            "https://api.example.com"
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is Python?"}
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Unable to connect to the AI service."
    }

def test_empty_question():

    response = client.post(
        "/ask",
        json={"question": "    "}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Value error, Question cannot be empty"
    