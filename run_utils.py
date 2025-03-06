from query_data import QueryData                               # Don't delete, to use all utils with the same import
from get_embedding_function import get_embedding_function      # Don't delete, to use all utils with the same import
from langchain_ollama import OllamaLLM as Ollama
import sys
from populate_database import _main as populate_db, get_all_chunk_embeddings


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def evaluate_response(actual_response, expected_response):
    """
    Evaluates the actual response against the expected response using an LLM.
    """
    model = Ollama(model="llama3.2:3b")
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=actual_response
    )
    evaluation_result = model.invoke(prompt)
    return evaluation_result.strip()


def populate_database(reset: bool = True, model_name: str = "jinaai/jina-embeddings-v3",
                      model_type: str = "sentence_transformer") -> str:
    """
    Populates the database with the given model.

    :param reset: reset the database
    :param model_name: Embedding model name to use in populating the database
    :param model_type: Embedding model type (sentence_transformer or ollama)
    :return: Success message
    """
    print("I am using this embedding in utils:", model_name)
    populate_db(reset=reset, model_name=model_name, model_type=model_type)
    return "Database populated successfully!"
