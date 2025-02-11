from query_data import query_rag                               # Don't delete, to use all utils with the same import
from get_embedding_function import get_embedding_function      # Don't delete, to use all utils with the same import
from langchain_ollama import OllamaLLM as Ollama
import sys
from populate_database import main as populate_db, get_all_chunk_embeddings


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


def populate_database(reset: bool = True, model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                      model_type: str = "sentence_transformer") -> str:

    sys.argv = ["populate_database.py"]
    if reset:
        sys.argv.append("--reset")
    if model_name:
        sys.argv.extend(["--model-type", model_type, "--model-name", model_name])

    print("I am using this embedding in utils:", model_name)
    populate_db()
    return "Database populated successfully!"
