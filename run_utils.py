from query_data import QueryData                               # Don't delete, to use all utils with the same import
from get_embedding_function import get_embedding_function      # Don't delete, to use all utils with the same import
from langchain_ollama import OllamaLLM as Ollama
import sys
from populate_database import _main as populate_db, get_all_chunk_embeddings
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import umap
import matplotlib
import os
import time
matplotlib.use('Agg')  # Use non-interactive backend


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


def visualize_query_embeddings(query, query_embedding, all_chunk_embeddings, retrieved_embeddings):
    """
    Creates a visualization of a query and its relationship with document chunks.
    
    Args:
        query (str): The query text
        query_embedding (numpy.ndarray): The embedding vector of the query
        all_chunk_embeddings (numpy.ndarray): Embeddings of all document chunks
        retrieved_embeddings (numpy.ndarray): Embeddings of retrieved chunks
        
    Returns:
        str: Path to the saved visualization image
    """
    # Reshape query embedding if needed
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    
    # Stack query and all chunk embeddings for dimensionality reduction
    all_embeddings = np.vstack((query_embedding, all_chunk_embeddings))
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Extract 2D positions
    query_position = reduced_embeddings[0]
    all_chunk_positions = reduced_embeddings[1:]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot all chunks in grey
    plt.scatter(all_chunk_positions[:, 0], all_chunk_positions[:, 1], 
                c='grey', label='All Chunks', alpha=0.3)
    
    # Plot query in red
    plt.scatter(query_position[0], query_position[1], 
                c='red', label='Query', s=100, edgecolor='black')
    
    # Plot retrieved chunks in yellow
    if len(retrieved_embeddings) > 0:
        retrieved_positions = reducer.transform(retrieved_embeddings)
        plt.scatter(retrieved_positions[:, 0], retrieved_positions[:, 1], 
                    c='yellow', label='Retrieved Chunks', s=70, edgecolor='black')
    
    # Add title and legend
    plt.title(f"Query: {query[:40]}{'...' if len(query) > 40 else ''}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Save to a file with a timestamp to avoid conflicts
    timestamp = int(time.time())
    output_path = f"visualizations/query_viz_{timestamp}.png"
    plt.savefig(output_path, format='png', dpi=100)
    plt.close()
    
    return output_path
