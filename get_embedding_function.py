from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    # Use the local nomic-embed-text-v1.5 model
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v1.5", run_local=True)
    return embeddings
