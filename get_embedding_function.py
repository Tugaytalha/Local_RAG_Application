from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer

def get_embedding_function(model_name_or_path="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", use_sentence_transformer=True):
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name: Name of the model on HuggingFace
        use_local: Whether to use local model
        local_path: Path to local model directory
    """
    if use_sentence_transformer:
        model = SentenceTransformer(model_name_or_path)
        # Create HuggingFaceEmbeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        embeddings = OllamaEmbeddings(model="bge-m3")
    
    
    return embeddings


