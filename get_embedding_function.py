from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer


def get_embedding_function(model_name_or_path="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                           model_type="sentence-transformer"):  # "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name_or_path (str): Model name or local path
        model_type (str): Model type (sentence-transformer or ollama)

        Returns:
        embeddings: Embedding function
    """
    if model_name_or_path is None:
        model_name_or_path = "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"

    if model_type == "sentence-transformer":
        # Create HuggingFaceEmbeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True}
        )
    else:
        embeddings = OllamaEmbeddings(model="bge-m3")

    return embeddings
