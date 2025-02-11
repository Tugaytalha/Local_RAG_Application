from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings


def get_embedding_function(model_name_or_path="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                           model_type="sentence_transformer"):  # "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name_or_path (str): Model name or local path
        model_type (str): Model type (sentence_transformer or ollama)

        Returns:
        embeddings: Embedding function
    """
    if model_name_or_path is None:
        model_name_or_path = "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"
        print("Model name is not provided.")

    print(f"Using model: {model_name_or_path}")
    if model_type == "sentence_transformer":
        import torch
        # Check if the cuda is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", device)

        # Create HuggingFaceEmbeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True, 'device': device}
        )
    elif model_type == "ollama":
        embeddings = OllamaEmbeddings(model="bge-m3")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True}
        )
        print("Model type is uncertain. Using HuggingFaceEmbeddings model.: ", model_type)

    return embeddings
