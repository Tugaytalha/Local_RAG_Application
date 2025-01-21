import argparse
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")    
    parser.add_argument("--model-type", type=str, help="Specify If model type is sentence_transformer")
    parser.add_argument("--model-name", type=str, 
                       default="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                       help="HuggingFace or Ollama model name or local path")
    args = parser.parse_args()

    _main(reset=args.reset, model_name=args.model_name, model_type=args.model_type)


def _main(reset, model_name, model_type):
    if reset:
        print("✨ Clearing Database")
        clear_database()

    print("I am using this embedding in populate:", model_name)

    # Initialize embedding function with appropriate settings
    embedding_function = get_embedding_function(
        model_name_or_path=model_name,
        model_type=model_type
    )

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks=chunks, embedding_func=embedding_function)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], embedding_func=get_embedding_function()):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_func
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


# def clear_database():
#     print("🗑️ Clearing the database")
#     global CHROMA_PATH
#     while os.path.exists(CHROMA_PATH):
#         print("🔥 Removing")
#         # Remove the database directory forcefully.
#         CHROMA_PATH += "1"
#         print("✅ Database cleared")
#     print("✅ Database cleared")

def clear_database():
    print("🗑️ Clearing the database")
    global CHROMA_PATH
    # Remove the database with db.delete_collection()
    Chroma(persist_directory=CHROMA_PATH).delete_collection()
    print("✅ Database cleared")

# def clear_database():
#     print("🗑️ Clearing the database")
#     # Delete database directory
#     import shutil
#     shutil.rmtree(CHROMA_PATH)
#     print("✅ Database cleared")

if __name__ == "__main__":
    main()
