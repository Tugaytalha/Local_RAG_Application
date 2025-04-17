import gradio as gr
import pandas as pd
import os
import time
from pathlib import Path
import numpy as np

from run_utils import populate_database, QueryData, get_embedding_function, visualize_query_embeddings
from populate_database import get_all_chunk_embeddings
from shutil import copy2

# Configuration constants
EMBEDDING_MODELS = [
    "jinaai/jina-embeddings-v3",
    "Omerhan/intfloat-fine-tuned-14376-v4",
    "intfloat/multilingual-e5-large-instruct",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
    "atasoglu/distilbert-base-turkish-cased-nli-stsb-tr"
]

LLM_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "llama3.3",
    "llama3.2-vision",
    "gemma3"
]

QUERY_AUGMENTATION_OPTIONS = [
    "None",
    "query",
    "answer"
]

DATA_PATH = "data"


def process_query(
        question: str,
        embedding_model: str,
        llm_model: str,
        use_multi_query: bool,
        query_augmentation: str
) -> tuple[str, gr.Dataframe, str, str]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None, "❌ Database not found", None

    start_time = time.time()
    status_msg = "🔍 Processing query..."

    try:
        # Get the embedding function
        embedding_func = get_embedding_function(
            model_name_or_path=embedding_model,
            model_type="sentence_transformer"
        )

        # Set augmentation to None if "None" is selected
        actual_augmentation = None if query_augmentation == "None" else query_augmentation

        # Query the RAG model
        response, chunks = QueryData.query_rag(
            query_text=question,
            embedding_function=embedding_func,
            model=llm_model,
            augmentation=actual_augmentation,
            multi_query=use_multi_query
        )

        # Create a DataFrame for display
        df_data = [
            [chunk['source'], chunk['content'], chunk['score']]
            for chunk in chunks
        ]

        # Generate visualization
        visualization_path = None
        # try:
        # Get all chunk embeddings from the database
        all_chunk_data = get_all_chunk_embeddings()
        if any(all_chunk_data) and 'embeddings' in all_chunk_data:
            all_chunk_embeddings = np.array(all_chunk_data["embeddings"])

            # Get query embedding
            query_embedding = np.array(embedding_func.embed_query(question))

            # Get embeddings for retrieved chunks
            retrieved_embeddings = np.array([embedding_func.embed_query(chunk['content']) for chunk in chunks])

            # Create visualization
            visualization_path = visualize_query_embeddings(
                question,
                query_embedding,
                all_chunk_embeddings,
                retrieved_embeddings
            )
        # except Exception as viz_error:
        #     print(f"Visualization error (non-critical): {str(viz_error)}")

        # Calculate sources for display
        sources = ", ".join(set([chunk['source'] for chunk in chunks]))
        elapsed_time = time.time() - start_time
        status_msg = f"✅ Query processed in {elapsed_time:.2f} seconds | Sources: {sources}"

        return response, gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            value=df_data
        ), status_msg, visualization_path
    except Exception as e:
        return f"Error processing query: {str(e)}", None, f"❌ Error: {str(e)}", None


def handle_file_upload(files, reset_db):
    if not files:
        return "No files uploaded."

    # Create data directory if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)

    # Copy uploaded files to the data directory
    file_count = 0
    for file in files:
        try:
            filename = Path(file.name).name
            destination = os.path.join(DATA_PATH, filename)
            copy2(file.name, destination)
            file_count += 1
        except Exception as e:
            return f"Error copying file {file.name}: {str(e)}"

    return f"✅ Successfully uploaded {file_count} files to the data directory."


def populate_db_with_params(reset_db, embedding_model):
    try:
        result = populate_database(
            reset=reset_db,
            model_name=embedding_model,
            model_type="sentence_transformer"
        )
        return f"✅ {result}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create the Gradio interface with improved styling
with gr.Blocks(title="AlbaraKa Document Q&A System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 AlbaraKa Document Q&A System
        
        Upload documents, ask questions, and get AI-powered answers based on your document content.
        """
    )

    with gr.Tab("Query Documents"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="Müşterim hangi ATMlerden para çekebilir?",
                    lines=4
                )
            with gr.Column(scale=1):
                status_display = gr.Textbox(label="Status", interactive=False, lines=3)

        query_button = gr.Button("Submit Query", variant="primary", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Response")
                output = gr.Textbox(label="AI Response", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    llm_dropdown = gr.Dropdown(
                        choices=LLM_MODELS,
                        value=LLM_MODELS[2],
                        label="LLM Model",
                        info="Select the large language model for response generation"
                    )

                    embedding_dropdown = gr.Dropdown(
                        choices=EMBEDDING_MODELS,
                        value=EMBEDDING_MODELS[1],
                        label="Embedding Model",
                        info="Select the embedding model for semantic search (have to be same as vectorDB model)"
                    )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Advanced Options")
                    multi_query_checkbox = gr.Checkbox(
                        label="Use Multi-Query Generation",
                        value=False,
                        info="Generate multiple search queries to improve retrieval for complex questions"
                    )

                    query_augmentation_dropdown = gr.Dropdown(
                        choices=QUERY_AUGMENTATION_OPTIONS,
                        value=QUERY_AUGMENTATION_OPTIONS[0],
                        label="Query Augmentation",
                        info="How to augment the query for better search results"
                    )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Visualization")
                viz_output = gr.Image(
                    label="Query-Document Visualization",
                    show_download_button=True,
                    type="filepath"
                )
                gr.Markdown("### Retrieved Chunks")
                chunks_output = gr.Dataframe(
                    headers=['Source', 'Content', 'Relevance Score'],
                    label="Retrieved Document Chunks",
                    wrap=True,
                    column_widths=[10, 30, 5]
                )

        query_button.click(
            fn=process_query,
            inputs=[
                query_input,
                embedding_dropdown,
                llm_dropdown,
                multi_query_checkbox,
                query_augmentation_dropdown
            ],
            outputs=[output, chunks_output, status_display, viz_output]
        )

    with gr.Tab("Document Management"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    file_types=["pdf", "docx", "txt", "csv", "xlsx"],
                    file_count="multiple",
                    label="Upload Files"
                )
                upload_button = gr.Button("Upload Files", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Database Control")
                with gr.Group():
                    reset_checkbox = gr.Checkbox(
                        label="Reset Database",
                        value=False,
                        info="Check to completely reset the database before populating"
                    )
                    db_embedding_dropdown = gr.Dropdown(
                        choices=EMBEDDING_MODELS,
                        value=EMBEDDING_MODELS[1],
                        label="Embedding Model for Database",
                        info="Select the embedding model to use for the vector database"
                    )
                    populate_button = gr.Button("Populate Database", variant="primary")
                    status_output = gr.Textbox(label="Population Status", interactive=False)

        upload_button.click(
            fn=handle_file_upload,
            inputs=[file_upload, reset_checkbox],
            outputs=upload_status
        )

        populate_button.click(
            fn=populate_db_with_params,
            inputs=[reset_checkbox, db_embedding_dropdown],
            outputs=status_output
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
            ## About AlbaraKa Document Q&A System
            
            This system uses Retrieval-Augmented Generation (RAG) to provide accurate answers to your questions based on your documents.
            
            ### Features
            
            - **Document Upload**: Support for PDF, DOCX, TXT, CSV, and XLSX files
            - **Advanced Retrieval**: Use state-of-the-art embedding models for semantic search
            - **Multi-Query Generation**: Generate multiple search queries for complex questions
            - **Query Augmentation**: Enhance your queries for better search results
            - **Customizable LLM**: Choose different language models for response generation
            
            ### How It Works
            
            1. Upload your documents in the Document Management tab
            2. Populate the database with your chosen embedding model
            3. Ask questions in the Query Documents tab
            4. The system will retrieve relevant information and generate a response
            
            For best results, use specific questions and experiment with different embedding models and settings.
            """
        )

if __name__ == "__main__":
    demo.launch(share=True)
