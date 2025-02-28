import gradio as gr
from run_utils import populate_database, query_rag, get_embedding_function
import os

def process_query(question: str, embedding_function: str="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr") -> tuple[str, gr.Dataframe]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None

    try:
        # Get the embedding function
        embedding_func = get_embedding_function(model_name_or_path="jinaai/jina-embeddings-v3")

        # Query the RAG model
        response, chunks = query_rag(question, embedding_func)

        # Create a DataFrame for display
        df_data = [
            [chunk['source'], chunk['content'], chunk['score']]
            for chunk in chunks
        ]

        return response, gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            value=df_data
        )
    except Exception as e:
        return f"Error processing query: {str(e)}", None


# Create the Gradio interface
with gr.Blocks(title="AlbaraKa Document Q&A System (Call Center)") as demo:
    gr.Markdown("# Document Question & Answer System")

    with gr.Tab("Query Documents"):
        query_input = gr.Textbox(
            label="Enter your question",
            placeholder="Müşterim hangi ATMlerden para çekebilir?"
        )
        query_button = gr.Button("Submit Query")
        output = gr.Textbox(label="Response")
        chunks_output = gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            label="Retrieved Chunks",
            wrap=True
        )
        query_button.click(
            fn=process_query,
            inputs=query_input,
            outputs=[output, chunks_output]
        )

    with gr.Tab("Database Management"):
        gr.Markdown("Populate or reset the document database")
        reset_checkbox = gr.Checkbox(label="Reset Database", value=False)
        populate_button = gr.Button("Populate Database")
        status_output = gr.Textbox(label="Status")
        populate_button.click(
            fn=populate_database,
            inputs=reset_checkbox,
            outputs=status_output
        )

if __name__ == "__main__":
    demo.launch(share=True)