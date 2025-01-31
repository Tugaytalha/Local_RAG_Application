from docx import Document
import umap
import numpy as np
from get_embedding_function import get_embedding_function
from run_utils import populate_database, evaluate_response

QUERIES = {
    "müşteriler bankamız ATMleri harici hangi ATMleri kullanabilir?":
        ["PTT", ["data\Çağrı merkezi chatbot için bilgiler v2.docx:None:44"]],
    "Hareketsiz hesap nedir?":
        [
            "Müşterimizin 1 yılı aşkın süre zarfında hesabına ilişkin para çıkışı yapmaması durumunda hesap hareketsiz konuma alınmaktadır. Ek9",
            ["data\Çağrı merkezi chatbot için bilgiler v2.docx:None:49"]],
    "Debit Kart kesin olarak kapatılmak istendiğinde Çağrı Merkezi personeli nasıl bir yol izlemelidir? > Hangi durumlarda Debit Kart geçici olarak kapatılır?":
        ["İstenen kartı kesin olarak kapatsa da derhal yenisinin başvurusunu almalıdır. > Ek8",
         ["data\Çağrı merkezi chatbot için bilgiler v2.docx:None:31",
          "data\Çağrı merkezi chatbot için bilgiler v2.docx:None:9"]],
    # "müşterinin hangi durumlarda para transferi işlemini Çağrı Merkezinden yapması mümkündür?": [
    #     "Son 20 işlem veya kayıtlı işlemleri Çağrı Merkezinden yapabilir. Güvenlik gereği başka para transferi işlemlerini yapamaz.",
    #     [""]],
    "kayıp çalışntı durumunda kapatılan karta bağlı HGS talimatı yeni verilen karta otomatik devrolur mu?": [
        "Evet",
        ["data\Çağrı merkezi chatbot için bilgiler v2.docx:None:11"]],
    "ATM'lerde yapılan işlemlerde hangi koşullarda müşteriden komisyon alınır?": [
        "Bankamız ATM'lerinde bankamız kartı ile yapılan hiçbir işlemde komisyon alınmaz, başka banka ATMlerinden işlem yapılması halinde komisyon alınır. Ek7",
        ["data\Çağrı merkezi chatbot için bilgiler v2.docx:None:45", "data\Çağrı merkezi chatbot için bilgiler v2.docx:None:46"]],
    "Kredi kartım suya düşse ne olur?": ["Buna cevap veremiyorum.", []],
}

EMBEDDING_MODELS = [
    #####"emrecan/convbert-base-turkish-mc4-cased-allnli_tr",
    ###"emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ###"atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
    ###"atasoglu/distilbert-base-turkish-cased-nli-stsb-tr",
    "atasoglu/xlm-roberta-base-nli-stsb-tr",
    "atasoglu/mbert-base-cased-nli-stsb-tr",
    "Omerhan/intfloat-fine-tuned-14376-v4",
    ###"atasoglu/turkish-base-bert-uncased-mean-nli-stsb-tr",
    "jinaai/jina-embeddings-v3",
]

import matplotlib.pyplot as plt


def visualize_with_umap(query, all_chunks, retrieved_chunks, expected_chunks, embedding_model_name):
    """
    Visualizes the query, all chunks, retrieved chunks, and expected chunks using UMAP.
    """
    # Get the embedding function
    embedding = get_embedding_function(embedding_model_name, "sentence_transformer")

    # Embed the query, all chunks, retrieved chunks, and expected chunks
    query_embedding = embedding([query])
    all_chunks_embeddings = embedding([chunk['content'] for chunk in all_chunks])
    retrieved_chunks_embeddings = embedding([chunk['content'] for chunk in retrieved_chunks])
    expected_chunks_embeddings = embedding([chunk['content'] for chunk in expected_chunks])

    # Combine all embeddings for UMAP
    combined_embeddings = np.vstack([
        query_embedding,
        all_chunks_embeddings,
        retrieved_chunks_embeddings,
        expected_chunks_embeddings
    ])

    # Apply UMAP to reduce dimensionality to 2D
    reducer = umap.UMAP(random_state=42)
    reduced_embeddings = reducer.fit_transform(combined_embeddings)

    # Separate the reduced embeddings
    query_point = reduced_embeddings[0]  # Query is the first point
    all_chunks_points = reduced_embeddings[1:1 + len(all_chunks_embeddings)]
    retrieved_chunks_points = reduced_embeddings[1 + len(all_chunks_embeddings):1 + len(all_chunks_embeddings) + len(
        retrieved_chunks_embeddings)]
    expected_chunks_points = reduced_embeddings[1 + len(all_chunks_embeddings) + len(retrieved_chunks_embeddings):]

    # Plot the points
    plt.figure(figsize=(10, 8))
    plt.scatter(all_chunks_points[:, 0], all_chunks_points[:, 1], c='grey', label='All Chunks', alpha=0.6)
    plt.scatter(retrieved_chunks_points[:, 0], retrieved_chunks_points[:, 1], c='yellow', label='Retrieved Chunks',
                edgecolor='black')
    plt.scatter(expected_chunks_points[:, 0], expected_chunks_points[:, 1], c='green', label='Expected Chunks',
                edgecolor='black')
    plt.scatter(query_point[0], query_point[1], c='red', label='Query', s=100, edgecolor='black')

    # Add labels and legend
    plt.title(f"UMAP Visualization for Embedding Model: {embedding_model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def try_rag_with_embeddings(embedding_model_name):
    """
    Tests the RAG application with a specific embedding model and visualizes the results with UMAP.
    """
    # Create a new DOCX document for the report
    document = Document()
    document.add_heading("RAG Application Test Report", 0)
    print(f"Testing with embedding: {embedding_model_name}")
    document.add_heading(f"Embedding: {embedding_model_name}", level=1)
    # Populate the database
    populate_database(reset=True, model_name=embedding_model_name, model_type="sentence_transformer")
    # Get the embedding function
    embedding = get_embedding_function(embedding_model_name, "sentence_transformer")
    all_sources = []
    # Test with each query
    for query, expected_response_chunk in QUERIES.items():
        try:
            from run_utils import query_rag, get_all_chunks
            response, retrieved_chunks = query_rag(query, embedding)
            all_chunks = get_all_chunks()  # Assume this function retrieves all chunks from the database
        except Exception as e:
            print(f"Error during query processing: {e}")
            response = "Error during query processing"
            retrieved_chunks = []
            all_chunks = []
        evaluation_result = evaluate_response(response, expected_response_chunk[0])
        # Take sources
        sources = [chunk['source'] for chunk in retrieved_chunks]
        # Add sources to the list
        all_sources.append(sources)
        # Add results to the document
        document.add_paragraph(f"Query: {query}")
        document.add_paragraph(f"Expected Response: {expected_response_chunk[0]}")
        document.add_paragraph(f"Actual Response: {response}")
        document.add_paragraph(f"Evaluation: {evaluation_result}")
        document.add_paragraph(f"Sources: {sources}")
        # Add retrieved chunks to the document
        document.add_paragraph("Retrieved Chunks:")
        for chunk in retrieved_chunks:
            document.add_paragraph(
                f" - Source: {chunk['source']}, Content: {chunk['content']}, Score: {chunk['score']}")
        document.add_paragraph("---")

        # Visualize with UMAP
        expected_chunks = [chunk for chunk in all_chunks if chunk['source'] in expected_response_chunk[1]]
        visualize_with_umap(query, all_chunks, retrieved_chunks, expected_chunks, embedding_model_name)

    # Add sources to the end of the document
    document.add_heading("Sources", level=1)
    for sources in all_sources:
        document.add_paragraph(f"Sources: {sources}")
    # Save the document
    document.save((f"rag_test_report_{embedding_model_name}.docx").replace("/", "_"))
    print(f"RAG test report generated: rag_test_report_{embedding_model_name}.docx")


def main():
    """
    Runs the RAG tests for all embedding models.
    """
    for embedding_model_name in EMBEDDING_MODELS:
        try_rag_with_embeddings(embedding_model_name)


if __name__ == "__main__":
    main()