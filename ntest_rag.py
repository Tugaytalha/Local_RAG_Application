from docx import Document
import umap
import numpy as np
import matplotlib.pyplot as plt
from run_utils import populate_database, evaluate_response, query_rag, get_embedding_function, get_all_chunk_embeddings

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


def visualize_queries(queries, query_embeddings, all_chunk_embeddings, retrieved_embeddings_list, expected_embeddings_list, model_name):
    """
    Creates subplots for each query, visualizing its relationship with all chunks, retrieved chunks, and expected chunks.
    """
    """
    Creates subplots for each query, visualizing its relationship with all chunks, retrieved chunks, and expected chunks.
    """
    num_queries = len(queries)
    cols = 2  # Number of columns for subplots
    rows = (num_queries + 1) // cols  # Compute required rows dynamically

    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    all_embeddings = np.vstack((query_embeddings, all_chunk_embeddings))
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    # Extract 2D positions
    query_positions = reduced_embeddings[:len(query_embeddings)]
    all_chunk_positions = reduced_embeddings[len(query_embeddings):]

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))  # Dynamic plot size
    axes = axes.flatten()

    for i, (query, query_embedding) in enumerate(zip(queries, query_positions)):
        ax = axes[i]

        # Plot all chunks in grey
        ax.scatter(all_chunk_positions[:, 0], all_chunk_positions[:, 1], c='grey', label='All Chunks', alpha=0.3)

        # Plot query in red
        ax.scatter(query_embedding[0], query_embedding[1], c='red', label='Query', s=100, edgecolor='black')

        # Plot retrieved chunks in yellow
        retrived_to_pilot = reducer.transform(retrieved_embeddings_list[i])
        ax.scatter(retrived_to_pilot[:, 0], retrived_to_pilot[:, 1], c='yellow', label='Retrieved Chunks', s=70,
                   edgecolor='black')

        # Plot expected chunks in green
        if len(expected_embeddings_list[i]) != 0:
            expected_to_pilot = reducer.transform(expected_embeddings_list[i])
            ax.scatter(expected_to_pilot[:, 0], expected_to_pilot[:, 1], c='green', label='Expected Chunks', s=70,
                       edgecolor='black')

        # Decide expected chunks retrieved or not
        expectation_list = [exp in retrieved_embeddings_list[i] for exp in expected_embeddings_list[i]]
        ax.set_title(f"Query {i + 1}: {query[:40]}{'...' if len(query) > 40 else ''}, R1:{any(expectation_list)}, RA:{all(expectation_list)}")  # Show first 40 chars of the query
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig((f"query_chunks_visualization_{model_name}.png").replace(" ", "_").replace("/", "_"))
    plt.close()


def try_rag_with_embeddings(embedding_model_name):
    """
    Tests the RAG application with a specific embedding model.
    """
    # Create a Word document to store the test results
    document = Document()
    document.add_heading("RAG Application Test Report", 0)

    print(f"Testing with embedding: {embedding_model_name}")
    document.add_heading(f"Embedding: {embedding_model_name}", level=1)

    # Populate the database with the specified embedding model
    populate_database(reset=True, model_name=embedding_model_name, model_type="sentence_transformer")

    # Initialize embedding function with appropriate settings
    embedding = get_embedding_function(
        model_name_or_path=embedding_model_name,
        model_type="sentence_transformer"
    )

    # Get all chunk embeddings from the database
    all_chunk_data = get_all_chunk_embeddings()
    all_chunk_embeddings = np.array(all_chunk_data["embeddings"])

    query_embeddings = []
    retrieved_embeddings_list = []
    expected_embeddings_list = []

    # Test with each query
    for query, expected_response_chunk in QUERIES.items():
        query_embedding = np.array(embedding.embed_query(query)).reshape(1, -1)
        query_embeddings.append(query_embedding)

        try:
            response, retrieved_chunks = query_rag(query, embedding)
        except Exception as e:
            print(f"Error during query processing: {e}")
            response = "Error during query processing"
            retrieved_chunks = []

        retrieved_embeddings = np.array([embedding.embed_query(chunk['content']) for chunk in retrieved_chunks])
        retrieved_embeddings_list.append(retrieved_embeddings)

        expected_embeddings = []
        for source in expected_response_chunk[1]:
            for chunk in all_chunk_data["ids"]:
                if chunk == source:
                    expected_embeddings.append(all_chunk_data["embeddings"][all_chunk_data["ids"].index(chunk)])
        expected_embeddings = np.array(expected_embeddings)
        expected_embeddings_list.append(expected_embeddings)

        evaluation_result = evaluate_response(response, expected_response_chunk[0])

        document.add_paragraph(f"Query: {query}")
        document.add_paragraph(f"Expected Response: {expected_response_chunk[0]}")
        document.add_paragraph(f"Actual Response: {response}")
        document.add_paragraph(f"Evaluation: {evaluation_result}")
        document.add_paragraph("---")

    visualize_queries(QUERIES.keys(), np.vstack(query_embeddings), all_chunk_embeddings, retrieved_embeddings_list,
                      expected_embeddings_list, embedding_model_name)

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