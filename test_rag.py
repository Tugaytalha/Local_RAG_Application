import os
import shutil
import subprocess
from docx import Document
from query_data import query_rag, PROMPT_TEMPLATE
from populate_database import clear_database, calculate_chunk_ids
from langchain_community.llms.ollama import Ollama
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

TEST_QUERIES = {
    "müşteriler bankamız ATMleri harici hangi ATMleri kullanabilir?": "PTT",
    "Hareketsiz hesap nedir?": "Müşterimizin 1 yılı aşkın süre zarfında hesabına ilişkin para çıkışı yapmaması durumunda hesap hareketsiz konuma alınmaktadır. Ek9",
    "Debit Kart kesin olarak kapatılmak istendiğinde Çağrı Merkezi personeli nasıl bir yol izlemelidir? > Hangi durumlarda Debit Kart geçici olarak kapatılır?": "İstenen kartı kesin olarak kapatsa da derhal yenisinin başvurusunu almalıdır. > Ek8",
    "müşterinin hangi durumlarda para transferi işlemini Çağrı Merkezinden yapması mümkündür?": "Son 20 işlem veya kayıtlı işlemleri Çağrı Merkezinden yapabilir. Güvenlik gereği başka para transferi işlemlerini yapamaz.",
    "kayıp çalışntı durumunda kapatılan karta bağlı HGS talimatı yeni verilen karta otomatik devrolur mu?": "Evet",
    "ATM'lerde yapılan işlemlerde hangi koşullarda müşteriden komisyon alınır?": "Bankamız ATM'lerinde bankamız kartı ile yapılan hiçbir işlemde komisyon alınmaz, başka banka ATMlerinden işlem yapılması halinde komisyon alınır. Ek7",
    "Kredi kartım suya düşse ne olur?": "Buna cevap veremiyorum.",
}

EMBEDDING_MODELS = [
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "emrecan/convbert-base-turkish-mc4-cased-allnli_tr",
    "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
    "atasoglu/distilbert-base-turkish-cased-nli-stsb-tr",
    "atasoglu/xlm-roberta-base-nli-stsb-tr",
    "atasoglu/mbert-base-cased-nli-stsb-tr",
    "Omerhan/intfloat-fine-tuned-14376-v4",
    "atasoglu/turkish-base-bert-uncased-mean-nli-stsb-tr",
    "jinaai/jina-embeddings-v3",
]


def test_rag_with_embeddings(embedding_model_name, test_queries, document):
    """
    Tests the RAG application with a specific embedding model.
    """

    print(f"Testing with embedding: {embedding_model_name}")
    document.add_heading(f"Embedding: {embedding_model_name}", level=1)

    # Reset and populate the database
    clear_database()

    # Initialize embedding function with appropriate settings
    command = ["python", "populate_database.py", "--reset", "--model-type", "sentence-transformer", "--model-name",
               embedding_model_name]
    subprocess.run(command, check=True)

    # Test with each query
    for query, expected_response in test_queries.items():
        try:
            response, retrieved_chunks = query_rag(query, get_embedding_function(embedding_model_name, True))
        except Exception as e:
            print(f"Error during query processing: {e}")
            response = "Error during query processing"
            retrieved_chunks = []

        evaluation_result = evaluate_response(response, expected_response)

        # Add results to the document
        document.add_paragraph(f"Query: {query}")
        document.add_paragraph(f"Expected Response: {expected_response}")
        document.add_paragraph(f"Actual Response: {response}")
        document.add_paragraph(f"Evaluation: {evaluation_result}")

        # Add retrieved chunks to the document
        document.add_paragraph("Retrieved Chunks:")
        for chunk in retrieved_chunks:
            document.add_paragraph(
                f" - Source: {chunk['source']}, Content: {chunk['content']}, Score: {chunk['score']}")

        document.add_paragraph("---")


def evaluate_response(actual_response, expected_response):
    """
    Evaluates the actual response against the expected response using an LLM.
    """
    model = Ollama(model="mistral")  # You can change the evaluation model if needed
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=actual_response
    )
    evaluation_result = model.invoke(prompt)
    return evaluation_result.strip()


def main():
    """
    Main function to run the RAG tests with multiple embeddings.
    """
    # Create a new DOCX document for the report
    document = Document()
    document.add_heading("RAG Application Test Report", 0)

    # Run tests for each embedding model
    for embedding_model_name in EMBEDDING_MODELS:
        test_rag_with_embeddings(embedding_model_name, TEST_QUERIES, document)

    # Save the document
    document.save("rag_test_report.docx")
    print("RAG test report generated: rag_test_report.docx")


if __name__ == "__main__":
    main()