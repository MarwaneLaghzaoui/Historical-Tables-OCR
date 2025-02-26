from ready_codes.pdf_extraction import extract_tables_from_pdf
from llm_table_correction import chat_with_ollama

if __name__ == "__main__":

    pdf_path = r"D://GitHub//HOCR//test_pages//test_page_mortstatsh_1905-207.pdf"
    csv_path = r"D://GitHub//HOCR//table_extraction//output.csv"
    # Extraction des tableaux du PDF
    extract_tables_from_pdf(pdf_path, csv_path)
    
    # Correction avec Ollama
    chat_with_ollama(csv_path)
