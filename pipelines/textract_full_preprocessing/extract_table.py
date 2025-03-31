import boto3
import csv
import os
import pypdf

def extract_tables_from_pdf(pdf_path, output_csv_path):
    """Extrait les tableaux d'un PDF page par page en utilisant AWS Textract et les enregistre dans un CSV."""
    
    session = boto3.Session(profile_name="cytech")
    textract = session.client("textract")
    
    # Charger le PDF et extraire chaque page
    pdf_reader = pypdf.PdfReader(pdf_path)
    num_pages = len(pdf_reader.pages)
    
    print(f"Le PDF contient {num_pages} pages. Extraction en cours...")
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="|")
        
        for page_num in range(num_pages):
            print(f"Extraction de la page {page_num + 1}...")
            
            # Créer un fichier PDF temporaire contenant une seule page
            pdf_writer = pypdf.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_num])
            temp_pdf_path = f"temp_page_{page_num + 1}.pdf"
            
            with open(temp_pdf_path, "wb") as temp_pdf:
                pdf_writer.write(temp_pdf)

            # Lire le fichier PDF temporaire
            with open(temp_pdf_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()

            # Envoyer à Textract
            response = textract.analyze_document(
                Document={'Bytes': pdf_bytes},
                FeatureTypes=['TABLES']
            )

            os.remove(temp_pdf_path)  # Supprimer le fichier temporaire

            # Extraire les tableaux
            tables = []
            for block in response['Blocks']:
                if block['BlockType'] == 'TABLE':
                    table = []
                    for relationship in block.get('Relationships', []):
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                cell = next((b for b in response['Blocks'] if b['Id'] == child_id), None)
                                if cell and cell['BlockType'] == 'CELL':
                                    row_index = cell['RowIndex'] - 1
                                    col_index = cell['ColumnIndex'] - 1
                                    text = ""

                                    # Vérifier si la cellule contient des mots
                                    if 'Relationships' in cell:
                                        for relation in cell['Relationships']:
                                            if relation['Type'] == 'CHILD':
                                                text = " ".join(
                                                    word['Text'] for word in response['Blocks']
                                                    if word['BlockType'] == 'WORD' and word['Id'] in relation['Ids']
                                                )

                                    while len(table) <= row_index:
                                        table.append([])
                                    while len(table[row_index]) <= col_index:
                                        table[row_index].append('')

                                    table[row_index][col_index] = text
                    tables.append(table)

            # Écrire les données dans le CSV
            for table in tables:
                csv_writer.writerows(table)
                csv_writer.writerow([])  # Séparateur entre tableaux
    
    print(f"Extraction terminée ! Les tableaux sont sauvegardés dans {output_csv_path}")
