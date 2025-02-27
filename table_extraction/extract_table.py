import boto3
import csv

def extract_tables_from_pdf(pdf_path, output_csv_path):
    """Extrait les tableaux d'un PDF en utilisant AWS Textract et les enregistre dans un CSV."""
    session = boto3.Session(profile_name="cytech")  
    textract = session.client("textract")

    # Lire le fichier PDF
    with open(pdf_path, 'rb') as pdf_file:
        pdf_bytes = pdf_file.read()

    # Appeler Textract pour analyser le document
    response = textract.analyze_document(
        Document={'Bytes': pdf_bytes},
        FeatureTypes=['TABLES']
    )

    # Extraire les tableaux du résultat
    tables = []
    for block in response['Blocks']:
        if block['BlockType'] == 'TABLE':
            table = []
            for relationship in block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        cell = next(
                            (b for b in response['Blocks'] if b['Id'] == child_id),
                            None
                        )
                        if cell and cell['BlockType'] == 'CELL':
                            row_index = cell['RowIndex'] - 1
                            col_index = cell['ColumnIndex'] - 1
                            text = ''.join(
                                word['Text'] for word in response['Blocks']
                                if word['BlockType'] == 'WORD' and
                                'Relationships' in cell and
                                any(word['Id'] in cell['Relationships'][0]['Ids'] for _ in cell['Relationships'])
                            )
                            while len(table) <= row_index:
                                table.append([])
                            while len(table[row_index]) <= col_index:
                                table[row_index].append('')
                            table[row_index][col_index] = text
            tables.append(table)

    # Sauvegarder les tableaux dans un fichier CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for table in tables:
            csv_writer.writerows(table)
            csv_writer.writerow([])  # Séparateur entre tableaux

    print(f"Les tableaux ont été extraits et sauvegardés dans {output_csv_path}")


file_path = r'D:/GitHub/HOCR/image_straightening/image_redressee_302.pdf'
output_path = r'D:/GitHub/HOCR/table_extraction/image_redressee_302_output.csv' 
extract_tables_from_pdf(file_path,output_path)