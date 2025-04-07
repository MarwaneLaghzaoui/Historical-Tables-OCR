import fitz
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

def extract_pdf_pages(pdf_path, output_pdf_path, page_nums):
    pdf_document = fitz.open(pdf_path)
    new_pdf = fitz.open()

    # for page_num in range(page_nums):
    for page_num in page_nums:
        if page_num < 0 or page_num >= len(pdf_document):
            continue

        print(f"Extraction de la page {page_num + 1}...")
        new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

    if len(new_pdf) > 0:
        new_pdf.save(output_pdf_path)
        print(f"PDF extrait enregistré sous : {output_pdf_path}")
    else:
        print("Aucune page valide extraite.")

Tk().withdraw()  # Ne pas afficher la fenêtre principale Tk
pdf_path = askopenfilename(title="Choisir un fichier PDF", filetypes=[("PDF files", "*.pdf")])
file_name = os.path.splitext(os.path.basename(pdf_path))[0]
output_pdf_path = file_name
page_nums = [207,208]

extract_pdf_pages(pdf_path, output_pdf_path, page_nums)
