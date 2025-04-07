import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from correct_skew_angle import correct_pdf_skew_angle
from extract_table import extract_tables_from_pdf

def main():

    Tk().withdraw()  # Ne pas afficher la fenêtre principale Tk
    selected_pdf_path = askopenfilename(title="Choisir un fichier PDF", filetypes=[("PDF files", "*.pdf")])

    if not selected_pdf_path:
        print("Aucun fichier sélectionné.")
        return

    # Chemin vers le fichier à traiter
    file_name = os.path.splitext(os.path.basename(selected_pdf_path))[0]
    pdf_file_name = os.path.basename(selected_pdf_path)
    csv_file_name = file_name + ".csv"

    # Définir les chemins de sortie
    pdf_folder = os.path.dirname(selected_pdf_path) + os.sep
    csv_folder = "./pipelines/textract_skew_only/csv_tables/"
    csv_path = os.path.join(csv_folder, csv_file_name)

    correct_pdf_skew_angle(pdf_folder,pdf_file_name)

    straightened_pdf_path = pdf_folder+"output_"+pdf_file_name
    # Extraction des tableaux du PDF
    extract_tables_from_pdf(straightened_pdf_path, csv_path)

main()