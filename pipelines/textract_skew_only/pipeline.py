import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Ajoute le dossier du script

from recadrer_image_v4 import correct_pdf_skew_angle
from extract_table import extract_tables_from_pdf

def main():

    #Define file names to make output names different (must be the name of file you want to process)
    pdf_file_name = r"mortstatsh_1905-207.pdf"
    csv_file_name = r"mortstatsh_1905-207.csv"

    pdf_path = r"D://GitHub//HOCR//pipelines//textract_skew_only//pdf_folder//"
    csv_path = r"D://GitHub//HOCR//pipelines//textract_skew_only//csv_tables//"+csv_file_name

    print(pdf_path+pdf_file_name)
    print(csv_path+csv_file_name)

    correct_pdf_skew_angle(pdf_path,pdf_file_name)

    pdf_straightened_path = pdf_path+"output_"+pdf_file_name
    # Extraction des tableaux du PDF
    extract_tables_from_pdf(pdf_straightened_path, csv_path)

main()