import csv
import re
import os

# Expression régulière pour détecter les erreurs
pattern_error = re.compile(r"\b(?:Il|II|li|Li|il|11) \d+\b | \b\d+ (?:Il|II|li|Li|il|11)\b | \b[A-Za-z]\b | [^A-Za-z0-9|.,' ]", re.VERBOSE)

def analyze_csv(csv_path, output_txt_path):
    """
    Analyse un fichier CSV et compte :
    - Le nombre de labels détectés
    - Le nombre de chiffres correctement extraits
    - Le nombre de chiffres contenant des erreurs (ex : "Il 53", "67 Il", etc.)
    - Le nombre d'erreurs (lettres isolées, caractères interdits)

    Résultats enregistrés dans un fichier texte.
    """

    label_count = 0
    correct_numbers = 0
    incorrect_numbers = 0
    error_count = 0

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter="|")  # Délimiteur "|"

        for row in reader:
            for cell in row:
                cell = cell.strip()

                # Vérification des erreurs
                if pattern_error.search(cell):
                    if re.search(r"\b(?:Il|II|li|Li|il) \d+\b | \b\d+ (?:Il|II|li|Li|il|11)\b", cell):
                        incorrect_numbers += 1  # Cas "chiffre + II"
                    elif re.search(r"\b\d+ (?:Il|II|li|Li|il)\b", cell):
                        incorrect_numbers += 1  # Cas "chiffre + II"
                    elif re.search(r"\b(?:Il|II|li|Li|il|11) \d+\b", cell):
                        incorrect_numbers += 1  # Cas "chiffre + II"
                    else:
                        error_count += 1  # Autres erreurs
                else:
                    # Vérifier si c'est un nombre ou un label
                    if re.match(r"^\d{1,3}(,\d{3})*$", cell):  # Format américain ex: 1,234
                        correct_numbers += 1
                    elif cell:  # Label (non vide)
                        label_count += 1

    # Sauvegarde des résultats
    with open(output_txt_path, mode="w", encoding="utf-8") as output_file:
        output_file.write(f"Labels détectés = {label_count}\n")
        output_file.write(f"Chiffres extraits = {correct_numbers}\n")
        output_file.write(f"Chiffres + II = {incorrect_numbers}\n")
        output_file.write(f"Erreurs détectées = {error_count}\n")

    print(f"Analyse terminée. Résultats sauvegardés dans {output_txt_path}")

# Exemple d'utilisation
csv_file_path = r"D:\GitHub\HOCR\pipelines\textract_full_preprocessing\csv_tables\csv_30_pages.csv"
# csv_file_path = r"D:\GitHub\HOCR\pipelines\textract_skew_only\csv_tables\csv_30_pages.csv"
output_txt_path = r"D:\GitHub\HOCR\result_anaysis\textract_full_processing.txt"

analyze_csv(csv_file_path, output_txt_path)
