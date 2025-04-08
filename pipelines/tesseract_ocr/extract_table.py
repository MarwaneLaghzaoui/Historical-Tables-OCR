import numpy as np
import pandas as pd
import fitz
import cv2
import pytesseract
from PIL import Image
from ultralyticsplus import YOLO

# Configuration pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def extract_tables_from_pdf(pdf_path, csv_path, psm=6):
    """Extrait les tableaux d'un PDF et les sauvegarde en CSV."""
    
    tessdata_dir_config = r'--tessdata-dir D:/Tesseract/tessdata'

    # Charger le document PDF
    pdf_document = fitz.open(pdf_path)
    print(f"Nombre de pages : {len(pdf_document)}")

    # Initialiser YOLO pour la détection des tables
    model = YOLO('keremberke/yolov8m-table-extraction')
    model.overrides.update({
        'conf': 0.25,  # Seuil de confiance
        'iou': 0.45,  # Seuil IoU
        'agnostic_nms': False,
        'max_det': 1000
    })

    all_data = []  # Liste pour stocker les données extraites de toutes les pages

    for page_num in range(len(pdf_document)):
        print(f"Traitement de la page {page_num + 1}...")

        # Convertir la page en image
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Détection des tables avec YOLO
        results = model.predict(image)

        if len(results[0].boxes) == 0:
            print(f"Aucune table détectée sur la page {page_num + 1}.")
            continue

        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])

            # Découper l'image pour extraire le tableau
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_image)

            # OCR sur l'image découpée
            print(f"Application de l'OCR sur la page {page_num + 1} (PSM={psm})...")
            extracted_text = pytesseract.image_to_string(
                cropped_image, config=f"--psm {psm} --oem 3 -l digitdetector {tessdata_dir_config}"
            )

            # Nettoyage et transformation en format CSV
            lines = extracted_text.strip().split('\n')
            data = [line.split() for line in lines if line.strip()]

            if data:
                all_data.extend(data)

    # Sauvegarde du fichier CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_path, index=False, header=False, encoding='utf-8', sep='|')
        print(f"Extraction terminée. Résultats enregistrés dans {csv_path}")
    else:
        print("Aucune donnée extraite, fichier CSV non créé.")
