import cv2
import numpy as np
import fitz  # PyMuPDF

def preprocess_pdf(pdf_path, save_path):
    # 1. Charger le PDF et extraire la première page en image
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc.load_page(0)  # Charger la première page
    
    # Rendu en image avec une résolution plus élevée
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # x2 pour améliorer la qualité
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    # 2. Convertir en niveaux de gris
    if img.shape[-1] == 3:  # Vérifie si l'image est en RGB
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img  # Si déjà en niveaux de gris, pas de conversion

    # 3. Appliquer un filtre morphologique pour enlever les petits points noirs isolés
    kernel = np.ones((3, 3), np.uint8)  # Noyau 3x3
    img_cleaned = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    
    # # 4. Binarisation avec Otsu pour un contraste maximal
    # _, img_binary = cv2.threshold(img_cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Sauvegarde de l’image prétraitée
    cv2.imwrite(save_path, img_cleaned)
    # cv2.imwrite(save_path, img_binary)
    print(f"Image sauvegardée sous {save_path}")

# Exemple d'utilisation
pdf_path = "D://GitHub//HOCR//test_pages//test_page_mortstatsh_1905-207.pdf"
preprocess_pdf(pdf_path, "table_pretraitee.jpg")
