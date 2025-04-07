from pdf2image import convert_from_path
import cv2
import numpy as np

# === 1. Convertir la première page du PDF en image ===
pdf_path = r'D:/GitHub/HOCR/pipelines/textract_skew_only/pdf_folder/mortstatsh_1905-207.pdf'
pdf_path = r"D:/GitHub/HOCR/pipelines/textract_skew_only/pdf_folder/2pages.pdf"
images = convert_from_path(pdf_path, dpi=300)

# Sauvegarder temporairement la première page comme image
image_path = 'page1.png'
images[0].save(image_path, 'PNG')

# === 2. Charger l’image avec OpenCV ===
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# === 3. Appliquer une opération morphologique (par exemple ouverture) ===
kernel = np.ones((1, 1), np.uint8)
img_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# === 4. Redimensionner les images pour affichage (ex: à 40 % de la taille d’origine) ===
scale_percent = 40  # pourcentage de réduction
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
img_morph_resized = cv2.resize(img_morph, dim, interpolation=cv2.INTER_AREA)

# === 5. Afficher les résultats réduits ===
cv2.imshow('Image originale (réduite)', img_resized)
cv2.imshow('Image après ouverture morphologique (réduite)', img_morph_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
