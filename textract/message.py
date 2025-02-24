import cv2
import numpy as np
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

# Charger le PDF et extraire la première page en tant qu'image
pdf_path = r'D://EISTI//Pfe//textract//test_page_mortstatsh_1905-207.pdf'
output_pdf_path = r"D://EISTI//Pfe//textract//output.pdf"

pdf_document = fitz.open(pdf_path)
page = pdf_document[0]
pix = page.get_pixmap(dpi=500)
image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# Convertir en niveaux de gris et binariser l'image
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)

# Détection des lignes et colonnes
def detection_colonnes_noires(image, longueur_min=200, espace_min=40, tolerance=2):
    hauteur, largeur = image.shape
    positions = []
    for x in range(largeur):
        colonne = image[:, x]
        if np.sum(colonne == 0) >= longueur_min:  # Vérifier le nombre de pixels noirs
            if not positions or (x - positions[-1] >= espace_min):
                positions.append(x)
                cv2.line(image, (x, 0), (x, hauteur), (0, 0, 0), 2)  # Tracer une ligne noire
    return positions

def detection_lignes_noires(image, longueur_min=200, espace_min=40, tolerance=2):
    hauteur, largeur = image.shape
    positions = []
    for y in range(hauteur):
        ligne = image[y, :]
        if np.sum(ligne == 0) >= longueur_min:
            if not positions or (y - positions[-1] >= espace_min):
                positions.append(y)
                cv2.line(image, (0, y), (largeur, y), (0, 0, 0), 2)  # Tracer une ligne noire
    return positions

# Ajouter les lignes et colonnes détectées
detection_colonnes_noires(binary_image)
detection_lignes_noires(binary_image)

# Convertir l'image en format compatible pour le PDF
image_pil = Image.fromarray(binary_image)
a4_width, a4_height = A4
image_pil = image_pil.resize((int(a4_width), int(a4_height)), Image.LANCZOS)

# Sauvegarder l'image temporairement
temp_image_path = "temp_image.png"
image_pil.save(temp_image_path, "PNG")

# Générer un PDF avec l'image
c = canvas.Canvas(output_pdf_path, pagesize=A4)
c.drawImage(temp_image_path, 0, 0, width=a4_width, height=a4_height)
c.showPage()
c.save()

print(f"✅ PDF généré avec succès : {output_pdf_path}")
