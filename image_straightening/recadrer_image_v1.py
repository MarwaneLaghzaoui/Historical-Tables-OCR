import cv2
import fitz
import numpy as np

def corriger_inclinaison(pdf_path):
    # Charger le PDF et récupérer la page
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc.load_page(0)  # Charger la première page

    # Rendu de la page en image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Augmenter la résolution
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # Vérifier et convertir l'image en niveaux de gris
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Appliquer un flou pour réduire le bruit
    flou = cv2.GaussianBlur(gris, (5, 5), 0)

    # Détection des contours
    contours = cv2.Canny(flou, 50, 150)

    # Trouver les lignes dans l'image
    lignes = cv2.HoughLinesP(contours, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lignes is None:
        print(f"Aucune ligne détectée dans l'image {pdf_path}")
        return image  # Retourner l'image originale si aucune ligne n'est détectée

    # Trier les lignes en fonction de la position verticale (prendre la ligne la plus haute)
    lignes = sorted(lignes, key=lambda ligne: min(ligne[0][1], ligne[0][3]))

    # Prendre la première ligne détectée
    x1, y1, x2, y2 = lignes[0][0]
    
    # Calcul de l'angle de la ligne par rapport à l'horizontale
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # Corriger l'angle
    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
    image_redressee = cv2.warpAffine(image, matrice_rotation, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return image_redressee

# Exemple d'utilisation
pdf_path = "D://GitHub//HOCR//test_pages//test_page_mortstatsh_1905-207.pdf"
image_corrigee = corriger_inclinaison(pdf_path)
if image_corrigee is not None:
    cv2.imwrite("image_redressee.png", cv2.cvtColor(image_corrigee, cv2.COLOR_RGB2BGR))
    print("Image redressée enregistrée sous 'image_redressee.png'")
