import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

extra_angle = +45.3

def deskew_image(image, extra_angle):
    """Redresse une image sans changer sa résolution"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]]
        median_angle = np.median(angles)
    else:
        median_angle = 0  # Pas d'inclinaison détectée

    corrected_angle = median_angle + extra_angle  # Appliquer la correction de +2°

    # Rotation de l'image sans modifier la résolution
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)

    # Utiliser borderMode=cv2.BORDER_CONSTANT pour éviter de couper l’image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    return rotated

def process_pdf(pdf_path, output_pdf_path):
    """Convertit un PDF en image, redresse chaque page (+2°) et sauvegarde en PDF sans changer la résolution"""
    images = convert_from_path(pdf_path, dpi=200)  # Conserver la haute résolution

    processed_images = []
    for i, img in enumerate(images):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        deskewed_img = deskew_image(img_cv, extra_angle)  # Ajout de +2°

        # Convertir OpenCV image -> PIL image
        deskewed_pil = Image.fromarray(cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2RGB))
        processed_images.append(deskewed_pil)

    # Sauvegarder en PDF sans perte de qualité
    processed_images[0].save(output_pdf_path, save_all=True, append_images=processed_images[1:])
    print(f"✅ PDF corrigé (+2°) sauvegardé sous : {output_pdf_path}")

# Exemple d'utilisation
process_pdf("D:/EISTI/Pfe/textract/mortstatsh_1905-206.pdf", "document_redresse_2deg.pdf")
