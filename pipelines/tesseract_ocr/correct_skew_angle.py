import cv2
import numpy as np
import fitz
from PIL import Image

def rotate_image(image, angle):

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return corrected

def determine_score(arr):
    histogram = np.sum(arr, axis=2, dtype=float)
    score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
    return score

def correct_skew(image, delta=0.1, limit=5):
    
    # Binarisation
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Recherche du meilleur angle de correction
    angles = np.arange(-limit, limit + delta, delta)
    img_stack = np.stack([rotate_image(thresh, angle) for angle in angles], axis=0)
    scores = determine_score(img_stack)
    best_angle = angles[np.argmax(scores)]
    
    # Correction finale
    corrected = rotate_image(image, best_angle)
    return best_angle, corrected

def load_pdf_as_images(pdf_path, dpi=300):

    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        print(f"Correction de la page {page_num + 1}/{len(pdf_document)}...")  # Affichage du numéro de page
        page = pdf_document[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convertir en niveaux de gris
        images.append(img_gray)
    return images

def save_images_as_pdf(images, pdf_output_path):

    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(pdf_output_path, save_all=True, append_images=pil_images[1:])
    print(f"PDF corrigé enregistré sous : {pdf_output_path}")


def correct_pdf_skew_angle(pdf_path,pdf_file_name):

    # Charger toutes les pages du PDF en images
    images = load_pdf_as_images(pdf_path+pdf_file_name)

    # Appliquer la correction d'inclinaison à chaque page
    corrected_images = []
    for i, img in enumerate(images):
        angle, corrected = correct_skew(img)
        print(f"Page {i+1} corrigée (angle : {angle:.2f}°)")
        corrected_images.append(corrected)


    pdf_output_path = pdf_path+"output_"+pdf_file_name
    # Sauvegarder l'ensemble des pages corrigées dans un nouveau PDF
    save_images_as_pdf(corrected_images, pdf_output_path)


# correct_pdf_skew_angle()