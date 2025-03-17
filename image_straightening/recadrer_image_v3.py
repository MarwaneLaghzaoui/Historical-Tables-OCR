import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from PIL import Image  # Bibliothèque Pillow


# Fonction de seuillage manuel
def manual_threshold(image, threshold):
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold] = 255
    return thresholded_image

# Fonction pour détecter l'angle d'inclinaison de l'image
# def detect_angle(binary_image):
#     edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    
#     if lines is not None:
#         angles = []
#         for line in lines:
#             rho, theta = line[0]
#             angle = np.degrees(theta) - 90
#             if angle < -45:
#                 angle += 180
#             if abs(angle) < 5:
#                 angles.append(angle)
#         if angles:
#             return np.mean(angles)
#     return 0

def detect_angle(binary_image):
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if angle < -45:
                angle += 180
            if abs(angle) < 15:
                angles.append(angle)
        if angles:
            return np.mean(angles)
    return 0


# def detect_angle(binary_image, crop_ratio=0.15):

#     if len(binary_image.shape) == 3:  
#         binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
#     h, w = binary_image.shape  # Récupérer les dimensions de l'image
#     cropped_image = binary_image[:int(h * crop_ratio), :]  # Garder seulement les premières lignes

#     # Détection des bords avec Canny
#     edges = cv2.Canny(cropped_image, 50, 150, apertureSize=3)

#     # Détection des lignes avec Hough Transform
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    
#     if lines is not None:
#         angles = []
#         for line in lines:
#             rho, theta = line[0]
#             angle = np.degrees(theta) - 90
#             if angle < -45:
#                 angle += 180  # Correction pour éviter des valeurs erronées
#             if abs(angle) < 10:  # Filtrer les angles aberrants
#                 angles.append(angle)
        
#         if angles:
#             return np.mean(angles)  # Retourner l'angle moyen

#     return 0  # Retourner 0 si aucune ligne détectée


# Fonction pour redresser l'image en fonction de l'angle détecté
def straighten_image(image, angle,pulse):
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle*pulse, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image




def main():
    # Chemin du fichier PDF source et du fichier PDF de sortie
    pdf_path = r'D:/GitHub/HOCR/image_straightening/mortstatsh_1905.pdf'

    # output_pdf_path = r'D:/GitHub/HOCR/image_straightening/image_redressee_302.pdf' 
    output_pdf_path = r'D:/GitHub/HOCR/image_straightening/tmp.pdf' 
    
    # Charger le document PDF
    pdf_document = fitz.open(pdf_path)

    # Liste pour stocker les images des pages traitées
    pages_images = []
    
    minimum_angle = 0.01
    # Parcourir chaque page du PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        angle=1000
        pulse=1 #To ensure visiting many possible combinations
        # Image binarisée
        binary_image = manual_threshold(image_gray, 200)
        while((abs(angle)>=minimum_angle)):

            angle = detect_angle(binary_image)
            # Redresser l'image
            image_redresse = straighten_image(image, angle,pulse)            
            binary_image = image_redresse
                        # Détecter l'angle d'inclinaison
            
            
            pulse = pulse + 1
            print(f"Page {page_num + 1}, Angle d'inclinaison détectée : {angle:.2f} degrés, Itération {pulse}")
            
        # Reconvertir l'image redressée en niveaux de gris et la binariser
        image_gray_redresse = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
        # image_gray_redresse = cv2.cvtColor(image_redresse, cv2.COLOR_RGB2GRAY)
        binary_image_redresse = manual_threshold(image_gray_redresse, 200)
        pil_image = Image.fromarray(binary_image_redresse)
        pages_images.append(pil_image)
    pages_images[0].save(output_pdf_path, save_all=True, append_images=pages_images[1:])


if __name__ == "__main__":
    main()
