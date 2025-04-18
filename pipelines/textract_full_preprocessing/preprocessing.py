import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz 
import os
from PIL import Image
import pytesseract
import pandas as pd
from openpyxl import load_workbook
import shutil
from pypdf import PdfReader, PdfWriter
import io
from matplotlib.backends.backend_pdf import PdfPages

# def manual_threshold(image, threshold):
#     thresholded_image = np.zeros_like(image)
#     thresholded_image[image > threshold] = 255
#     return thresholded_image

def detection_premiere_colonne_noire(binary_image, colored_image):
    seuil_noir = 0.03
    hauteur, largeur = binary_image.shape
    x = 0
    while x < largeur:
        colonne = binary_image[:, x]
        noirs = np.sum(colonne == 0)
        if noirs / hauteur >= seuil_noir:
            colored_image[:, x, :] = [255, 0, 0]
            if x >= largeur:
                break
            colonne = binary_image[:, x]
            noirs = np.sum(colonne == 0)
            colonne_noire_positions.append(x)
            x = largeur + 1
        x += 1

def detection_colonnes_noires(binary_image, longueur_min, espace_min, tolerance, decalage_premiere_colonne, colored_image):
    hauteur, largeur = binary_image.shape
    
    x = colonne_noire_positions[0] + decalage_premiere_colonne
    while x < largeur:
        colonne = binary_image[:, x]
        compteur_noirs = 0
        compteur_trous = 0
        debut_colonne = None

        # Vérifier si la colonne contient une séquence noire assez longue
        for y in range(hauteur):
            if colonne[y] == 0:  # Pixel noir
                if compteur_noirs == 0:
                    debut_colonne = y
                compteur_noirs += 1
                compteur_trous = 0  # Réinitialiser le compteur de trous
            else:
                if compteur_noirs > 0 and compteur_trous < tolerance:
                    compteur_trous += 1  # Tolérer un petit trou
                else:
                    if compteur_noirs >= longueur_min:  # Colonne noire validée
                        if not colonne_noire_positions or (x - colonne_noire_positions[-1] >= espace_min):
                            colonne_noire_positions.append(x)
                            colored_image[:, x, :] = [255, 0, 0]  # Colorer en rouge
                            x += espace_min -1 # Avancer mais pas trop loin
                    compteur_noirs = 0
                    compteur_trous = 0  # Réinitialiser

        # Vérification si la colonne se termine par une séquence noire valide
        if compteur_noirs >= longueur_min:
            if not colonne_noire_positions or (x - colonne_noire_positions[-1] >= espace_min):
                colonne_noire_positions.append(x)
                colored_image[:, x, :] = [255, 0, 0]  # Colorer en rouge
                x += espace_min - 1  # Avancer mais sans sauter trop loin

        x += 1  # On avance toujours d'au moins un pixel

    return colonne_noire_positions

def detection_lignes_noires(binary_image, longueur_min, espace_min, tolerance, colored_image):
    hauteur, largeur = binary_image.shape
    
    y = 0
    while y < hauteur:
        ligne = binary_image[y, :]
        compteur_noirs = 0
        compteur_trous = 0
        debut_ligne = None
        
        # Vérifier si la ligne contient une séquence noire assez longue
        for x in range(colonne_noire_positions[1]):
            if ligne[x] == 0:  # Pixel noir
                if compteur_noirs == 0:
                    debut_ligne = x
                compteur_noirs += 1
                compteur_trous = 0  # Réinitialiser le compteur de trous
            else:
                if compteur_noirs > 0 and compteur_trous < tolerance:
                    compteur_trous += 1  # Tolérer un petit trou
                else:
                    if compteur_noirs >= longueur_min:  # Ligne noire validée
                        if not ligne_noire_positions or (y - ligne_noire_positions[-1] >= espace_min):
                            ligne_noire_positions.append(y)
                            colored_image[y, :, :] = [0, 0, 255]  # Colorer en bleu
                            y += espace_min -1 # Avancer mais pas trop loin
                    compteur_noirs = 0
                    compteur_trous = 0  # Réinitialiser

        # Vérification si la ligne se termine par une séquence noire valide
        if compteur_noirs >= longueur_min:
            if not ligne_noire_positions or (y - ligne_noire_positions[-1] >= espace_min):
                ligne_noire_positions.append(y)
                colored_image[y, :, :] = [0, 0, 255]  # Colorer en bleu
                y += espace_min - 1  # Avancer mais sans sauter trop loin

        y += 1  # On avance toujours d'au moins un pixel

    return ligne_noire_positions

# Fonction pour insérer des lignes avec décalage
def inserer_lignes_decalees(binary_image, start_col, end_col, lignes_noires_colonne_gauche, coefficient_haut):
    hauteur, largeur = binary_image.shape
    lignes_blanches = np.full((int(3*coefficient_haut), largeur), 255, dtype=np.uint8)
    lignes_noires = np.full((int(2*coefficient_haut), largeur), 255, dtype=np.uint8)
    lignes_noires[:, start_col:end_col] = 1
    sequence_lignes = np.vstack((lignes_blanches, lignes_noires, lignes_blanches))
    
    image_etendue = binary_image.copy()
    decalage = 0
    for y in sorted(lignes_noires_colonne_gauche):
        if y + decalage < image_etendue.shape[0]:
            image_etendue = np.insert(image_etendue, y + int(4*coefficient_haut) + decalage, sequence_lignes, axis=0)
            decalage += 2*int(3*coefficient_haut)+int(2*coefficient_haut)
    return image_etendue

def process_image_to_pdf_page(image):
    """Convert a numpy image to a PDF page"""
    # Save image to a BytesIO object
    img_byte_arr = io.BytesIO()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(img_byte_arr, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Create PDF reader from the BytesIO object
    img_byte_arr.seek(0)
    pdf_reader = PdfReader(img_byte_arr)
    
    return pdf_reader.pages[0]

def traitement_page(indice, pdf_document, pdf_writer):
    global colonne_noire_positions
    global ligne_noire_positions
    global lignes_noires_colonne_gauche_save 

    colonne_noire_positions = []
    ligne_noire_positions = []
    lignes_noires_colonne_gauche_save = []

    min_area_points = 5
    max_area_points = 60
    distance_max_point = 5
    tolerance_apres_ligne = 1

    longueur_min_colonne = 100

    longueur_min_ligne = 100

    terme_longueur_ajoutee = 20
    
    page = pdf_document[indice]
    # mat = fitz.Matrix(1.0, 1.0)
    pix = page.get_pixmap(dpi=100)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    binary_image = image_gray
    # binary_image = manual_threshold(image_gray, 200)
    
    #calcul du coefficient de résolution
    hauteur, largeur = binary_image.shape
    coefficient = hauteur * largeur / (3450 * 2659)
    coefficient_haut = hauteur / 3450 + 1
    coefficient_larg = largeur / 2659 + 1
    
    # Détection des colonnes noires
    colored_image = np.stack([binary_image] * 3, axis=-1)
    
    detection_premiere_colonne_noire(binary_image, colored_image)
    detection_colonnes_noires(binary_image, longueur_min_colonne* int(coefficient_haut), 40* int(coefficient_haut),5* int(coefficient_haut), 40* int(coefficient_larg), colored_image)
    detection_lignes_noires(binary_image, longueur_min_ligne* int(coefficient_larg), 40 * int(coefficient_larg), 0* int(coefficient_larg), colored_image)
    
    if(colonne_noire_positions[1] - colonne_noire_positions[0] > 200 * coefficient_larg):
        print("Traitement de la page " + str(indice) + " (recto)")
        
        # On fait face au recto (distance premiere colonne a deuxieme superieure a 200 pixels)
        zone_start = colonne_noire_positions[0]
        zone_end = colonne_noire_positions[1] + int(tolerance_apres_ligne * coefficient_larg)
        ligne_depart = ligne_noire_positions[1]
        zone_interet = binary_image[ligne_noire_positions[1]:, zone_start:zone_end]

        # Sauvegarder la zone d'intérêt
        zone_interet_sauvegardee = zone_interet.copy()

        # Détection des points dans la zone d'intérêt
        min_area = min_area_points * coefficient
        max_area = max_area_points * coefficient

        contours, _ = cv2.findContours(255 - zone_interet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        binary_image_colored = np.stack([binary_image] * 3, axis=-1)
        droite_points = {}

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                if (x > 0 and y > 0 and x + w < zone_interet.shape[1] and y + h < zone_interet.shape[0]):
                    surrounding_area = zone_interet[y-1:y+h+1, x-1:x+w+1]
                    if np.all(surrounding_area[0, :] == 255) and np.all(surrounding_area[-1, :] == 255) and \
                       np.all(surrounding_area[:, 0] == 255) and np.all(surrounding_area[:, -1] == 255):
                        cv2.rectangle(colored_image, (x + zone_start, y + ligne_depart), (x + w + zone_start, y + ligne_depart + h), (0, 0, 255), 2)
                        ligne = y // 10
                        if (ligne not in droite_points or x + w > droite_points[ligne][0]):
                            droite_points[ligne] = (x + w, y + ligne_depart)

        droite_colonne = zone_interet.shape[1] - 1

        filtered_droite_points = {ligne: point for ligne, point in droite_points.items()
                                  if abs(droite_colonne - point[0]) <= (int(distance_max_point * coefficient_larg) + int(tolerance_apres_ligne * coefficient_larg))}

        # Dessiner des rectangles distincts pour les points restants et ajouter un numéro
        for i, (ligne, point) in enumerate(filtered_droite_points.items(), start=1):
            cv2.rectangle(colored_image, (point[0] + zone_start, point[1]),
                          (point[0] + zone_start + 5, point[1] + 5), (0, 255, 0), 2)
            cv2.putText(colored_image, str(i), (point[0] + zone_start + 7, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        if filtered_droite_points:  # Check if there are any filtered points
            last_ligne, last_point = list(filtered_droite_points.items())[-1]
            last_y = last_point[1]
            new_ligne_depart = last_y - 10
            new_zone_interet = binary_image[new_ligne_depart:, zone_start:zone_end]
            new_zone_interet_sauvegardee = new_zone_interet.copy()
        else:
            # Handle the case where there are no filtered points
            new_ligne_depart = ligne_depart
            new_zone_interet = zone_interet.copy()
            new_zone_interet_sauvegardee = new_zone_interet.copy()

        lignes_noires_colonne_gauche = [point[1] for point in filtered_droite_points.values()]
        
        # Save these positions for verso pages
        lignes_noires_colonne_gauche_save.clear()  # Clear previous values
        lignes_noires_colonne_gauche_save.extend([x - ligne_noire_positions[0] for x in lignes_noires_colonne_gauche])

        # Recherche de la première colonne blanche après la deuxième colonne noire
        colonne_blanche = None
        for i in range(colonne_noire_positions[1] + 1, binary_image.shape[1]):
            if np.all(binary_image[:, i] == 255):  # Colonne composée uniquement de blancs
                colonne_blanche = i
                break            

        image_etendue = inserer_lignes_decalees(binary_image, colonne_noire_positions[1], colonne_blanche if colonne_blanche else binary_image.shape[1], lignes_noires_colonne_gauche, coefficient_haut)

        # Étirement de la zone sauvegardée
        if new_zone_interet.size > 0:  # Check if the zone is not empty
            nouvelle_hauteur = image_etendue.shape[0] - ligne_noire_positions[1] + int(terme_longueur_ajoutee*coefficient_haut)
            zone_interet_etiree = cv2.resize(new_zone_interet_sauvegardee, (new_zone_interet.shape[1], nouvelle_hauteur), interpolation=cv2.INTER_LINEAR)
            hauteur_disponible = image_etendue.shape[0] - new_ligne_depart
            zone_interet_etiree = zone_interet_etiree[:hauteur_disponible, :]

            # Réinsérer la zone étirée
            image_etendue[new_ligne_depart:, zone_start:zone_end] = zone_interet_etiree

        # Convert the processed image to a PDF page and add to writer
        pdf_page = process_image_to_pdf_page(image_etendue)
        pdf_writer.add_page(pdf_page)
        
    else:
        # On fait face au verso        
        print("Traitement de la page " + str(indice) + " (verso)")
        
        # Check if we have previously saved lines from recto pages
        if not lignes_noires_colonne_gauche_save:
            print(f"Warning: No saved lines for verso page {indice}. Processing as basic page.")
            # Process as basic page and add to writer
            pdf_page = process_image_to_pdf_page(binary_image)
            pdf_writer.add_page(pdf_page)
            return
            
        colonne_blanche = None
        for i in range(colonne_noire_positions[1] + 1, binary_image.shape[1]):
            if np.all(binary_image[:, i] == 255):  # Colonne composée uniquement de blancs
                colonne_blanche = i
                break
        
        # Make sure ligne_noire_positions has elements before accessing it
        if ligne_noire_positions:
            lignes_noires_colonne_gauche = [x + ligne_noire_positions[0] for x in lignes_noires_colonne_gauche_save]
            image_etendue = inserer_lignes_decalees(binary_image, colonne_noire_positions[0], 
                                                   colonne_blanche if colonne_blanche else binary_image.shape[1], 
                                                   lignes_noires_colonne_gauche, coefficient_haut)
            
            # Convert the processed image to a PDF page and add to writer
            pdf_page = process_image_to_pdf_page(image_etendue)
            pdf_writer.add_page(pdf_page)
        else:
            print(f"Warning: No line positions detected for verso page {indice}.")
            pdf_page = process_image_to_pdf_page(binary_image)
            pdf_writer.add_page(pdf_page)

def traitement_document(pdf_path):
    # Configuration du chemin
    pdf_document = fitz.open(pdf_path)
    nombre_pages = len(pdf_document)

    BASE_DIR = os.path.dirname(__file__)
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    dossier_page = os.path.join(BASE_DIR, "pdf_folder")
    
    # Ensure output directory exists
    os.makedirs(dossier_page, exist_ok=True)

    # Création d'un objet PdfWriter pour le fichier final
    pdf_writer = PdfWriter()

    # Global variables for page processing
    global colonne_noire_positions
    global ligne_noire_positions
    global lignes_noires_colonne_gauche_save
    
    colonne_noire_positions = []
    ligne_noire_positions = []
    lignes_noires_colonne_gauche_save = []

    # Process each page and add directly to the PDF writer
    for p in range(nombre_pages):
        traitement_page(p, pdf_document, pdf_writer)
        colonne_noire_positions = []
        ligne_noire_positions = []
        lignes_noires_inserees = []

    # Save the final PDF
    output_path = os.path.join(dossier_page, f"{file_name}.pdf")
    print(f"Saving final PDF to: {output_path}")
    with open(output_path, "wb") as fichier_sortie:
        pdf_writer.write(fichier_sortie)

    print("Toutes les pages ont été traitées.")


# traitement_document()