import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
# Configuration du chemin vers Tesseract
pdf_path = r'D:\EISTI\Pfe\textract\test_page_mortstatsh_1905-207.pdf'
pdf_document = fitz.open(pdf_path)

# Chargement de la première page en image
page = pdf_document[0]
pix = page.get_pixmap(dpi=300)
image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Fonction de seuillage manuel
def manual_threshold(image, threshold):
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold] = 255
    return thresholded_image

binary_image = manual_threshold(image_gray, 200)

# Détection des colonnes noires
colonne_noire_positions = []
ligne_noire_positions = []
colored_image = np.stack([binary_image] * 3, axis=-1) 

def detection_colonnes_noires(binary_image, longueur_min, espace_min, tolerance=5):
    hauteur, largeur = binary_image.shape
    colored_image = np.stack([binary_image] * 3, axis=-1)  # Convertir en RGB
    
    x = 0
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
                            x += espace_min - 1  # Avancer mais pas trop loin
                    compteur_noirs = 0
                    compteur_trous = 0  # Réinitialiser

        # Vérification si la colonne se termine par une séquence noire valide
        if compteur_noirs >= longueur_min:
            if not colonne_noire_positions or (x - colonne_noire_positions[-1] >= espace_min):
                colonne_noire_positions.append(x)
                colored_image[:, x, :] = [255, 0, 0]  # Colorer en rouge
                x += espace_min - 1  # Avancer mais sans sauter trop loin

        x += 1  # On avance toujours d’au moins un pixel

    return colonne_noire_positions

# Exemple d'utilisation
detection_colonnes_noires(binary_image, 200, 40, tolerance=2)



def detection_lignes_noires(binary_image, longueur_min, espace_min, tolerance=5):
    hauteur, largeur = binary_image.shape
    colored_image = np.stack([binary_image] * 3, axis=-1)  # Convertir en RGB
    
    y = 0
    while y < hauteur:
        ligne = binary_image[y, :]
        compteur_noirs = 0
        compteur_trous = 0
        debut_ligne = None

        # Vérifier si la ligne contient une séquence noire assez longue
        for x in range(largeur):
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
                            y += espace_min - 1  # Avancer mais pas trop loin
                    compteur_noirs = 0
                    compteur_trous = 0  # Réinitialiser

        # Vérification si la ligne se termine par une séquence noire valide
        if compteur_noirs >= longueur_min:
            if not ligne_noire_positions or (y - ligne_noire_positions[-1] >= espace_min):
                ligne_noire_positions.append(y)
                colored_image[y, :, :] = [0, 0, 255]  # Colorer en bleu
                y += espace_min - 1  # Avancer mais sans sauter trop loin

        y += 1  # On avance toujours d’au moins un pixel

    return ligne_noire_positions

# Exemple d'utilisation
detection_lignes_noires(binary_image, 200, 40, tolerance=2)

# Définir la zone d'intérêt entre les colonnes
zone_start = colonne_noire_positions[0]
zone_end = colonne_noire_positions[1] +10
ligne_depart = ligne_noire_positions[2]
'''petite tolérance'''
zone_interet = binary_image[ligne_noire_positions[2]:, zone_start:zone_end]

# Sauvegarder la zone d'intérêt
zone_interet_sauvegardee = zone_interet.copy()

# Détection des points dans la zone d'intérêt
min_area = 5
max_area = 30

contours, _ = cv2.findContours(255 - zone_interet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

binary_image_colored = np.stack([binary_image] * 3, axis=-1)
droite_points = {}

# Recherche de la première colonne blanche après la deuxième colonne noire


for contour in contours:
    area = cv2.contourArea(contour)
    if min_area <= area <= max_area:
        x, y, w, h = cv2.boundingRect(contour)
        if (x > 0 and y > 0 and x + w < zone_interet.shape[1] and y + h < zone_interet.shape[0]):
            surrounding_area = zone_interet[y-1:y+h+1, x-1:x+w+1]
            if np.all(surrounding_area[0, :] == 255) and np.all(surrounding_area[-1, :] == 255) and \
               np.all(surrounding_area[:, 0] == 255) and np.all(surrounding_area[:, -1] == 255):
                cv2.rectangle(colored_image, (x + zone_start, y + ligne_depart ), (x + w + zone_start, y + ligne_depart + h), (0, 0, 255), 2)
                ligne = y // 10
                if (ligne not in droite_points or x + w > droite_points[ligne][0]):
                    droite_points[ligne] = (x + w, y + ligne_depart)

droite_colonne = zone_interet.shape[1] - 1

filtered_droite_points = {ligne: point for ligne, point in droite_points.items()
                          if abs(droite_colonne - point[0]) <= 15}

# Dessiner des rectangles distincts pour les points restants et ajouter un numéro
for i, (ligne, point) in enumerate(filtered_droite_points.items(), start=1):
    cv2.rectangle(colored_image, (point[0] + zone_start, point[1]),
                  (point[0] + zone_start + 5, point[1] + 5), (0, 255, 0), 2)
    cv2.putText(colored_image, str(i), (point[0] + zone_start + 7, point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

last_ligne, last_point = list(filtered_droite_points.items())[-1]
last_y = last_point[1]

print(last_point)
print(last_y)

new_ligne_depart = last_y - 10
''' encore une marge'''

new_zone_interet = binary_image[new_ligne_depart:, zone_start:zone_end]
new_zone_interet_sauvegardee = new_zone_interet.copy()

lignes_noires_colonne_gauche = [point[1] for point in filtered_droite_points.values()]


colonne_blanche = None
for i in range(colonne_noire_positions[1] + 1, binary_image.shape[1]):
    if np.all(binary_image[:, i] == 255):  # Colonne composée uniquement de blancs
        colonne_blanche = i
        break


# Fonction pour insérer des lignes avec décalage
def inserer_lignes_decalees(binary_image, start_col, end_col, lignes_noires_colonne_gauche):
    hauteur, largeur = binary_image.shape
    lignes_blanches = np.full((10, largeur), 255, dtype=np.uint8)
    lignes_noires = np.full((4, largeur), 255, dtype=np.uint8)
    lignes_noires[:, start_col:end_col] = 0
    sequence_lignes = np.vstack((lignes_blanches, lignes_noires, lignes_blanches))
    
    image_etendue = binary_image.copy()
    decalage = 0
    for y in sorted(lignes_noires_colonne_gauche):
        if y + decalage < image_etendue.shape[0]:
            image_etendue = np.insert(image_etendue, y + 10 + decalage, sequence_lignes, axis=0)
            decalage += 24
    return image_etendue


image_etendue = inserer_lignes_decalees(binary_image, colonne_noire_positions[1], colonne_blanche, lignes_noires_colonne_gauche)

# Étirement de la zone sauvegardée
nouvelle_hauteur = image_etendue.shape[0] - new_ligne_depart
zone_interet_etiree = cv2.resize(new_zone_interet_sauvegardee, (new_zone_interet.shape[1], nouvelle_hauteur), interpolation=cv2.INTER_LINEAR)

# Réinsérer la zone étirée
image_etendue[new_ligne_depart:, zone_start:zone_end] = zone_interet_etiree

# Visualisation de l'image finale
# Visualisation de l'image finale
plt.figure(figsize=(10, 10))
plt.imshow(image_etendue, cmap='gray')
plt.axis('off')

# Enregistrement de l'image au format PDF
plt.savefig('output_image.pdf', bbox_inches='tight', pad_inches=0)

# Fermeture de la figure pour éviter l'affichage
plt.close()

