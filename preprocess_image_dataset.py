import os

# Répertoire contenant les dossiers 1 à 9
base_dir = "dataset_tiff"

# Vérifier les dossiers de 1 à 9 (exclure 0)
for digit in range(1, 10):
    digit_path = os.path.join(base_dir, str(digit))

    if os.path.isdir(digit_path):  # Vérifie que c'est un dossier
        for filename in os.listdir(digit_path):
            if filename.endswith(".box"):
                box_path = os.path.join(digit_path, filename)
                
                # Lire le fichier .box
                with open(box_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                if lines:  # Vérifier qu'il y a du contenu
                    first_char = lines[0][0]  # Premier caractère du fichier
                    if first_char != str(digit):
                        print(f"❌ Erreur dans {box_path} : attendu {digit}, trouvé {first_char}")
                    
                else:
                    print(f"⚠️ Fichier vide : {box_path}")
