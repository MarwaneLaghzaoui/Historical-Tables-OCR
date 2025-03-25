import pandas as pd
import re

def remove_unwanted_caracters_from_csv(input_csv):
    # Lire le fichier CSV
    df = pd.read_csv(input_csv, quotechar='"')

    # Fonction pour nettoyer chaque cellule
    def clean_string(s):
        if isinstance(s, str):
            # Supprime les caractères indésirables (tout sauf lettres, chiffres, points et pipes)
            s = re.sub(r'[^a-zA-Z0-9.,|]', '', s)
        return s  # Ne modifie pas les autres types

    # Appliquer la fonction de nettoyage à chaque cellule du DataFrame
    df = df.applymap(clean_string)

    # Sauvegarde avec un séparateur `|` sans ajouter de guillemets
    df.to_csv(input_csv, index=False, sep='|', quoting=3)  # `quoting=3` désactive les guillemets
