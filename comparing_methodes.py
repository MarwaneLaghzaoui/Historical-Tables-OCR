import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Dossier contenant les fichiers de résultats
dossier_resultats = r"D:\GitHub\HOCR\result_anaysis"

# Expression régulière pour extraire les valeurs des fichiers
regex = {
    "Labels détectés": r"Labels détectés = (\d+)",
    "Chiffres extraits": r"Chiffres extraits = (\d+)",
    "Chiffres + II": r"Chiffres \+ II = (\d+)",
    "Erreurs détectées": r"Erreurs détectées = (\d+)"
}

# Stockage des résultats
methodes = []
resultats = {"Labels détectés": [], "Chiffres extraits": [], "Chiffres + II": [], "Erreurs détectées": []}

# Lire chaque fichier dans le dossier
for fichier in os.listdir(dossier_resultats):
    if fichier.endswith(".txt"):  # Vérifier que c'est un fichier texte
        chemin_fichier = os.path.join(dossier_resultats, fichier)

        with open(chemin_fichier, "r", encoding="utf-8") as f:
            contenu = f.read()

            # Nom de la méthode basé sur le fichier
            methode = os.path.splitext(fichier)[0]
            methodes.append(methode)

            # Extraire les valeurs avec regex
            for key, pattern in regex.items():
                match = re.search(pattern, contenu)
                if match:
                    resultats[key].append(int(match.group(1)))
                else:
                    resultats[key].append(0)  # Si absent, mettre 0

# Convertir en tableau numpy pour empilement
labels_detectes = np.array(resultats["Labels détectés"])
chiffres_extraits = np.array(resultats["Chiffres extraits"])
chiffres_ii = np.array(resultats["Chiffres + II"])
erreurs = np.array(resultats["Erreurs détectées"])

# Indices pour le graphique
x = np.arange(len(methodes))
width = 0.6  # Largeur des barres

# Création du graphique empilé
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, labels_detectes, width, label="Labels détectés", color="blue", alpha=0.7)
ax.bar(x, chiffres_extraits, width, bottom=labels_detectes, label="Chiffres extraits", color="green", alpha=0.7)
ax.bar(x, chiffres_ii, width, bottom=labels_detectes + chiffres_extraits, label="Chiffres + II", color="orange", alpha=0.7)
ax.bar(x, erreurs, width, bottom=labels_detectes + chiffres_extraits + chiffres_ii, label="Erreurs détectées", color="red", alpha=0.7)

# Labels et titre
ax.set_xlabel("Méthodes")
ax.set_ylabel("Nombre détecté")
ax.set_title("Comparaison des résultats des différentes méthodes")
ax.set_xticks(x)
ax.set_xticklabels(methodes, rotation=45, ha="right")
ax.legend()

# Afficher le graphique
plt.tight_layout()
plt.show()
