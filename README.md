# Prérequis afin d'exécuter le code

Installer python 3.12.9 de préférence et pip 25.0.1 et les ajouter aux variables d'environnement pour éviter les conflits avec d'autres versions de python

# Éventuellement creer un environnement virtuel 
python -m venv mon_environnement

windows : mon_environnement\Scripts\activate pour l'activer
linux : source env/bin/activate

# Installer les packages
pip install -r requirements.txt

# Tesseract 
Installez tesseract, 5.5 de préférence, depuis le github officiel de Tesseract. Une fois fait, ajoutez le fichier digitdetector.traineddata que vous trouverez dans le dossier "model" au dossier tessdata de tesseract généralement situé à

Windows : C:\Program Files (x86)\Tesseract-OCR\tessdata pour Windows
Linux : /usr/share/tesseract/4/tessdata/

# Tkinter
Il s'agit du module python qui permet d'ouvrir des interfaces graphiques pour charger les fichiers notamment. Sur windows, il est déjà inclus lors de l'installation de python
mais pour linux, lancez la commande : sudo apt-get install python3-tk
NB : Attention au chemin du fichier que vous ouvrez lorsque vous changez de méthode, il se peut que vous lanciez le pipeline de "./tesseract_ocr" mais que vous choisissez un fichier
d'un autre dossier.


# Exécuter le programme
Dans le dossier pipelines, vous trouverez les trois versions d'extraction des données. Exécutez avec python le fichier pipeline.py, de manière graphique grâce à un IDE ou en exécutant la commande dans un terminal python pipelines\nom_de_la_methode_voulue\pipeline.py

Une interface graphique s'ouvrira, choisissez le document que vous souhaitez traiter. Vous obtiendrez une version transformée avec la méthode correspondante dans le dossier pdf_folder (nom du fichier précédé de output_x.pdf) et un fichier csv ou Excell (textract ou tesseract) dans le dossier csv_tables.

