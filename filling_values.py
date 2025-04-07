import os
import subprocess

def generate_lstmf_from_box(input_folder):
    """Génère les fichiers .lstmf à partir des fichiers .box dans le dossier donné."""
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".box"):
                box_path = os.path.join(root, file)
                tiff_path = os.path.splitext(box_path)[0] + ".tiff"  # Cherche l'image TIFF correspondante
                lstmf_path = os.path.splitext(box_path)[0] + ".lstmf"
                
                # Vérifie que l'image TIFF existe avant de procéder
                if os.path.exists(tiff_path):
                    command = [
                        "tesseract", tiff_path, lstmf_path, "batch.nochop", "makebox"
                    ]
                    
                    print(f"Generating .lstmf from: {box_path}")
                    try:
                        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"Generated: {lstmf_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error generating .lstmf for {tiff_path}: {e.stderr.decode()}")
                else:
                    print(f"Image not found for: {box_path}")

if __name__ == "__main__":
    base_folders = ["dataset_tiff", "fintuning_cells", "cellules_recadrees"]
    
    for folder in base_folders:
        if os.path.exists(folder):
            generate_lstmf_from_box(folder)
        else:
            print(f"Folder not found: {folder}")
