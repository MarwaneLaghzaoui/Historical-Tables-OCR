# import cv2
# import csv
# import fitz
# import pytesseract
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# from pdf2image import convert_from_path

# model = YOLO("pipelines/yolo_ocr/model/best.pt")

# pdf_path = r"D://GitHub//HOCR//pipelines//yolo_ocr//pdf_folder//1.pdf"
# output_csv = r"D://GitHub//HOCR//pipelines//yolo_ocr//csv_tables//output.csv"

# pdf_document = fitz.open(pdf_path)

# # Ouvrir un fichier CSV pour écrire les résultats
# with open(output_csv, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Page", "Classe", "X1", "Y1", "X2", "Y2"])  # En-tête du CSV

#     for page_num in range(len(pdf_document)):
#         print(f"Traitement de la page {page_num + 1}")

#         # Récupérer l'image de la page en format numpy
#         page = pdf_document[page_num]
#         pix = page.get_pixmap()  # Augmenter le DPI pour meilleure qualité
#         img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

#         # Convertir en format OpenCV
#         image_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         # Détection des objets avec YOLO
#         results = model(image_cv)

#         # Enregistrer les détections
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Découper la région de l'image contenant le chiffre détecté
#                 roi = image_cv[y1:y2, x1:x2]

#                 # Utiliser Tesseract pour extraire le texte de la région
#                 text = pytesseract.image_to_string(roi, config='--oem 3 --psm 6')  # PSM 6 pour mode "assume a single uniform block of text"
                
#                 # Si le texte contient un chiffre, l'ajouter au CSV
#                 if text.isdigit():  # Vérifier si le texte extrait est un chiffre
#                     writer.writerow([page_num + 1, text, x1, y1, x2, y2])

#         # Sauvegarder l'image annotée pour vérification
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.imwrite(f"D:\GitHub\HOCR\pipelines\yolo_ocr\output_images\page_{page_num + 1}.png", image_cv)

# image_display = cv2.resize(image_cv, (1080, 1920))  # Ajuste selon tes besoins
# cv2.imshow("Image traitée", image_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print("Détection terminée. Résultats enregistrés dans", output_csv)













# import cv2
# import csv
# import fitz
# import pytesseract
# import numpy as np
# from PIL import Image
# from pdf2image import convert_from_path


# from ultralyticsplus import YOLO, render_result

# # Load the model
# model = YOLO('foduucom/table-detection-and-extraction')

# # Set model parameters
# model.overrides['conf'] = 0.25  # NMS confidence threshold
# model.overrides['iou'] = 0.45  # NMS IoU threshold
# model.overrides['agnostic_nms'] = False  # NMS class-agnostic
# model.overrides['max_det'] = 1000  # maximum number of detections per image


# pdf_path = r"D://GitHub//HOCR//pipelines//yolo_ocr//pdf_folder//1.pdf"
# output_csv = r"D://GitHub//HOCR//pipelines//yolo_ocr//csv_tables//output.csv"

# pdf_document = fitz.open(pdf_path)

# page = pdf_document[0]
# pix = page.get_pixmap()  # Augmenter le DPI pour meilleure qualité
# img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

# # Convertir en format OpenCV
# image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # Set the image file


# # Perform inference
# results = model.predict(image)

# # Observe the results
# print(results[0].boxes)

# # Render the results
# render = render_result(model=model, image=image, result=results[0])
# render.show()



import cv2
import csv
import fitz
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
 
pdf_path = r"D://GitHub//HOCR//pipelines//yolo_ocr//pdf_folder//1.pdf"
output_csv = r"D://GitHub//HOCR//pipelines//yolo_ocr//csv_tables//output.csv"

pdf_document = fitz.open(pdf_path)

page = pdf_document[0]
pix = page.get_pixmap()  # Augmenter le DPI pour meilleure qualité
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

# Convertir en format OpenCV
image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

sv.plot_image(image)

model = get_model(model_id="yolov8n-640")
results = model.infer(image)[0]
results = sv.Detections.from_inference(results)
annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, results)
annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
annotated_image = annotator.annotate(annotated_image, results)
sv.plot_image(annotated_image)