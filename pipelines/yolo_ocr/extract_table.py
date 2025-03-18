import numpy as np
import pandas as pd
import fitz
import cv2
import supervision as sv

import pytesseract
from pytesseract import Output

from ultralyticsplus import YOLO, render_result
from PIL import Image


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# pdf_path = r"D://GitHub//HOCR//pipelines//yolo_ocr//pdf_folder//1.pdf"
# csv_path = r"D://GitHub//HOCR//pipelines//yolo_ocr//csv_tables//output.csv"

def extract_tables_from_pdf(pdf_path, csv_path):

    pdf_document = fitz.open(pdf_path)

    page = pdf_document[0]
    pix = page.get_pixmap()  # Augmenter le DPI pour meilleure qualité
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Convertir en format OpenCV
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # sv.plot_image(image)

    model = YOLO('keremberke/yolov8m-table-extraction')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # perform inference
    results = model.predict(img)

    # observe results
    print('Boxes: ', results[0].boxes)
    render = render_result(model=model, image=img, result=results[0])
    # sv.plot_image(render)

    x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.cpu().numpy()[0])
    img = np.array(image)
    #cropping
    cropped_image = img[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped_image)


    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract\tesseract.exe' #Path to tesseract.exe because adding to PATH may not work

    psm = 3 # 3 is also great for only getting numbers and ignore labels

    print(f"Testing with PSM: {psm}")
    extracted_text = pytesseract.image_to_string(cropped_image, config=f"--psm {psm} --oem 3")
    # Transformation en format CSV
    lines = extracted_text.strip().split('\n')
    data = [line.split() for line in lines]

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, header=False, encoding='utf-8')

    print(f"Extraction terminée. Résultats enregistrés dans {csv_path}")

    # sv.plot_image(cropped_image)

