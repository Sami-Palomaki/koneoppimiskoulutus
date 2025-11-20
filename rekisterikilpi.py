import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
import os

# --- 1. Tesseractin polku (Windows) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- 2. YOLOv8-malli (voit käyttää esim. pretrained COCO-mallia tai omaa ANPR-mallia) ---
# Tässä käytetään esimerkkimallia 'yolov8n.pt'
model = YOLO('yolov8n.pt')  # pieni ja nopea malli, mutta voit käyttää custom-mallia

# --- 3. Kansio kuvia varten ---
folder_path = "kuvat"
output_folder = "tulokset"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 4. Käy kaikki kuvat läpi ---
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img_display = img.copy()

        # --- 5. YOLO tunnistaa kilvet ---
        results = model.predict(img)  # oletetaan, että malli tunnistaa kilvet

        for r in results:
            boxes = r.boxes  # kaikki löydetyt objektit
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = img[y1:y2, x1:x2]

                # --- 6. OCR: Tesseract ---
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                text = pytesseract.image_to_string(gray, config='--psm 8').strip().replace("\n", "")
                
                print(f"{filename}: {text}")

                # Piirrä laatikko ja teksti
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_display, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tallenna tulos
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_display)

print("Kaikki kuvat käsitelty ja tallennettu kansioon 'tulokset'.")
