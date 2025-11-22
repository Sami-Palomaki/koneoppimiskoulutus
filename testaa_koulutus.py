import cv2
import pytesseract
import glob
import os
import numpy as np
from ultralytics import YOLO

# Tesseract-polku
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# YOLO-malli - viides kerta todensanoo
model = YOLO("runs/detect/train5/weights/last.pt")

input_folder = "kuvat/"
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

# Poistaa sinisen EU/FIN-kaistaleen värintunnistuksen perusteella
def remove_eu_band(plate_img):

    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    # Sinisen raja-arvot
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Tätä käytetään sitten löytämään, missä kohtaa EU/FIN-sininen kaista loppuu, jotta sen voi leikata pois
    blue_per_column = mask.mean(axis=0)

    # Etsitään raja missä sininen loppuu
    threshold = 10  # alle 10 = ei enää sinistä
    blue_columns = np.where(blue_per_column > threshold)[0]

    if len(blue_columns) == 0:
        # Ei sinistä, joten ei leikata mitään
        return plate_img

    end_of_blue = blue_columns[-1]  # viimeinen sininen sarake

    # Leikkaa EU-kaistale
    return plate_img[:, end_of_blue + 2:]  # +2 pikseliä


print(f"Löydetty {len(image_paths)} JPG-kuvaa.\n")

for img_path in image_paths:
    print(f"\nTestataan kuva: {img_path}")

    img = cv2.imread(img_path)
    results = model.predict(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = img[y1:y2, x1:x2]

            # Poista vain EU-kaistale, ei kirjaimia
            clean_plate = remove_eu_band(plate_img)
            if clean_plate is None or clean_plate.size == 0:        # Jos tyhjä kuva, skippaa
                print("WARNING: EU-kaistan poiston jälkeen kuva on tyhjä")
                continue  # ohita tämä detekti

            gray = cv2.cvtColor(clean_plate, cv2.COLOR_BGR2GRAY) # Tämä muuntaa kuvan harmaasävyiseksi
            gray = cv2.resize(gray, None, fx=2, fy=2) # Tämä kasvattaa kuvan koon kaksinkertaiseksi

            text = pytesseract.image_to_string(
                gray,
                config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ).strip()

            print(" → Kilven teksti:", text)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Piirtää vihreän laatikon tunnistetun kilven ympärille
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)                          # Kirjoittaa tekstin kilven yläpuolelle

    cv2.imshow("Tulokset", img)
    cv2.waitKey(2500)
