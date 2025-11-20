import os
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import re
import csv
import sys

# ------------------- ASETUKSET -------------------
# Polut / mallit
MODEL_PATH = "runs/detect/train5/weights/best.pt"  # jos sinulla on koulutettu ANPR-malli; muuten esim "yolov8n.pt"
INPUT_FOLDER = "kuvat"
OUTPUT_FOLDER = "tulokset"
CSV_OUTPUT = os.path.join(OUTPUT_FOLDER, "tunnistukset.csv")

# Tesseract polku (Windows-esimerkki) — muokkaa tarvittaessa tai poista rivin kommentti
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR-asetukset
TESSERACT_PSM = "7"  # 7 = treat the image as a single text line (hyvä kilville)
TESSERACT_OEM = "3"  # 3 = Default, based on what is available
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"  # hyväksytyt merkit

# YOLO-asetukset
YOLO_CONF_THRESH = 0.25  # min confidence
YOLO_IOU = 0.45

# Esikäsittely
REMOVE_LEFT_PERCENT = 0.18  # kuinka paljon vasemmasta reunasta leikataan pois (EU-tunniste). 0 = ei leikata
SCALE_UP = 2.5             # kuinka paljon kilpea kasvatetaan ennen OCR:ää
BLUR_KERNEL = (3, 3)

# Regex validointia varten (yksinkertainen, yleisluontoinen)
# Etsii esim. ABC-123, ABC123, AB-1234, 123-ABC yms. Mukautettava maan formaatin mukaan.
PLATE_REGEX = re.compile(r"[A-Z0-9]{2,4}[- ]?[A-Z0-9]{1,4}")

# Common char fixes (jos haluat korvata esim. O->0 jne.)
COMMON_REPLACES = {
    'O': '0',
    'Q': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8'
}
# --------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ladataan YOLO-malli
print("Ladataan YOLO-mallia:", MODEL_PATH)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Mallin lataus epäonnistui:", e)
    print("Yritetään ladata yleismallia 'yolov8n.pt'...")
    model = YOLO("yolov8n.pt")

# Apufunktiot
def expand_box(x1, y1, x2, y2, img_w, img_h, pad=0.02):
    """Lisää vähän paddingia laatikkoon suhteessa kuvan kokoon."""
    w = x2 - x1
    h = y2 - y1
    dx = int(w * pad)
    dy = int(h * pad)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w - 1, x2 + dx)
    ny2 = min(img_h - 1, y2 + dy)
    return nx1, ny1, nx2, ny2

def preprocess_plate(plate_img, remove_left_percent=REMOVE_LEFT_PERCENT, scale_up=SCALE_UP):
    """Esikäsittely ennen OCR: poista vasen osa (EU), skaalaa, blur, threshold."""
    if plate_img is None or plate_img.size == 0:
        return None
    h, w = plate_img.shape[:2]
    # Poista vasen osa (esim. EU-alue)
    if remove_left_percent and remove_left_percent > 0:
        cut = int(w * remove_left_percent)
        plate_img = plate_img[:, cut:]
        h, w = plate_img.shape[:2]
    # harmaasävy
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    # kasvata resoluutiota
    new_w = int(w * scale_up)
    new_h = int(h * scale_up)
    if new_w <= 0 or new_h <= 0:
        return None
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # blur ja threshold
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    # adapt. threshold toimii usein parhaiten vaihtelevassa valossa
    try:
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 2)
    except Exception:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pieni morfologinen avaaminen poistaa pienet kohinat
    kernel = np.ones((2,2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return thr

def ocr_plate(image_for_ocr):
    """Kutsu Tesseractia whitelistillä ja antamalla psm/oem."""
    if image_for_ocr is None:
        return ""
    config = f"-c tessedit_char_whitelist={WHITELIST} --oem {TESSERACT_OEM} --psm {TESSERACT_PSM}"
    text = pytesseract.image_to_string(image_for_ocr, config=config)
    if not text:
        return ""
    text = text.strip().upper()
    # Poista ei-toivotut merkit
    text = re.sub(r'[^A-Z0-9\- ]', '', text)
    return text

def normalize_text(text):
    """Korjaa yleisimpiä OCR-virheitä tunnistuksessa (mahdollistaa Z->2 ym.)."""
    if not text:
        return text
    # Poista ylimääräiset välilyönnit
    t = text.replace(" ", "").replace("\n", "").strip()
    # Korjaa yleisimpiä virheitä (vain jos merkki on kirjain tai numero)
    corrected = []
    for ch in t:
        if ch in COMMON_REPLACES:
            corrected.append(COMMON_REPLACES[ch])
        else:
            corrected.append(ch)
    return "".join(corrected)

def find_plate_by_regex(text):
    """Etsi regexillä mahdollinen plate-string."""
    if not text:
        return None
    matches = PLATE_REGEX.findall(text)
    if not matches:
        return None
    # Palauta pisin/sopivin match
    matches = sorted(matches, key=lambda s: len(s), reverse=True)
    return matches[0]

# CSV head
with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'bbox', 'ocr_raw', 'ocr_normalized', 'final_plate'])

# Käy kansio läpi
files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not files:
    print("Ei kuvia kansiossa:", INPUT_FOLDER)
    sys.exit(1)

for filename in files:
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)
    if img is None:
        print("Ei voitu lukea kuvaa:", img_path)
        continue
    img_h, img_w = img.shape[:2]
    img_disp = img.copy()

    # YOLO predict (konf ja iou voi laittaa paramina)
    try:
        results = model.predict(source=img, conf=YOLO_CONF_THRESH, iou=YOLO_IOU, verbose=False)
    except Exception as e:
        print("YOLO predict error:", e)
        results = []

    detections = []
    for r in results:
        # r.boxes sisältää bounding boxit
        for box in r.boxes:
            # box.xyxy voi olla tensor -> np
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
            x1, y1, x2, y2 = map(int, xyxy)
            # Lisää hieman paddingia
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, img_w, img_h, pad=0.03)
            plate_crop = img[y1:y2, x1:x2].copy()

            # Esiprosessointi
            pre = preprocess_plate(plate_crop)
            ocr_raw = ocr_plate(pre)
            ocr_norm = normalize_text(ocr_raw)
            # Etsi regex-ehdokas
            candidate = find_plate_by_regex(ocr_norm)
            # jos ei löytynyt, käytä ocr_norm suoraan
            final_plate = candidate if candidate else ocr_norm

            # tallenna tulos listaan + piirrä kuvaan
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'ocr_raw': ocr_raw,
                'ocr_normalized': ocr_norm,
                'final': final_plate
            })

            # Piirrä bounding box ja teksti
            label = final_plate if final_plate else "UNKNOWN"
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_disp, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tallenna kuva ja kirjoita CSV-rivit
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(out_path, img_disp)

    # Kirjoita CSV:lle kaikki detektit yhdelle riville (tai useita rivejä)
    with open(CSV_OUTPUT, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not detections:
            writer.writerow([filename, "", "", "", ""])
            print(f"{filename}: Ei detektejä")
        else:
            for d in detections:
                writer.writerow([filename, f"{d['bbox']}", d['ocr_raw'], d['ocr_normalized'], d['final']])
            print(f"{filename}: {len(detections)} detektiota. Ensimmäinen:", detections[0]['final'])

print("Käsittely valmis. Tulokset:", OUTPUT_FOLDER, "CSV:", CSV_OUTPUT)
