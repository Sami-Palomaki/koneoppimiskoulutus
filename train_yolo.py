from ultralytics import YOLO

# Lataa pieni esikoulutettu malli (nopea)
model = YOLO("yolov8n.pt")

# Kouluta omaan data.yaml
results = model.train(
    data="D:/TypeScript_Haaga-Helia/koneoppimiskoulutus/kuvat/data.yaml",
    epochs=50,        # voit nostaa esim. 100 myöhemmin
    imgsz=416,        # kuvan koko
    batch=8,         # batch-koko (GPU nopeuttaa)
    device='cpu'
)

# Malli tallentuu automaattisesti runs/train/weights/best.pt
