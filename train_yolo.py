from ultralytics import YOLO

# Pieni esikoulutettu malli
model = YOLO("yolov8n.pt")

# Kouluta omaan data.yaml
results = model.train(
    data="D:/TypeScript_Haaga-Helia/koneoppimiskoulutus/kuvat/data.yaml",
    epochs=50,
    imgsz=416,
    batch=8,
    device='cpu'
)

# Malli tallentuu automaattisesti runs/detect/trainX/weights/best.pt
