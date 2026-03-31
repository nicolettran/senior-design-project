from ultralytics import YOLO

model = YOLO("vegetation_model.pt")
model.export(format="ncnn")

# Convert YOLOv11n model to NCNN for better performance on Pi