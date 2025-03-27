from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="/Users/aryanvohra/Documents/Object_detection/Configurations/COCO.yaml", epochs=100, imgsz=640)