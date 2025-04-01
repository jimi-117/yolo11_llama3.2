from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models import FeedbackData
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import json

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global model instance
model = YOLO("yolo11x.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.5):
    # load image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # prediction
    start_time = time.time()
    results = model(img, conf=confidence, verbose=False)
    inference_time = time.time() - start_time

    # process results
    detections = []
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            detection = {
                "id": i,
                "class_name": result.names[box.cls[0].item()],
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].tolist()
            }
            detections.append(detection)

    return {
        "inference_time": inference_time,
        "detections": detections,
        "annotated_image": results[0].plot()
    }

@app.post("/feedback/")
async def save_feedback(feedback: FeedbackData):
    feedback_dir = Path("feedback")
    feedback_dir.mkdir(exist_ok=True)
    
    timestamp = feedback.timestamp.strftime("%Y%m%d_%H%M%S")
    feedback_file = feedback_dir / f"feedback_{timestamp}.json"
    
    with open(feedback_file, "w", encoding='utf-8') as f:
        json.dump(feedback.dict(), f, indent=4, ensure_ascii=False)
    
    return {"status": "success"}