from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models import FeedbackData
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import json
import base64
from datetime import datetime

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
    try:
        # read image file
        contents = await file.read()
        if not contents:
            return {"error": "Empty file received"}

        # transform byte data to image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Failed to decode image"}

        # predict objects
        start_time = time.time()
        results = model(img, conf=confidence, verbose=False)
        inference_time = time.time() - start_time

        # process results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for i, box in enumerate(boxes):
                detection = {
                    "id": i,
                    "class_name": result.names[box.cls[0].item()],
                    "confidence": float(box.conf[0].item()),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)

            # generate annotated image from detection result
            annotated_img = result.plot()
        else:
            # 検出結果がない場合は、入力画像そのままを返す等の対応
            annotated_img = img  # または、必要なら別途アノテーション処理を追加

        success, buffer = cv2.imencode('.jpg', annotated_img)
        if not success:
            return {"error": "Failed to encode result image"}
        img_str = base64.b64encode(buffer).decode()

        return {
            "inference_time": inference_time,
            "detections": detections,
            "annotated_image": img_str
        }
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@app.post("/feedback/")
async def save_feedback(feedback: FeedbackData):
    feedback_dir = Path("feedback")
    feedback_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_file = feedback_dir / f"feedback_{timestamp}.json"
    
    feedback_dict = feedback.model_dump()
    
    with open(feedback_file, "w", encoding='utf-8') as f:
        json.dump(feedback_dict, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)
    
    return {"status": "success"}