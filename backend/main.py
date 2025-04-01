from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models.detection import FeedbackData
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


###### model training #######
from dotenv import load_dotenv
import os

load_dotenv()

from clearml import Task
import asyncio
from typing import Dict
from models.training import TrainingConfig
from pathlib import Path
import shutil

# ClearML設定を初期化
Task.set_credentials(
    api_host=os.getenv("CLEARML_API_HOST"),
    web_host=os.getenv("CLEARML_WEB_HOST"),
    files_host=os.getenv("CLEARML_FILES_HOST"),
    key=os.getenv("CLEARML_API_ACCESS_KEY"),
    secret=os.getenv("CLEARML_API_SECRET_KEY")
)


# トレーニング状態を保持するグローバル変数
training_tasks: Dict[str, dict] = {}

@app.post("/train/")
async def start_training(config: TrainingConfig):
    try:
        # ClearMLタスクの初期化
        task = Task.init(
            project_name=config.clearml_project,
            task_name=config.clearml_task_name,
            tags=config.tags
        )

        # プロジェクトディレクトリの作成
        project_dir = Path("projects") / config.project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # データセットの検証
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            return {"error": "Dataset path does not exist"}

        # トレーニングハイパーパラメータの設定とログ
        train_args = {
            "data": str(dataset_path / "data.yaml"),
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.imgsz,
            "workers": config.workers,
            "device": config.device or "auto",
            "pretrained": config.pretrained,
            "project": str(project_dir),
            "name": "train",
        }
        
        # ClearMLにパラメータを記録
        task.connect(train_args)

        async def train_model():
            try:
                # 新しいYOLOインスタンスを作成
                train_model = YOLO("yolo11x.pt")
                
                # トレーニング実行
                results = train_model.train(**train_args)
                
                # メトリクスをClearMLに記録
                task.upload_artifact("best_model", str(project_dir / "train" / "weights" / "best.pt"))
                task.upload_artifact("last_model", str(project_dir / "train" / "weights" / "last.pt"))
                
                # トレーニング完了の記録
                training_tasks[config.project_name]["status"] = "completed"
                training_tasks[config.project_name]["completed_at"] = datetime.now().isoformat()
                training_tasks[config.project_name]["results"] = results
                
                task.close()
                
            except Exception as e:
                training_tasks[config.project_name]["status"] = "failed"
                training_tasks[config.project_name]["error"] = str(e)
                task.mark_failed(str(e))

        # トレーニングタスクの登録
        training_tasks[config.project_name] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "config": config.model_dump(),
            "clearml_task_id": task.id
        }

        # バックグラウンドでトレーニングを実行
        asyncio.create_task(train_model())

        return {
            "status": "training_started",
            "project_name": config.project_name,
            "clearml_task_id": task.id
        }

    except Exception as e:
        return {"error": f"Training setup failed: {str(e)}"}

@app.get("/train/{project_name}/status")
async def get_training_status(project_name: str):
    if project_name not in training_tasks:
        return {"error": "Training project not found"}
    
    return training_tasks[project_name]