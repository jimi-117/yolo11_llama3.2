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

from clearml import Task, Dataset
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


# トレーニング状態を保持するグローバル変数a# main.py (またはアプリケーションのエントリーポイントとなるファイル)

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader # APIキーヘッダー用
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio
from typing import Dict, List, Optional

from pydantic import BaseModel, Field # Placeholder models
from loguru import logger # Use loguru for better logging

# --- Placeholder Pydantic Models ---
# (本来は models/detection.py や models/training.py に定義)
class FeedbackData(BaseModel):
    image_base64: Optional[str] = None # Example field, adjust as needed
    corrected_labels: List[dict] # Example field
    user_comment: Optional[str] = None

class TrainingConfig(BaseModel):
    clearml_project: str = "QoS Project"
    clearml_task_name: str = "YOLO Training"
    tags: List[str] = ["yolo", "training"]
    project_name: str # Directory name for saving results locally
    dataset_path: str # Path to dataset (e.g., from ClearML Dataset)
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    workers: int = 8
    device: Optional[str] = None # e.g., "0" for GPU 0, "cpu", or None for auto
    pretrained: bool = True # Start from pretrained weights

# --- Configuration ---
# Load .env file (assuming it's in the same directory or parent)
load_dotenv()

# Basic logging setup
logger.add(lambda msg: print(msg, end=""), level=os.getenv("LOG_LEVEL", "INFO").upper())

# API Key settings (should ideally be in a config module)
API_KEY_NAME = "X-API-KEY"
ALLOWED_API_KEYS_JSON = os.getenv("ALLOWED_API_KEYS_JSON", '[]')
try:
    ALLOWED_API_KEYS = json.loads(ALLOWED_API_KEYS_JSON)
    if not isinstance(ALLOWED_API_KEYS, list):
        logger.warning("ALLOWED_API_KEYS_JSON is not a valid JSON list. No API keys allowed.")
        ALLOWED_API_KEYS = []
except json.JSONDecodeError:
    logger.warning("Could not parse ALLOWED_API_KEYS_JSON. No API keys allowed.")
    ALLOWED_API_KEYS = []

# Security dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # auto_error=False for custom message

async def verify_api_key(key: str = Depends(api_key_header)):
    """Dependency to verify the API key."""
    if not key:
        logger.warning("API Key missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing",
        )
    if not ALLOWED_API_KEYS:
        logger.error("API Key authentication is not configured on the server (no keys loaded).")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API Key authentication is not configured.",
        )
    if key not in ALLOWED_API_KEYS:
        logger.warning(f"Invalid API Key received: '{key[:5]}...'") # Log only prefix
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    logger.debug("Valid API Key received.")
    return key # Return key if valid (can be used later if needed)


# --- FastAPI App ---
app = FastAPI(title="QoS AI Backend")

# CORS settings (Restrict in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Allow all origins - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# Load the YOLO model globally (consider lazy loading or lifespan event)
try:
    MODEL_PATH = "yolo11x.pt" # Consider making this configurable
    if not Path(MODEL_PATH).exists():
         logger.error(f"Model file not found at {MODEL_PATH}")
         # Exit or use a placeholder? For now, let it raise error later.
         model = None
    else:
        model = YOLO(MODEL_PATH)
        logger.info(f"YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None # Ensure model is None if loading fails

# --- Endpoints ---

@app.post("/detect/", dependencies=[Depends(verify_api_key)]) # Apply authentication
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.5):
    """Detect objects in an uploaded image using YOLO model."""
    if model is None:
         raise HTTPException(status_code=503, detail="YOLO model is not loaded.")

    logger.info(f"Received detection request for file: {file.filename}")
    try:
        # Read image file
        contents = await file.read()
        if not contents:
            logger.warning("Empty file received for detection.")
            raise HTTPException(status_code=400, detail="Empty file received")

        # Transform byte data to image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to decode image: {file.filename}")
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Predict objects
        start_time = time.time()
        results = model(img, conf=confidence, verbose=False) # verbose=False is good
        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.4f} seconds")

        # Process results
        detections = []
        annotated_img = img # Default to original if no results
        if results and len(results) > 0:
            result = results[0] # Assuming single image result
            boxes = result.boxes
            class_names = result.names # Get class names map from the result

            for i, box in enumerate(boxes):
                class_id = box.cls[0].item()
                detection = {
                    "id": i,
                    "class_name": class_names.get(class_id, f"unknown_class_{int(class_id)}"), # Use .get for safety
                    "confidence": float(box.conf[0].item()),
                    "bbox": box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                }
                detections.append(detection)

            # Generate annotated image
            annotated_img = result.plot() # Use YOLO's plotting function
            logger.info(f"Detected {len(detections)} objects.")
        else:
             logger.info("No objects detected.")


        # Encode result image to base64
        success, buffer = cv2.imencode('.jpg', annotated_img)
        if not success:
            logger.error("Failed to encode result image to JPEG.")
            raise HTTPException(status_code=500, detail="Failed to encode result image")
        img_str = base64.b64encode(buffer).decode()

        return {
            "inference_time": inference_time,
            "detections": detections,
            "annotated_image": img_str
        }
    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI's HTTP exceptions
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
         await file.close() # Ensure file is closed

# Custom JSON Encoder for datetime
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@app.post("/feedback/", dependencies=[Depends(verify_api_key)]) # Apply authentication
async def save_feedback(feedback: FeedbackData):
    """
    Save feedback data to a local JSON file.
    NOTE: It's recommended to move this logic to QoS_DATA and use a database.
    """
    logger.info("Received feedback data.")
    feedback_dir = Path("feedback") # Make configurable
    feedback_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Add microseconds for uniqueness
    feedback_file = feedback_dir / f"feedback_{timestamp}.json"

    try:
        # Use Pydantic's serialization which handles datetime better if possible
        # feedback_dict = feedback.model_dump(mode='json') # Pydantic V2
        feedback_dict = feedback.dict() # Pydantic V1 fallback

        with open(feedback_file, "w", encoding='utf-8') as f:
            # Use custom encoder only if Pydantic doesn't handle datetime
            json.dump(feedback_dict, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)
        logger.info(f"Feedback saved to {feedback_file}")
        return {"status": "success", "file": str(feedback_file)}
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# --- Model Training Section ---

# ClearML Credentials Setup (Done once at startup ideally)
try:
    Task.set_credentials(
        api_host=os.getenv("CLEARML_API_HOST"),
        web_host=os.getenv("CLEARML_WEB_HOST"),
        files_host=os.getenv("CLEARML_FILES_HOST"),
        key=os.getenv("CLEARML_API_ACCESS_KEY"),
        secret=os.getenv("CLEARML_API_SECRET_KEY")
    )
    logger.info("ClearML credentials set from environment variables (if provided).")
except Exception as e:
    logger.warning(f"Could not set ClearML credentials from env vars: {e}. SDK might use config file.")


# Training state (WARNING: In-memory dict is not suitable for production)
training_tasks: Dict[str, dict] = {}

async def _run_training_task(config: TrainingConfig, project_dir: Path, task: Task):
    """Helper function to run training in the background."""
    task_id = task.id
    project_name = config.project_name
    logger.info(f"Starting training task {task_id} for project {project_name} in background.")
    try:
        # Training hyperparameters
        train_args = {
            "data": str(Path(config.dataset_path) / "data.yaml"), # data.yaml path
            "epochs": config.epochs,
            "batch": config.batch_size,
            "imgsz": config.imgsz,
            "workers": config.workers,
            "device": config.device if config.device else "auto", # Handle None
            "pretrained": config.pretrained,
            "project": str(project_dir.parent), # YOLO saves to project/name
            "name": project_dir.name, # Use project_name as YOLO's run name
        }
        logger.info(f"YOLO train args: {train_args}")
        task.connect(train_args) # Log parameters to ClearML

        # Create a new YOLO instance for training to avoid potential conflicts
        # Use the same base model specified globally or in config
        train_model = YOLO(MODEL_PATH if config.pretrained else "yolov8n.yaml") # Adjust base model if needed

        # Execute training
        # Note: model.train() might be blocking depending on the ultralytics version/setup.
        # If it blocks the asyncio loop, it needs to be run in a separate thread or process.
        # For now, assume it yields control appropriately or runs quickly enough for demo.
        results = train_model.train(**train_args)
        logger.info(f"Training task {task_id} completed.")

        # Find best/last weights
        train_run_dir = project_dir / project_dir.name # Default YOLO output: project/name
        best_pt_path = train_run_dir / "weights" / "best.pt"
        last_pt_path = train_run_dir / "weights" / "last.pt"

        # Upload artifacts to ClearML
        if best_pt_path.exists():
            task.upload_artifact("best_model", artifact_object=str(best_pt_path))
            logger.info(f"Uploaded best model artifact: {best_pt_path}")
        else:
             logger.warning(f"best.pt not found in {best_pt_path.parent}")

        if last_pt_path.exists():
            task.upload_artifact("last_model", artifact_object=str(last_pt_path))
            logger.info(f"Uploaded last model artifact: {last_pt_path}")
        else:
             logger.warning(f"last.pt not found in {last_pt_path.parent}")


        # Update global status (not robust)
        if project_name in training_tasks:
            training_tasks[project_name]["status"] = "completed"
            training_tasks[project_name]["completed_at"] = datetime.now().isoformat()
            # Storing full results might be too large, store summary if needed
            # training_tasks[project_name]["results_summary"] = {"map50": results.maps.get('metrics/mAP50-B')}
        task.close()

    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}", exc_info=True)
        if project_name in training_tasks:
            training_tasks[project_name]["status"] = "failed"
            training_tasks[project_name]["error"] = str(e)
        # Mark ClearML task as failed
        if task:
            task.mark_failed(status_reason=str(e))
            task.close()


@app.post("/train/", dependencies=[Depends(verify_api_key)]) # Apply authentication
async def start_training(config: TrainingConfig):
    """Start a new YOLO training task in the background."""
    logger.info(f"Received training request for project: {config.project_name}")
    if config.project_name in training_tasks and training_tasks[config.project_name]["status"] == "running":
        logger.warning(f"Training for project '{config.project_name}' is already running.")
        raise HTTPException(status_code=409, detail=f"Training for project '{config.project_name}' is already running.")

    try:
        # Initialize ClearML Task
        task = Task.init(
            project_name=config.clearml_project,
            task_name=config.clearml_task_name or f"Training_{config.project_name}", # Default name
            tags=config.tags,
            # auto_connect_frameworks=False # Disable auto logging if manual logging is preferred
        )
        logger.info(f"ClearML Task initialized: {task.id}")

        # Create local project directory for YOLO output
        # Be careful with concurrent runs if dirs aren't unique per run
        project_dir = Path("projects") / config.project_name # YOLO output dir = project/name
        # project_dir.mkdir(parents=True, exist_ok=True) # YOLO creates this

        # Validate dataset path
        dataset_path = Path(config.dataset_path)
        if not (dataset_path.exists() and (dataset_path / "data.yaml").exists()):
             logger.error(f"Dataset path or data.yaml not found: {dataset_path}")
             task.mark_failed(status_reason="Dataset path or data.yaml not found")
             task.close()
             raise HTTPException(status_code=400, detail="Dataset path or data.yaml not found")

        # Register task status (in-memory dict - not robust)
        training_tasks[config.project_name] = {
            "status": "starting",
            "started_at": datetime.now().isoformat(),
            "config": config.model_dump(),
            "clearml_task_id": task.id,
            "error": None,
            "completed_at": None,
        }

        # Run training in background
        # Using asyncio.create_task assumes model.train() is awaitable or non-blocking enough.
        # If model.train() is blocking, use BackgroundTasks or run_in_executor.
        asyncio.create_task(_run_training_task(config, project_dir, task))
        training_tasks[config.project_name]["status"] = "running" # Update status after task creation

        return {
            "status": "training_started",
            "project_name": config.project_name,
            "clearml_task_id": task.id
        }

    except Exception as e:
        logger.error(f"Training setup failed: {e}", exc_info=True)
        # Clean up task status if setup failed before background task started
        if config.project_name in training_tasks and training_tasks[config.project_name]["status"] == "starting":
             del training_tasks[config.project_name]
        raise HTTPException(status_code=500, detail=f"Training setup failed: {str(e)}")


@app.get("/train/{project_name}/status", dependencies=[Depends(verify_api_key)]) # Apply authentication
async def get_training_status(project_name: str):
    """Get the status of a specific training task."""
    logger.debug(f"Request received for training status: {project_name}")
    if project_name not in training_tasks:
        logger.warning(f"Training project not found: {project_name}")
        raise HTTPException(status_code=404, detail="Training project not found")

    return training_tasks[project_name]

# --- Optional: Root endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to QoS AI Backend"}

# --- Uvicorn runner for local dev ---
# (Remove or guard this if running via Docker CMD)
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting uvicorn server for development...")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level=os.getenv("LOG_LEVEL", "info").lower())


# --- API認証設定 ---
# このAPIへのアクセスを許可するAPIキーのリスト (JSON文字列形式)


# --- ClearML SDK認証情報 (推奨: 環境変数または clearml.conf で設定) ---
# CLEARML_API_ACCESS_KEY="YOUR_CLEARML_ACCESS_KEY"
# CLEARML_API_SECRET_KEY="YOUR_CLEARML_SECRET_KEY"
# CLEARML_API_HOST="YOUR_CLEARML_API_SERVER_URL" # Self-hostedの場合

# --- ロギング設定 ---
LOG_LEVEL="INFO"


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