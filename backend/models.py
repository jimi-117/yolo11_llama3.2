from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Detection(BaseModel):
    id: int
    class_name: str
    confidence: float
    feedback: str
    bbox: List[float]

class FeedbackData(BaseModel):
    timestamp: datetime
    image_name: str
    inference_time: float
    detections: List[Detection]