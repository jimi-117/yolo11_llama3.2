from pydantic import BaseModel
from typing import Optional, List

class TrainingConfig(BaseModel):
    project_name: str
    dataset_path: str
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: Optional[str] = None
    workers: int = 8
    pretrained: bool = True
    clearml_project: str
    clearml_task_name: str
    tags: List[str] = []