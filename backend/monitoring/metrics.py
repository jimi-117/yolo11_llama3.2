from prometheus_client import Counter, Histogram, Gauge
from typing import Dict

# set ClearML metrics
class MLMetrics:
    def __init__(self):
        # detection
        self.detection_requests = Counter(
            'yolo_detection_requests_total',
            'Total number of detection requests'
        )
        self.detection_errors = Counter(
            'yolo_detection_errors_total',
            'Total number of detection errors'
        )
        # Visualize the inference time of detection
        self.inference_time = Histogram(
            'yolo_inference_time_seconds',
            'Time spent on inference',
            buckets=[.05, .1, .2, .5, 1, 2, 5]
        )

        # Metrics for treaining / re-treaining
        self.training_tasks = Counter(
            'yolo_training_tasks_total',
            'Number of training tasks',
            ['status']
        )
        self.active_trainings = Gauge(
            'yolo_active_trainings',
            'Number of currently running training tasks'
        )
        self.training_duration = Histogram(
            'yolo_training_duration_hours',
            'Training duration in hours',
            buckets=[1, 2, 4, 8, 12, 24, 48]
        )

metrics = MLMetrics()