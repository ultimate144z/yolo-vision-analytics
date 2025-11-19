"""Detection modules for YOLOv8 object detection"""

from .yolo_detector import YOLODetector, Detection
from .video_processor import VideoProcessor, BatchVideoProcessor

__all__ = ['YOLODetector', 'Detection', 'VideoProcessor', 'BatchVideoProcessor']
