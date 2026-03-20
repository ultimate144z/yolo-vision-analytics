"""
YOLOv8 Object Detection Module
Handles model loading, inference, and object tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

import config.config as config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Data class for single object detection"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    frame_number: Optional[int] = None


class YOLODetector:
    """
    YOLOv8 object detector with tracking capabilities
    Handles model loading, inference, and result processing
    """
    
    def __init__(self, 
                 model_name: str = None,
                 confidence_threshold: float = None,
                 device: str = None,
                 tracked_classes: List[str] = None):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model name (default from config)
            confidence_threshold: Minimum confidence for detections
            device: 'cuda' or 'cpu' (auto-detect if None)
            tracked_classes: List of class names to track
        """
        self.model_name = model_name or config.YOLO_MODEL
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.device = device or self._detect_device()
        self.tracked_classes = tracked_classes or config.TRACKED_CLASSES
        
        logger.info(f"Initializing YOLOv8 detector with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Load model (returns tuple of model and format)
        self.model, self.model_format = self._load_model()
        
        # Tracking data
        self.track_history = defaultdict(list)
        self.detection_count = defaultdict(int)
        self.frame_count = 0
        
        logger.info("YOLOv8 detector initialized successfully")
    
    def _detect_device(self) -> str:
        """Auto-detect available device (CUDA or CPU)"""
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected and will be used")
            return "cuda"
        else:
            logger.warning("No GPU detected, using CPU (slower)")
            return "cpu"
    
    def _load_model(self) -> Tuple[YOLO, str]:
        """
        Load YOLO model from disk or download if not exists
        Automatically exports to TensorRT (GPU) or ONNX (CPU) for faster inference
        
        Returns:
            Tuple of (model, format) where format is 'pt', 'onnx', or 'engine'
        """
        model_path = config.MODEL_DIR / self.model_name
        
        try:
            # Define optimized model paths
            trt_path = model_path.with_suffix('.engine')
            onnx_path = model_path.with_suffix('.onnx')
            loaded_format = 'pt'  # Track which format we loaded
            
            if model_path.exists():
                # GPU: Prefer TensorRT engine for maximum speed
                if self.device == 'cuda' and trt_path.exists():
                    logger.info(f"Loading optimized TensorRT model: {trt_path}")
                    model = YOLO(str(trt_path), task='detect')
                    loaded_format = 'engine'
                # CPU: Prefer ONNX for faster inference
                elif self.device == 'cpu' and onnx_path.exists():
                    logger.info(f"Loading optimized ONNX model: {onnx_path}")
                    model = YOLO(str(onnx_path), task='detect')
                    loaded_format = 'onnx'
                else:
                    logger.info(f"Loading model from: {model_path}")
                    model = YOLO(str(model_path))
            else:
                logger.info(f"Model not found, downloading {self.model_name}...")
                # Download model - it will be cached in current directory first
                model = YOLO(self.model_name)
                
                # Copy downloaded model to models directory
                import shutil
                downloaded_path = Path(self.model_name)
                if downloaded_path.exists():
                    logger.info(f"Copying model to: {model_path}")
                    shutil.copy(str(downloaded_path), str(model_path))
                    logger.info(f"Model saved to {model_path}")
            
            # GPU Optimization: Export to TensorRT for CUDA users (only if using .pt)
            if loaded_format == 'pt' and self.device == 'cuda' and not trt_path.exists():
                logger.info("CUDA detected. Exporting to TensorRT for maximum GPU performance...")
                try:
                    model.export(format='engine')
                    if trt_path.exists():
                        logger.info("TensorRT export successful. Reloading optimized engine...")
                        model = YOLO(str(trt_path), task='detect')
                        loaded_format = 'engine'
                except Exception as e:
                    logger.warning(f"TensorRT export failed (continuing with PyTorch): {e}")
            # CPU Optimization: ONNX export DISABLED - causes compatibility issues with tracking
            # If you want ONNX, manually export and ensure your use case doesn't need tracking
            # elif loaded_format == 'pt' and self.device == 'cpu' and not onnx_path.exists():
            #     logger.info("CPU mode detected. Exporting to ONNX for faster CPU inference...")
            #     try:
            #         model.export(format='onnx', simplify=True)
            #         if onnx_path.exists():
            #             logger.info("ONNX export successful. Reloading optimized model...")
            #             model = YOLO(str(onnx_path), task='detect')
            #             loaded_format = 'onnx'
            #     except Exception as e:
            #         logger.warning(f"ONNX export failed (continuing with PyTorch): {e}")

            # Move model to device ONLY for PyTorch models (not ONNX/TensorRT)
            if loaded_format == 'pt':
                model.to(self.device)
                
            logger.info(f"Model loaded successfully (format: {loaded_format})")
            return model, loaded_format
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, 
               frame: Union[np.ndarray, List[np.ndarray]], 
               track: bool = True,
               timestamp: Union[float, List[float]] = None) -> Union[List[Detection], List[List[Detection]]]:
        """
        Perform object detection on a single frame or batch of frames
        
        Args:
            frame: Input frame (BGR) or list of frames
            track: Enable object tracking
            timestamp: Optional timestamp(s)
        
        Returns:
            List of Detection objects (if single frame) OR List of lists of Detection objects (if batch)
        """
        
        try:
            is_batch = isinstance(frame, list)
            
            if is_batch:
                self.frame_count += len(frame)
            else:
                self.frame_count += 1
            
            # ONNX/TensorRT models don't support tracking, use regular inference
            use_track = track and self.model_format == 'pt'
            
            # Run inference
            if use_track:
                # YOLOv8 tracking (PyTorch only)
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=config.IOU_THRESHOLD,
                    imgsz=config.INFERENCE_SIZE,
                    persist=True,
                    verbose=False
                )
            else:
                # Regular inference (works with all formats)
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=config.IOU_THRESHOLD,
                    imgsz=config.INFERENCE_SIZE,
                    verbose=False
                )
            
            # Process results
            all_detections = []
            
            # Handle single result vs list of results
            if not isinstance(results, list):
                results = [results]
                
            for i, result in enumerate(results):
                current_timestamp = timestamp[i] if is_batch and timestamp else timestamp
                # frame_number calc is approximate if not passed explicitly, but we rely on internal counting
                current_frame_num = self.frame_count - len(frame) + i + 1 if is_batch else self.frame_count
                
                detections = self._process_results(
                    result, 
                    timestamp=current_timestamp,
                    frame_number=current_frame_num
                )
                all_detections.append(detections)
            
            if is_batch:
                return all_detections
            else:
                return all_detections[0]
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return [] if not isinstance(frame, list) else [[] for _ in frame]
    
    def _process_results(self, 
                        result, 
                        timestamp: float = None,
                        frame_number: int = None) -> List[Detection]:
        """
        Process YOLO results into Detection objects
        
        Args:
            result: YOLO result object
            timestamp: Optional timestamp
            frame_number: Frame number
        
        Returns:
            List of Detection objects
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.cpu().numpy()
        
        for i, box in enumerate(boxes):
            # Get class name
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Filter by tracked classes
            if class_name not in self.tracked_classes:
                continue
            
            # Get bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            
            # Get confidence
            confidence = float(box.conf[0])
            
            # Get track ID if available
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
            
            # Create detection
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                track_id=track_id,
                timestamp=timestamp,
                frame_number=frame_number
            )
            
            detections.append(detection)
            
            # Update counts
            self.detection_count[class_name] += 1
        
        return detections
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: List[Detection],
                       show_confidence: bool = True,
                       show_track_id: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of Detection objects
            show_confidence: Show confidence scores
            show_track_id: Show tracking IDs
        
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color map for different classes
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'bicycle': (0, 255, 255),   # Yellow
            'motorcycle': (255, 0, 255), # Magenta
            'bus': (255, 128, 0),       # Orange
            'truck': (128, 0, 255),     # Purple
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [det.class_name]
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if show_track_id and det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def get_detection_summary(self) -> Dict[str, int]:
        """
        Get summary of all detections by class
        
        Returns:
            Dictionary with class names and counts
        """
        return dict(self.detection_count)
    
    def reset_counts(self):
        """Reset detection counters"""
        self.detection_count.clear()
        self.frame_count = 0
        self.track_history.clear()
        logger.info("Detection counts reset")
    
    def get_class_names(self) -> List[str]:
        """Get list of all available class names"""
        return list(self.model.names.values())
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold dynamically"""
        if 0.0 <= threshold <= 1.0:
            old_threshold = self.confidence_threshold
            self.confidence_threshold = threshold
            if old_threshold != threshold:
                logger.info(f"Confidence threshold updated: {old_threshold:.2f} -> {threshold:.2f}")
        else:
            logger.warning(f"Invalid threshold: {threshold}. Must be between 0 and 1")
    
    def export_model_info(self) -> Dict:
        """
        Export model information for reporting
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'tracked_classes': self.tracked_classes,
            'total_frames_processed': self.frame_count,
            'detection_summary': self.get_detection_summary()
        }
