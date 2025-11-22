"""
Video Processing Module
Handles video file reading, webcam capture, and frame extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
from datetime import datetime
import time

import config.config as config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_available_cameras(max_cameras: int = 5) -> list:
    """
    Detect available camera indices
    
    Args:
        max_cameras: Maximum number of camera indices to check
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


class VideoProcessor:
    """
    Video processing class for handling various video sources
    Supports video files and webcam input
    """
    
    def __init__(self, source: Union[str, int, Path], output_path: str = None):
        """
        Initialize video processor
        
        Args:
            source: Video file path, webcam index, or 'webcam'
            output_path: Optional path to save processed video (on G: drive)
        """
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.is_webcam = False
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.current_frame = 0
        
        self._initialize_capture()
        self._initialize_writer()
    
    def _initialize_capture(self):
        """Initialize video capture from file or webcam"""
        try:
            # Handle different source types
            if isinstance(self.source, str) and self.source.lower() == 'webcam':
                self.source = config.WEBCAM_INDEX
                self.is_webcam = True
            elif isinstance(self.source, int):
                self.is_webcam = True
            else:
                self.source = Path(self.source)
                if not self.source.exists():
                    raise FileNotFoundError(f"Video file not found: {self.source}")
                
                # Validate file format
                if self.source.suffix.lower() not in config.SUPPORTED_FORMATS:
                    raise ValueError(
                        f"Unsupported format: {self.source.suffix}. "
                        f"Supported: {config.SUPPORTED_FORMATS}"
                    )
                
                # Check file size
                file_size_mb = self.source.stat().st_size / (1024 * 1024)
                if file_size_mb > config.MAX_VIDEO_SIZE_MB:
                    logger.warning(
                        f"Video size ({file_size_mb:.1f} MB) exceeds recommended "
                        f"limit ({config.MAX_VIDEO_SIZE_MB} MB)"
                    )
            
            # Open capture
            if self.is_webcam:
                logger.info(f"Opening webcam (index: {self.source})")
                self.cap = cv2.VideoCapture(self.source)
                
                # If failed, try to find available cameras
                if not self.cap.isOpened():
                    logger.warning(f"Camera index {self.source} not available, detecting cameras...")
                    available_cameras = get_available_cameras()
                    
                    if not available_cameras:
                        raise RuntimeError(
                            "No webcams detected. Please check:\n"
                            "1. Webcam is connected\n"
                            "2. Camera permissions are granted\n"
                            "3. Camera is not being used by another application"
                        )
                    
                    logger.info(f"Available cameras: {available_cameras}")
                    self.source = available_cameras[0]
                    logger.info(f"Using camera index: {self.source}")
                    self.cap = cv2.VideoCapture(self.source)
                
                if config.WEBCAM_RESOLUTION and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_RESOLUTION[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_RESOLUTION[1])
            else:
                logger.info(f"Opening video file: {self.source}")
                self.cap = cv2.VideoCapture(str(self.source))
            
            if not self.cap.isOpened():
                raise RuntimeError(
                    "Failed to open video source. "
                    "If using webcam, try a different camera index (0, 1, 2, etc.)"
                )
            
            # Get video properties
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.fps == 0:
                self.fps = config.WEBCAM_FPS if self.is_webcam else 30
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.is_webcam:
                logger.info(f"Webcam initialized: {self.width}x{self.height} @ {self.fps} FPS")
            else:
                logger.info(
                    f"Video loaded: {self.width}x{self.height} @ {self.fps} FPS, "
                    f"{self.total_frames} frames"
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize capture: {e}")
            raise
    
    def _initialize_writer(self):
        """Initialize video writer for output (on G: drive)"""
        if self.output_path is None:
            return
        
        try:
            # Ensure output path is on G: drive
            output_path = Path(self.output_path)
            if not str(output_path).startswith('G:') and not str(output_path).startswith('g:'):
                output_path = config.OUTPUT_DIR / output_path.name
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                raise RuntimeError("Failed to initialize video writer")
            
            logger.info(f"Output video will be saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            self.writer = None
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video source
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame += 1
        
        return ret, frame
    
    def frames(self, frame_skip: int = 1) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator that yields frames with metadata
        
        Args:
            frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
                       For video files, this skips reading intermediate frames for performance.
                       For webcams, frames are still read but only yielded every Nth frame.
        
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        start_time = time.time()
        frames_yielded = 0
        
        if self.is_webcam:
            # For webcams, we must read all frames (can't skip)
            frames_read = 0
            while True:
                ret, frame = self.read_frame()
                
                if not ret:
                    break
                
                frames_read += 1
                
                # Only yield every Nth frame
                if (frames_read % frame_skip) == 0:
                    timestamp = time.time() - start_time
                    yield frame, self.current_frame, timestamp
                    frames_yielded += 1
            
            logger.info(f"Webcam: Read {frames_read} frames, yielded {frames_yielded}")
        else:
            # For video files, we can skip reading frames for better performance
            frame_position = 0
            
            while frame_position < self.total_frames:
                # Set position to next frame we want to process
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                self.current_frame = frame_position + 1
                timestamp = time.time() - start_time
                yield frame, self.current_frame, timestamp
                frames_yielded += 1
                
                # Skip to next frame position
                frame_position += frame_skip
            
            logger.info(f"Video file: Yielded {frames_yielded} frames (skipped {self.total_frames - frames_yielded} frames)")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write frame to output video
        
        Args:
            frame: Frame to write
        """
        if self.writer is not None:
            self.writer.write(frame)
    
    def get_progress(self) -> float:
        """
        Get processing progress percentage
        
        Returns:
            Progress percentage (0-100)
        """
        if self.is_webcam or self.total_frames == 0:
            return 0.0
        
        return (self.current_frame / self.total_frames) * 100
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by number (not available for webcam)
        
        Args:
            frame_number: Frame number to retrieve
        
        Returns:
            Frame or None if not available
        """
        if self.is_webcam:
            logger.warning("Random frame access not available for webcam")
            return None
        
        if self.cap is None:
            return None
        
        # Save current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Seek to desired frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frame if ret else None
    
    def resize_frame(self, frame: np.ndarray, 
                    target_width: int = None, 
                    target_height: int = None,
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize frame to target dimensions
        
        Args:
            frame: Input frame
            target_width: Target width (use config default if None)
            target_height: Target height (use config default if None)
            maintain_aspect: Maintain aspect ratio
        
        Returns:
            Resized frame
        """
        if target_width is None:
            target_width = config.FRAME_WIDTH
        if target_height is None:
            target_height = config.FRAME_HEIGHT
        
        if maintain_aspect:
            # Calculate aspect ratio
            h, w = frame.shape[:2]
            aspect = w / h
            
            if w > h:
                target_height = int(target_width / aspect)
            else:
                target_width = int(target_height * aspect)
        
        return cv2.resize(frame, (target_width, target_height))
    
    def get_video_info(self) -> dict:
        """
        Get video information
        
        Returns:
            Dictionary with video metadata
        """
        return {
            'source': str(self.source),
            'is_webcam': self.is_webcam,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'current_frame': self.current_frame,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
        }
    
    def release(self):
        """Release video capture and writer resources"""
        try:
            if self.cap is not None:
                self.cap.release()
                logger.info("Video capture released")
            
            if self.writer is not None:
                self.writer.release()
                logger.info(f"Video saved to: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error releasing resources: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


class BatchVideoProcessor:
    """
    Batch processor for multiple video files
    """
    
    def __init__(self, video_paths: list, output_dir: Path = None):
        """
        Initialize batch processor
        
        Args:
            video_paths: List of video file paths
            output_dir: Output directory for processed videos (on G: drive)
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.output_dir = output_dir or config.OUTPUT_DIR
        
        # Ensure output dir is on G: drive
        if not str(self.output_dir).startswith('G:') and not str(self.output_dir).startswith('g:'):
            self.output_dir = config.OUTPUT_DIR
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch processor initialized with {len(self.video_paths)} videos")
    
    def process_all(self, detector, callback=None):
        """
        Process all videos with given detector
        
        Args:
            detector: YOLODetector instance
            callback: Optional callback function(video_path, progress)
        """
        results = []
        
        for i, video_path in enumerate(self.video_paths):
            logger.info(f"Processing video {i+1}/{len(self.video_paths)}: {video_path.name}")
            
            output_path = self.output_dir / f"processed_{video_path.name}"
            
            try:
                with VideoProcessor(video_path, str(output_path)) as processor:
                    video_results = []
                    
                    for frame, frame_num, timestamp in processor.frames():
                        detections = detector.detect(frame, track=True, timestamp=timestamp)
                        annotated_frame = detector.draw_detections(frame, detections)
                        processor.write_frame(annotated_frame)
                        video_results.append(detections)
                        
                        if callback:
                            progress = processor.get_progress()
                            callback(video_path, progress)
                    
                    results.append({
                        'video_path': str(video_path),
                        'output_path': str(output_path),
                        'detections': video_results,
                        'summary': detector.get_detection_summary()
                    })
                
                # Reset detector counts for next video
                detector.reset_counts()
                
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {e}")
                results.append({
                    'video_path': str(video_path),
                    'error': str(e)
                })
        
        return results


def save_frame(frame: np.ndarray, output_path: Union[str, Path]):
    """
    Save single frame as image (on G: drive)
    
    Args:
        frame: Frame to save
        output_path: Output path for image
    """
    output_path = Path(output_path)
    
    # Ensure on G: drive
    if not str(output_path).startswith('G:') and not str(output_path).startswith('g:'):
        output_path = config.OUTPUT_DIR / output_path.name
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), frame)
    logger.info(f"Frame saved to: {output_path}")
