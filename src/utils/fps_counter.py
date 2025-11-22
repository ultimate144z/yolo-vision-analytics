"""
FPS Counter for Performance Monitoring
Tracks frames per second and processing statistics
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    Real-time FPS counter with moving average
    Measures actual processing throughput (processed frames per second)
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.total_processing_time = 0.0
    
    def update(self) -> float:
        """
        Update FPS counter with new frame
        Measures instantaneous FPS based on time since last processed frame
        
        Returns:
            Current FPS (moving average)
        """
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self.frame_count += 1
        self.total_processing_time += frame_time
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """
        Get current FPS (instantaneous, based on recent frames)
        
        Returns:
            Frames per second (moving average)
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0
    
    def get_average_fps(self) -> float:
        """
        Get average FPS since start (total processed frames / wall clock time)
        This is the true throughput metric
        
        Returns:
            Average frames per second
        """
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def get_eta(self, total_frames: int, current_frame: int = None) -> Optional[str]:
        """
        Get estimated time remaining
        
        Args:
            total_frames: Total number of frames to process
            current_frame: Current frame number (if None, uses self.frame_count)
            
        Returns:
            Formatted ETA string or None
        """
        frames_processed = current_frame if current_frame is not None else self.frame_count
        
        if frames_processed == 0:
            return None
        
        avg_fps = self.get_average_fps()
        if avg_fps == 0:
            return None
        
        remaining_frames = total_frames - frames_processed
        if remaining_frames <= 0:
            return "00:00"
        
        eta_seconds = remaining_frames / avg_fps
        minutes = int(eta_seconds // 60)
        seconds = int(eta_seconds % 60)
        
        return f"{minutes:02d}:{seconds:02d}"
    
    def reset(self):
        """Reset counter"""
        self.frame_times.clear()
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.total_processing_time = 0.0
