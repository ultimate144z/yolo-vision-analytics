"""
Analytics Engine
Processes detection data and generates statistics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import config.config as config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.detection.yolo_detector import Detection

logger = get_logger(__name__)


class DetectionAnalytics:
    """
    Analytics engine for processing detection data
    Generates statistics, trends, and insights
    """
    
    def __init__(self):
        """Initialize analytics engine"""
        self.detections_history = []
        self.frame_detections = defaultdict(list)
        self.class_timeseries = defaultdict(list)
        self.total_detections = 0
        
        logger.info("Analytics engine initialized")
    
    def add_detections(self, detections: List['Detection'], frame_number: int):
        """
        Add detections from a frame
        
        Args:
            detections: List of Detection objects
            frame_number: Frame number
        """
        self.detections_history.extend(detections)
        self.frame_detections[frame_number] = detections
        self.total_detections += len(detections)
        
        # Update time series
        for det in detections:
            self.class_timeseries[det.class_name].append({
                'frame': frame_number,
                'timestamp': det.timestamp,
                'confidence': det.confidence
            })
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get count of detections by class
        
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = Counter(det.class_name for det in self.detections_history)
        return dict(class_counts)
    
    def get_confidence_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get confidence statistics by class
        
        Returns:
            Dictionary with mean, min, max confidence per class
        """
        stats = {}
        
        for class_name in set(det.class_name for det in self.detections_history):
            confidences = [
                det.confidence for det in self.detections_history 
                if det.class_name == class_name
            ]
            
            if confidences:
                stats[class_name] = {
                    'mean': np.mean(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'std': np.std(confidences)
                }
        
        return stats
    
    def get_detection_frequency(self, window_size: int = 30) -> Dict[str, List[int]]:
        """
        Calculate detection frequency over time windows
        
        Args:
            window_size: Number of frames per window
        
        Returns:
            Dictionary with detection counts per window for each class
        """
        frequency = defaultdict(list)
        
        if not self.detections_history:
            return dict(frequency)
        
        # Get frame range
        max_frame = max(self.frame_detections.keys())
        
        # Calculate frequency per window
        for start_frame in range(0, max_frame + 1, window_size):
            end_frame = start_frame + window_size
            
            window_detections = [
                det for frame in range(start_frame, end_frame)
                for det in self.frame_detections.get(frame, [])
            ]
            
            class_counts = Counter(det.class_name for det in window_detections)
            
            for class_name in config.TRACKED_CLASSES:
                frequency[class_name].append(class_counts.get(class_name, 0))
        
        return dict(frequency)
    
    def get_peak_activity_periods(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """
        Identify frames with highest detection activity
        
        Args:
            top_n: Number of peak periods to return
        
        Returns:
            List of (frame_number, detection_count) tuples
        """
        frame_counts = [
            (frame, len(dets)) 
            for frame, dets in self.frame_detections.items()
        ]
        
        # Sort by detection count
        frame_counts.sort(key=lambda x: x[1], reverse=True)
        
        return frame_counts[:top_n]
    
    def get_tracking_statistics(self) -> Dict[str, any]:
        """
        Get statistics about tracked objects
        
        Returns:
            Dictionary with tracking metrics
        """
        tracked_detections = [
            det for det in self.detections_history 
            if det.track_id is not None
        ]
        
        if not tracked_detections:
            return {
                'total_tracked_objects': 0,
                'unique_track_ids': 0,
                'average_track_length': 0
            }
        
        # Count unique track IDs
        unique_ids = set(det.track_id for det in tracked_detections)
        
        # Calculate average track length
        track_lengths = defaultdict(int)
        for det in tracked_detections:
            track_lengths[det.track_id] += 1
        
        avg_length = np.mean(list(track_lengths.values())) if track_lengths else 0
        
        return {
            'total_tracked_objects': len(tracked_detections),
            'unique_track_ids': len(unique_ids),
            'average_track_length': avg_length,
            'track_lengths': dict(track_lengths)
        }
    
    def get_temporal_patterns(self) -> Dict[str, any]:
        """
        Analyze temporal patterns in detections
        
        Returns:
            Dictionary with temporal analysis
        """
        if not self.detections_history:
            return {}
        
        # Group by timestamp intervals (e.g., every second)
        time_buckets = defaultdict(list)
        
        for det in self.detections_history:
            if det.timestamp is not None:
                # Round to nearest second
                bucket = int(det.timestamp)
                time_buckets[bucket].append(det)
        
        # Calculate statistics
        detections_per_second = [len(dets) for dets in time_buckets.values()]
        
        return {
            'total_seconds': len(time_buckets),
            'avg_detections_per_second': np.mean(detections_per_second) if detections_per_second else 0,
            'max_detections_per_second': max(detections_per_second) if detections_per_second else 0,
            'min_detections_per_second': min(detections_per_second) if detections_per_second else 0
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert detections to pandas DataFrame
        
        Returns:
            DataFrame with detection data
        """
        data = []
        
        for det in self.detections_history:
            data.append({
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox_x1': det.bbox[0],
                'bbox_y1': det.bbox[1],
                'bbox_x2': det.bbox[2],
                'bbox_y2': det.bbox[3],
                'track_id': det.track_id,
                'timestamp': det.timestamp,
                'frame_number': det.frame_number
            })
        
        return pd.DataFrame(data)
    
    def get_spatial_distribution(self, frame_width: int, frame_height: int,
                                 grid_size: int = 10) -> np.ndarray:
        """
        Calculate spatial distribution of detections
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            grid_size: Number of grid cells (grid_size x grid_size)
        
        Returns:
            2D array with detection counts per grid cell
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        cell_width = frame_width / grid_size
        cell_height = frame_height / grid_size
        
        for det in self.detections_history:
            # Get center of bounding box
            center_x = (det.bbox[0] + det.bbox[2]) / 2
            center_y = (det.bbox[1] + det.bbox[3]) / 2
            
            # Map to grid cell
            grid_x = int(center_x / cell_width)
            grid_y = int(center_y / cell_height)
            
            # Ensure within bounds
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)
            
            grid[grid_y, grid_x] += 1
        
        return grid
    
    def get_summary_report(self) -> Dict[str, any]:
        """
        Generate comprehensive summary report
        
        Returns:
            Dictionary with all analytics
        """
        return {
            'total_detections': self.total_detections,
            'total_frames': len(self.frame_detections),
            'class_distribution': self.get_class_distribution(),
            'confidence_statistics': self.get_confidence_statistics(),
            'tracking_statistics': self.get_tracking_statistics(),
            'temporal_patterns': self.get_temporal_patterns(),
            'peak_activity': self.get_peak_activity_periods(top_n=5)
        }
    
    def export_timeseries_data(self) -> Dict[str, pd.DataFrame]:
        """
        Export time series data for each class
        
        Returns:
            Dictionary mapping class names to DataFrames
        """
        timeseries_dfs = {}
        
        for class_name, data in self.class_timeseries.items():
            if data:
                df = pd.DataFrame(data)
                timeseries_dfs[class_name] = df
        
        return timeseries_dfs
    
    def reset(self):
        """Reset all analytics data"""
        self.detections_history.clear()
        self.frame_detections.clear()
        self.class_timeseries.clear()
        self.total_detections = 0
        logger.info("Analytics data reset")


class PerformanceMetrics:
    """
    Track performance metrics for the detection system
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.frame_times = []
        self.inference_times = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop timing"""
        self.end_time = datetime.now()
    
    def add_frame_time(self, time_ms: float):
        """Add frame processing time"""
        self.frame_times.append(time_ms)
    
    def add_inference_time(self, time_ms: float):
        """Add inference time"""
        self.inference_times.append(time_ms)
    
    def get_fps(self) -> float:
        """Calculate average FPS"""
        if not self.frame_times:
            return 0.0
        
        avg_time_s = np.mean(self.frame_times) / 1000.0
        return 1.0 / avg_time_s if avg_time_s > 0 else 0.0
    
    def get_total_time(self) -> float:
        """
        Get total processing time in seconds
        
        Returns:
            Total time in seconds, or 0 if not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_metrics(self) -> Dict[str, any]:
        """Get all performance metrics"""
        total_time = None
        if self.start_time and self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            'total_time_seconds': total_time,
            'total_frames': len(self.frame_times),
            'average_fps': self.get_fps(),
            'avg_frame_time_ms': np.mean(self.frame_times) if self.frame_times else 0,
            'avg_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0,
            'min_frame_time_ms': np.min(self.frame_times) if self.frame_times else 0,
            'max_frame_time_ms': np.max(self.frame_times) if self.frame_times else 0
        }
