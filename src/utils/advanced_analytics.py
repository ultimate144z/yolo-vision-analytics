"""
Advanced Analytics Module
Extended statistics and zone-based analysis
"""

import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict
import pandas as pd

if TYPE_CHECKING:
    from src.detection.yolo_detector import Detection


class ZoneAnalyzer:
    """
    Analyzes detections by spatial zones
    Divides frame into grid and tracks activity per zone
    """
    
    def __init__(self, frame_width: int, frame_height: int, grid_size: Tuple[int, int] = (3, 3)):
        """
        Initialize zone analyzer
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            grid_size: Grid dimensions (rows, cols)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows, self.grid_cols = grid_size
        
        # Calculate zone dimensions
        self.zone_width = frame_width / self.grid_cols
        self.zone_height = frame_height / self.grid_rows
        
        # Track detections per zone
        self.zone_counts = defaultdict(int)
        self.zone_classes = defaultdict(lambda: defaultdict(int))
    
    def get_zone(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get zone index for bounding box center
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Zone coordinates (row, col)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        row = min(int(center_y / self.zone_height), self.grid_rows - 1)
        col = min(int(center_x / self.zone_width), self.grid_cols - 1)
        
        return (row, col)
    
    def add_detection(self, detection: 'Detection'):
        """
        Add detection to zone analysis
        
        Args:
            detection: Detection object
        """
        zone = self.get_zone(detection.bbox)
        self.zone_counts[zone] += 1
        self.zone_classes[zone][detection.class_name] += 1
    
    def get_zone_heatmap(self) -> np.ndarray:
        """
        Get zone activity as 2D heatmap array
        
        Returns:
            Array of shape (grid_rows, grid_cols) with counts
        """
        heatmap = np.zeros((self.grid_rows, self.grid_cols))
        
        for (row, col), count in self.zone_counts.items():
            heatmap[row, col] = count
        
        return heatmap
    
    def get_hotspot_zones(self, top_n: int = 3) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get zones with highest activity
        
        Args:
            top_n: Number of top zones to return
            
        Returns:
            List of ((row, col), count) tuples
        """
        sorted_zones = sorted(
            self.zone_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_zones[:top_n]
    
    def get_zone_summary(self) -> Dict:
        """
        Get comprehensive zone analysis summary
        
        Returns:
            Dictionary with zone statistics
        """
        heatmap = self.get_zone_heatmap()
        hotspots = self.get_hotspot_zones()
        
        return {
            'grid_size': (self.grid_rows, self.grid_cols),
            'zone_heatmap': heatmap.tolist(),
            'total_zones_with_activity': len(self.zone_counts),
            'hotspot_zones': [
                {
                    'zone': f"({row},{col})",
                    'count': count,
                    'classes': dict(self.zone_classes[(row, col)])
                }
                for (row, col), count in hotspots
            ],
            'average_detections_per_zone': np.mean(list(self.zone_counts.values())) if self.zone_counts else 0
        }


class ObjectSizeAnalyzer:
    """
    Analyzes distribution of object sizes
    Useful for understanding object distances and types
    """
    
    def __init__(self):
        """Initialize size analyzer"""
        self.sizes = []
        self.class_sizes = defaultdict(list)
    
    def add_detection(self, detection: 'Detection'):
        """
        Add detection for size analysis
        
        Args:
            detection: Detection object
        """
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        self.sizes.append({
            'width': width,
            'height': height,
            'area': area,
            'aspect_ratio': width / height if height > 0 else 0
        })
        
        self.class_sizes[detection.class_name].append(area)
    
    def get_size_statistics(self) -> Dict:
        """
        Get size distribution statistics
        
        Returns:
            Dictionary with size stats
        """
        if not self.sizes:
            return {}
        
        areas = [s['area'] for s in self.sizes]
        widths = [s['width'] for s in self.sizes]
        heights = [s['height'] for s in self.sizes]
        
        return {
            'area': {
                'mean': np.mean(areas),
                'median': np.median(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'width': {
                'mean': np.mean(widths),
                'median': np.median(widths),
                'std': np.std(widths)
            },
            'height': {
                'mean': np.mean(heights),
                'median': np.median(heights),
                'std': np.std(heights)
            },
            'size_distribution': {
                'small': sum(1 for a in areas if a < 5000),
                'medium': sum(1 for a in areas if 5000 <= a < 20000),
                'large': sum(1 for a in areas if a >= 20000)
            }
        }
    
    def get_class_size_comparison(self) -> Dict[str, Dict]:
        """
        Get average sizes by class
        
        Returns:
            Dictionary mapping classes to average sizes
        """
        comparison = {}
        
        for class_name, areas in self.class_sizes.items():
            if areas:
                comparison[class_name] = {
                    'mean_area': np.mean(areas),
                    'median_area': np.median(areas),
                    'count': len(areas)
                }
        
        return comparison


class ConfidenceTemporalAnalyzer:
    """
    Analyzes confidence trends over time
    Helps identify model performance patterns
    """
    
    def __init__(self):
        """Initialize confidence analyzer"""
        self.frame_confidences = defaultdict(list)
        self.class_confidences_over_time = defaultdict(lambda: defaultdict(list))
    
    def add_detection(self, detection: 'Detection', frame_number: int):
        """
        Add detection for confidence analysis
        
        Args:
            detection: Detection object
            frame_number: Frame number
        """
        self.frame_confidences[frame_number].append(detection.confidence)
        self.class_confidences_over_time[detection.class_name][frame_number].append(
            detection.confidence
        )
    
    def get_confidence_trend(self) -> Dict:
        """
        Get confidence trend over frames
        
        Returns:
            Dictionary with trend data
        """
        if not self.frame_confidences:
            return {}
        
        frames = sorted(self.frame_confidences.keys())
        avg_confidences = [
            np.mean(self.frame_confidences[f]) for f in frames
        ]
        
        return {
            'frames': frames,
            'average_confidences': avg_confidences,
            'overall_mean': np.mean(avg_confidences),
            'overall_std': np.std(avg_confidences),
            'trend': 'increasing' if avg_confidences[-1] > avg_confidences[0] else 'decreasing'
        }
    
    def get_low_confidence_frames(self, threshold: float = 0.5) -> List[int]:
        """
        Identify frames with low average confidence
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of frame numbers
        """
        low_conf_frames = []
        
        for frame, confidences in self.frame_confidences.items():
            avg_conf = np.mean(confidences)
            if avg_conf < threshold:
                low_conf_frames.append(frame)
        
        return sorted(low_conf_frames)
