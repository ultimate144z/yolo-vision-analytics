"""
Heatmap Generation Module
Creates activity heatmaps from detection data
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config.config as config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.detection.yolo_detector import Detection

logger = get_logger(__name__)


class HeatmapGenerator:
    """
    Generate activity heatmaps from object detections
    Uses Gaussian kernel density estimation
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize heatmap generator
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.detection_points = []
        
        logger.info(f"Heatmap generator initialized: {frame_width}x{frame_height}")
    
    def add_detection(self, detection: 'Detection'):
        """
        Add detection to heatmap
        
        Args:
            detection: Detection object
        """
        # Calculate center point of bounding box
        x1, y1, x2, y2 = detection.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Ensure within bounds
        center_x = max(0, min(center_x, self.frame_width - 1))
        center_y = max(0, min(center_y, self.frame_height - 1))
        
        self.detection_points.append((center_x, center_y))
    
    def add_detections(self, detections: List['Detection']):
        """
        Add multiple detections to heatmap
        
        Args:
            detections: List of Detection objects
        """
        for det in detections:
            self.add_detection(det)
    
    def generate(self, kernel_size: int = None, sigma: float = None) -> np.ndarray:
        """
        Generate heatmap using Gaussian blur
        
        Args:
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation for Gaussian kernel
        
        Returns:
            Heatmap as 2D array
        """
        if kernel_size is None:
            kernel_size = config.HEATMAP_KERNEL_SIZE
        if sigma is None:
            sigma = config.HEATMAP_SIGMA
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        logger.info(f"Generating heatmap with {len(self.detection_points)} points")
        
        # Create accumulation map
        accumulation = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Add each detection point
        for x, y in self.detection_points:
            accumulation[y, x] += 1
        
        # Apply Gaussian blur for smooth heatmap
        if len(self.detection_points) > 0:
            self.heatmap = cv2.GaussianBlur(
                accumulation,
                (kernel_size, kernel_size),
                sigma
            )
        
        return self.heatmap
    
    def apply_colormap(self, heatmap: np.ndarray = None, 
                      colormap: str = None) -> np.ndarray:
        """
        Apply colormap to heatmap
        
        Args:
            heatmap: Heatmap array (uses self.heatmap if None)
            colormap: OpenCV colormap name
        
        Returns:
            Colored heatmap (BGR format)
        """
        if heatmap is None:
            heatmap = self.heatmap
        
        if colormap is None:
            colormap = config.HEATMAP_COLORMAP
        
        # Normalize to 0-255
        if heatmap.max() > 0:
            normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(heatmap, dtype=np.uint8)
        
        # Apply colormap
        colormap_code = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(normalized, colormap_code)
        
        return colored
    
    def overlay_on_frame(self, frame: np.ndarray, 
                        alpha: float = 0.5,
                        colormap: str = None) -> np.ndarray:
        """
        Overlay heatmap on original frame
        
        Args:
            frame: Original frame
            alpha: Transparency (0=transparent, 1=opaque)
            colormap: Colormap to use
        
        Returns:
            Frame with heatmap overlay
        """
        # Generate colored heatmap
        colored_heatmap = self.apply_colormap(colormap=colormap)
        
        # Resize heatmap if needed
        if colored_heatmap.shape[:2] != frame.shape[:2]:
            colored_heatmap = cv2.resize(colored_heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        return overlay
    
    def get_high_activity_zones(self, threshold_percentile: float = 75) -> List[Tuple[int, int]]:
        """
        Identify high activity zones in heatmap
        
        Args:
            threshold_percentile: Percentile threshold for high activity
        
        Returns:
            List of (x, y) coordinates of high activity zones
        """
        if self.heatmap.max() == 0:
            return []
        
        # Calculate threshold
        threshold = np.percentile(self.heatmap, threshold_percentile)
        
        # Find zones above threshold
        high_activity = np.where(self.heatmap > threshold)
        
        # Convert to list of coordinates
        zones = list(zip(high_activity[1], high_activity[0]))  # x, y
        
        logger.info(f"Found {len(zones)} high activity points (>{threshold_percentile}th percentile)")
        
        return zones
    
    def save(self, output_path: str, include_colorbar: bool = True):
        """
        Save heatmap as image file (on G: drive)
        
        Args:
            output_path: Output file path
            include_colorbar: Include colorbar in saved image
        """
        output_path = Path(output_path)
        
        # Ensure on G: drive
        if not str(output_path).startswith('G:') and not str(output_path).startswith('g:'):
            output_path = config.OUTPUT_DIR / output_path.name
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_colorbar:
            # Use matplotlib for better visualization with colorbar
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(self.heatmap, cmap=config.HEATMAP_COLORMAP, 
                          interpolation='bilinear')
            ax.set_title('Activity Heatmap')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Detection Density')
            
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Save colored heatmap directly
            colored = self.apply_colormap()
            cv2.imwrite(str(output_path), colored)
        
        logger.info(f"Heatmap saved to: {output_path}")
    
    def get_statistics(self) -> dict:
        """
        Get heatmap statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.heatmap.max() == 0:
            return {
                'total_detections': len(self.detection_points),
                'max_density': 0,
                'mean_density': 0,
                'high_activity_zones': 0
            }
        
        return {
            'total_detections': len(self.detection_points),
            'max_density': float(self.heatmap.max()),
            'mean_density': float(self.heatmap.mean()),
            'std_density': float(self.heatmap.std()),
            'high_activity_zones': len(self.get_high_activity_zones())
        }
    
    def reset(self):
        """Reset heatmap data"""
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.detection_points.clear()
        logger.info("Heatmap reset")


class MultiClassHeatmapGenerator:
    """
    Generate separate heatmaps for different object classes
    """
    
    def __init__(self, frame_width: int, frame_height: int, 
                 classes: List[str] = None):
        """
        Initialize multi-class heatmap generator
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            classes: List of classes to track
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.classes = classes or config.TRACKED_CLASSES
        
        # Create separate heatmap for each class
        self.heatmaps = {
            cls: HeatmapGenerator(frame_width, frame_height)
            for cls in self.classes
        }
        
        logger.info(f"Multi-class heatmap generator initialized for: {self.classes}")
    
    def add_detections(self, detections: List['Detection']):
        """
        Add detections to appropriate class heatmaps
        
        Args:
            detections: List of Detection objects
        """
        for det in detections:
            if det.class_name in self.heatmaps:
                self.heatmaps[det.class_name].add_detection(det)
    
    def generate_all(self) -> dict:
        """
        Generate all class heatmaps
        
        Returns:
            Dictionary mapping class names to heatmaps
        """
        heatmaps = {}
        for class_name, generator in self.heatmaps.items():
            heatmaps[class_name] = generator.generate()
        
        logger.info(f"Generated {len(heatmaps)} class heatmaps")
        return heatmaps
    
    def save_all(self, output_dir: Path):
        """
        Save all class heatmaps
        
        Args:
            output_dir: Output directory (on G: drive)
        """
        output_dir = Path(output_dir)
        
        # Ensure on G: drive
        if not str(output_dir).startswith('G:') and not str(output_dir).startswith('g:'):
            output_dir = config.OUTPUT_DIR / output_dir.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_name, generator in self.heatmaps.items():
            output_path = output_dir / f"heatmap_{class_name}.png"
            generator.generate()
            generator.save(str(output_path))
        
        logger.info(f"All heatmaps saved to: {output_dir}")
    
    def get_combined_heatmap(self) -> np.ndarray:
        """
        Combine all class heatmaps into single heatmap
        
        Returns:
            Combined heatmap
        """
        combined = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        for generator in self.heatmaps.values():
            combined += generator.generate()
        
        return combined
    
    def reset_all(self):
        """Reset all class heatmaps"""
        for generator in self.heatmaps.values():
            generator.reset()
        logger.info("All heatmaps reset")


def create_comparison_heatmap(heatmap1: np.ndarray, 
                              heatmap2: np.ndarray,
                              labels: Tuple[str, str] = ('Period 1', 'Period 2')) -> np.ndarray:
    """
    Create side-by-side comparison of two heatmaps
    
    Args:
        heatmap1: First heatmap
        heatmap2: Second heatmap
        labels: Labels for each heatmap
    
    Returns:
        Combined comparison image
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot first heatmap
    im1 = axes[0].imshow(heatmap1, cmap=config.HEATMAP_COLORMAP)
    axes[0].set_title(labels[0])
    plt.colorbar(im1, ax=axes[0])
    
    # Plot second heatmap
    im2 = axes[1].imshow(heatmap2, cmap=config.HEATMAP_COLORMAP)
    axes[1].set_title(labels[1])
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return image
