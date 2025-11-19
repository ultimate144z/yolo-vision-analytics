"""Unit tests for YOLO detector module"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config.config as config
from src.detection.yolo_detector import YOLODetector, Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestYOLODetector:
    """Test cases for YOLODetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return YOLODetector(
            confidence_threshold=0.5,
            tracked_classes=['person', 'car']
        )
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert detector.confidence_threshold == 0.5
        assert 'person' in detector.tracked_classes
        assert 'car' in detector.tracked_classes
        logger.info("✓ Detector initialization test passed")
    
    def test_model_loading(self, detector):
        """Test model loads successfully"""
        assert detector.model is not None
        assert hasattr(detector.model, 'names')
        logger.info("✓ Model loading test passed")
    
    def test_detection_on_blank_frame(self, detector):
        """Test detection on blank frame"""
        # Create blank frame
        blank_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(blank_frame, track=False)
        
        # Should return empty list or minimal detections
        assert isinstance(detections, list)
        logger.info(f"✓ Blank frame test passed: {len(detections)} detections")
    
    def test_detection_summary(self, detector):
        """Test detection summary generation"""
        summary = detector.get_detection_summary()
        
        assert isinstance(summary, dict)
        logger.info("✓ Detection summary test passed")
    
    def test_confidence_threshold_update(self, detector):
        """Test updating confidence threshold"""
        new_threshold = 0.7
        detector.set_confidence_threshold(new_threshold)
        
        assert detector.confidence_threshold == new_threshold
        logger.info("✓ Confidence threshold update test passed")
    
    def test_class_names(self, detector):
        """Test getting class names"""
        class_names = detector.get_class_names()
        
        assert isinstance(class_names, list)
        assert len(class_names) > 0
        logger.info(f"✓ Class names test passed: {len(class_names)} classes available")
    
    def test_reset_counts(self, detector):
        """Test resetting detection counts"""
        detector.reset_counts()
        summary = detector.get_detection_summary()
        
        assert len(summary) == 0 or all(v == 0 for v in summary.values())
        logger.info("✓ Reset counts test passed")


class TestDetection:
    """Test cases for Detection dataclass"""
    
    def test_detection_creation(self):
        """Test creating Detection object"""
        detection = Detection(
            class_name='person',
            confidence=0.95,
            bbox=(100, 100, 200, 200),
            track_id=1,
            timestamp=1.5,
            frame_number=45
        )
        
        assert detection.class_name == 'person'
        assert detection.confidence == 0.95
        assert detection.bbox == (100, 100, 200, 200)
        assert detection.track_id == 1
        assert detection.timestamp == 1.5
        assert detection.frame_number == 45
        logger.info("✓ Detection creation test passed")


def test_config_paths():
    """Test that all config paths are on G: drive"""
    paths_to_check = [
        config.BASE_DIR,
        config.DATA_DIR,
        config.MODEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
        config.INPUT_DIR,
        config.OUTPUT_DIR,
        config.EXPORTS_DIR
    ]
    
    for path in paths_to_check:
        path_str = str(path).upper()
        assert path_str.startswith('G:'), f"Path not on G: drive: {path}"
    
    logger.info("✓ Config paths test passed - all on G: drive")


def test_directories_exist():
    """Test that all required directories exist"""
    assert config.DATA_DIR.exists()
    assert config.MODEL_DIR.exists()
    assert config.REPORTS_DIR.exists()
    assert config.LOGS_DIR.exists()
    
    logger.info("✓ Directories existence test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
