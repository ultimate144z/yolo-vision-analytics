"""
Logger utility for YOLOv8 Security Monitor
Provides file and console logging with rotation
All logs stored on G: drive only
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import config.config as config


class SecurityLogger:
    """
    Custom logger class with file rotation and console output
    Ensures all logs are written to G: drive
    """
    
    def __init__(self, name: str, log_file: str = None):
        """
        Initialize logger with file and console handlers
        
        Args:
            name: Logger name
            log_file: Optional custom log file path (on G: drive)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler with rotation (on G: drive)
        if log_file is None:
            log_file = config.LOG_FILE
        else:
            log_file = Path(log_file)
            # Ensure path is on G: drive
            if not str(log_file).startswith('G:') and not str(log_file).startswith('g:'):
                log_file = config.LOGS_DIR / log_file.name
        
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.MAX_LOG_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(config.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(message)


def get_logger(name: str = __name__, log_file: str = None) -> SecurityLogger:
    """
    Factory function to get logger instance
    
    Args:
        name: Logger name
        log_file: Optional custom log file path
    
    Returns:
        SecurityLogger instance
    """
    return SecurityLogger(name, log_file)


def log_detection_event(logger: SecurityLogger, frame_num: int, detections: list):
    """
    Log detection event with frame number and detection count
    
    Args:
        logger: SecurityLogger instance
        frame_num: Frame number
        detections: List of detections
    """
    logger.debug(f"Frame {frame_num}: {len(detections)} objects detected")


def log_system_info(logger: SecurityLogger):
    """
    Log system information at startup
    
    Args:
        logger: SecurityLogger instance
    """
    import torch
    import cv2
    
    logger.info("="*50)
    logger.info("YOLOv8 Security Monitor - System Information")
    logger.info("="*50)
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"OpenCV Version: {cv2.__version__}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Base Directory: {config.BASE_DIR}")
    logger.info(f"Model Directory: {config.MODEL_DIR}")
    logger.info(f"Data Directory: {config.DATA_DIR}")
    logger.info(f"Log File: {config.LOG_FILE}")
    logger.info("="*50)


# Create default logger instance
default_logger = get_logger("SecurityMonitor")
