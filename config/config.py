import os
from pathlib import Path

# Base paths - All on G: drive, never use C: drive
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
EXPORTS_DIR = REPORTS_DIR / "exports"

# Ensure all directories exist on G: drive
for directory in [DATA_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR, INPUT_DIR, OUTPUT_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings
YOLO_MODEL = "yolov8n.pt"  # nano model for speed (n, s, m, l, x)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"  # Auto-detect GPU

# Video settings
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
FPS_TARGET = 30

# Detection settings
TRACKED_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
YOLO_ALL_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Tracking settings
MAX_TRACK_AGE = 30  # frames
MIN_TRACK_HITS = 3  # minimum detections to confirm track

# Heatmap settings
HEATMAP_KERNEL_SIZE = 51
HEATMAP_SIGMA = 25
HEATMAP_COLORMAP = 'jet'

# Dashboard settings
STREAMLIT_PORT = 8501
PAGE_TITLE = "YOLOv8 Security Monitor"
PAGE_ICON = ":camera:"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"
MAX_LOG_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

# Performance settings
BATCH_SIZE = 1
MAX_WORKERS = 4
MEMORY_LIMIT_MB = 2048

# Export settings
EXPORT_FORMATS = ['json', 'csv', 'pdf']
INCLUDE_METADATA = True
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# Webcam settings
WEBCAM_INDEX = 0
WEBCAM_RESOLUTION = (640, 480)
WEBCAM_FPS = 30
