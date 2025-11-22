# YOLO Vision Analytics – Real-Time Object Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models YOLOv8](https://img.shields.io/badge/Models-YOLOv8-navy)](https://github.com/ultralytics/ultralytics)
[![Framework Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Status Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)](#project-status)
[![Tests 100% Passing](https://img.shields.io/badge/Tests-100%25%20Passing-success)](tests/)
[![Last Updated](https://img.shields.io/badge/Updated-Nov%2022%202025-lightgrey)](#)
[![Issues](https://img.shields.io/badge/Issues-GitHub%20Tracker-informational)](https://github.com/ultimate144z/yolo-vision-analytics/issues)

Version: 2.0.0

A production-grade real-time object detection and analytics system built on YOLOv8, OpenCV, and Streamlit. Designed for security monitoring, traffic analysis, and operational video intelligence.

## Purpose & Scope

This README provides a high-level product overview. Operational usage lives in `PROJECT_GUIDE.md`. Architecture, extension points, and internal module detail live in `DEVELOPER_GUIDE.md`. Performance rationale, methodology, and benchmark artifacts live in `OPTIMIZATIONS.md`.

Capabilities:
- Real-time multi-class object detection (80 COCO classes)
- Configurable model selection (nano → xlarge)
- Frame sampling (skip factors for throughput scaling)
- Per-frame inference + FPS metrics (ETA removed—see Optimizations doc)
- Object counting & temporal distributions
- Spatial analytics: heatmaps & zone occupancy
- Size & confidence trend analytics
- Annotated video export (optional)
- Structured reporting (JSON / CSV / PDF)
- Stateless batch or interactive session execution

## Highlights in 2.0

### Performance
- Cached model initialization (milliseconds after first load)
- Moving-average FPS sampling (stable performance signal)
- Frame skipping (adaptive workload reduction)
- Five selectable model tiers (speed ↔ accuracy continuum)

### Analytics
- Zone analyzer (3×3 spatial grid activity)
- Size distribution & proximity inference
- Confidence temporal trend & anomaly surfacing

### User Experience
- Declarative sidebar configuration
- Real-time progress metrics (Avg FPS, inference latency, detections)
- Dynamic confidence threshold adjustment (updates in real-time)
- Flexible class selection (Default/All/Custom presets)
- Smooth progress tracking for all frame skip settings
- Cached navigation (no remount delays)
- Optional annotated export path

### Performance
- Efficient frame skipping (2-5x faster processing)
- Smart video seeking (skips reading unnecessary frames)
- Constant per-frame FPS regardless of skip setting
- Progress updates every frame for smooth UI feedback

Refer to `OPTIMIZATIONS.md` for methodology, deeper reasoning, and benchmark protocol.

## Tech Stack

- **YOLOv8** (Ultralytics) - Object detection model
- **OpenCV** - Video processing and computer vision
- **Streamlit** - Interactive dashboard UI
- **Plotly** - Interactive data visualization
- **Pandas** - Data analysis and manipulation
- **NumPy** - Numerical computations

## Project Structure (Condensed)

```
yolov8_security_monitor/
 src/
    detection/
       yolo_detector.py       # YOLOv8 detection logic
       video_processor.py     # Video stream handling
    dashboard/
       app.py                 # Main Streamlit app
       components.py          # Dashboard UI components
    utils/
        logger.py              # Logging utilities
        analytics.py           # Analytics computation
        heatmap_generator.py   # Heatmap generation
        report_generator.py    # Report export functionality
 data/
    input/                     # Input videos
    output/                    # Processed videos
    logs/                      # Detection logs (JSON/CSV)
 models/                        # YOLOv8 model weights
 config/
    config.py                  # Configuration settings
 reports/
    exports/                   # Generated reports
 tests/                         # Unit tests
 docs/                          # Documentation

```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam (optional, for live detection)

### Installation

**Clone the repository:**
```bash
git clone https://github.com/ultimate144z/yolo-vision-analytics.git
cd yolo-vision-analytics
```

**Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Model Weights:** Auto-downloaded on first selection (no manual step required). Larger weights are only fetched if explicitly chosen, minimizing footprint.

### Running the Application

```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Launch the dashboard
streamlit run src/dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

## Key Features

### 1. Real-Time Detection
- Video file or webcam ingestion
- Responsive bounding boxes & class labels
- Confidence scores filtered by threshold

### 2. Object Analytics
- Distribution by class & time segment
- Temporal frequency analysis
- Confidence dispersion metrics

### 3. Activity Heatmaps
- Spatial density mapping
- Zone ranking & hotspot surfacing
- Exportable static artifacts

### 4. Detection Logging
- Structured event capture (frame, class, confidence, region)
- Multi-format export pipeline
- Class-level filtering

### 5. Interactive Dashboard
- Parameterized session control
- Real-time metric and analytics panels
- Export orchestration tab

## Configuration

Edit `config/config.py` to customize:

```python
# Model settings
YOLO_MODEL = "yolov8n.pt"  # Choose: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Tracked object classes
TRACKED_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
```

## Usage Examples

### Upload Video
1. Launch dashboard
2. Click "Upload Video" in sidebar
3. Select MP4/AVI/MOV file
4. Click "Start Detection"

### Use Webcam
1. Launch dashboard
2. Select "Use Webcam" option
3. Grant camera permissions
4. Click "Start Detection"

### Generate Reports
1. Complete detection session
2. Navigate to "Analytics" tab
3. Select report format (CSV/JSON/PDF)
4. Click "Download Report"

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Processing Flow

```
Video Input → YOLOv8 Detection → Object Tracking → Analytics Generation → Dashboard Visualization
                                                                              ↓
                                                  Logs (JSON/CSV) ← Report Generator
```

## Engineering Focus Areas

This project demonstrates:
- Computer vision with deep learning
- Real-time video processing
- Web application development
- Data visualization
- Software engineering best practices

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [OpenCV](https://opencv.org/) community
- [Streamlit](https://streamlit.io/) team

## Author & Contacts

**ultimate144z**
- GitHub: [@ultimate144z](https://github.com/ultimate144z)
- Project: [YOLO Vision Analytics](https://github.com/ultimate144z/yolo-vision-analytics)

---

## Project Status

**Version 2.0** - Production Ready

**Feature Set:**
- Real-time detection & logging
- Video + webcam ingestion
- Five model tiers (speed/accuracy dial)
- Frame sampling for throughput scaling
- Advanced analytics (zones / size / confidence trends)
- Annotated video export option
- Multi-format reporting stack
- Interactive Streamlit dashboard
- Test suite (10/10 passing)

Last Updated: November 22, 2025

---
For detailed operational guidance: `PROJECT_GUIDE.md`  
For architecture & extension points: `DEVELOPER_GUIDE.md`  
For optimization methodology: `OPTIMIZATIONS.md`
