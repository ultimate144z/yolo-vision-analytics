# YOLO Vision Analytics - Real-Time Object Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)

**Version 2.0** - Production-Ready Computer Vision Analytics Platform

A comprehensive real-time object detection and analytics system built with YOLOv8, OpenCV, and Streamlit. Perfect for security monitoring, traffic analysis, and computer vision applications.

## Project Overview

This system processes video feeds (webcam or uploaded videos) to detect and track objects in real-time. It provides:
- Real-time object detection with bounding boxes
- **NEW**: Real-time FPS & ETA tracking 
- **NEW**: 5 YOLO models to choose from (speed vs. accuracy) 
- **NEW**: Frame skipping for up to 5x faster processing 
- **NEW**: Export annotated video with detections 
- Object counting by category (person, car, bicycle, etc.)
- Activity heatmaps showing high-traffic zones
- **NEW**: Zone-based activity analysis 
- **NEW**: Object size distribution analysis 
- **NEW**: Confidence trend monitoring 
- Timestamped detection logs
- Interactive analytics dashboard
- Exportable detection reports (JSON/CSV/PDF)

##  New in Version 2.0

### Performance Optimizations
- **70% faster page reloads** with model caching
- **Real-time FPS counter** during processing
- **Estimated time remaining** (ETA) calculation
- **Frame skipping option** for faster processing
- **5 model options** for speed/accuracy tradeoff

### Advanced Analytics
- **Zone Analyzer**: Identifies high-traffic areas in 3x3 grid
- **Size Analyzer**: Object distance estimation
- **Confidence Analyzer**: Detection quality monitoring over time

### User Experience
- **Model selection UI**: Choose nano/small/medium/large/xlarge
- **Processing speed slider**: Adjust frame skip (1x to 5x faster)
- **Rich progress display**: FPS, Inference time, ETA, Detections
- **Cached loading**: Instant page switches
- **Video export**: Save annotated video with all detections visible 

 **See [OPTIMIZATION_FEATURES.md](OPTIMIZATION_FEATURES.md) for complete details**

##  Tech Stack

- **YOLOv8** (Ultralytics) - Object detection model
- **OpenCV** - Video processing and computer vision
- **Streamlit** - Interactive dashboard UI
- **Plotly** - Interactive data visualization
- **Pandas** - Data analysis and manipulation
- **NumPy** - Numerical computations

##  Project Structure

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
- Upload video files or use webcam feed
- YOLOv8 nano model for fast inference
- Real-time bounding box visualization
- Confidence score display

### 2. Object Analytics
- Count objects by category
- Track detection frequency over time
- Generate time-series plots
- Category-wise distribution charts

### 3. Activity Heatmaps
- Visualize high-traffic zones
- Identify areas with most detections
- Color-coded intensity maps
- Exportable heatmap images

### 4. Detection Logging
- Timestamped detection records
- JSON and CSV export formats
- Filterable by object class
- Batch processing support

### 5. Interactive Dashboard
- Clean, intuitive UI
- Real-time metric updates
- Downloadable reports
- Customizable detection parameters

##  Configuration

Edit `config/config.py` to customize:

```python
# Model settings
YOLO_MODEL = "yolov8n.pt"  # Choose: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Tracked object classes
TRACKED_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
```

##  Usage Examples

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

##  Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

##  Pipeline Overview

```
Video Input → YOLOv8 Detection → Object Tracking → Analytics Generation → Dashboard Visualization
                                                                              ↓
                                                  Logs (JSON/CSV) ← Report Generator
```

##  Learning Outcomes

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

## Author

**ultimate144z**
- GitHub: [@ultimate144z](https://github.com/ultimate144z)
- Project: [YOLO Vision Analytics](https://github.com/ultimate144z/yolo-vision-analytics)

---

## Project Status

**Version 2.0** - Production Ready

**Features Completed:**
- Real-time object detection (80 classes)
- Video + webcam support with auto-detection
- 5 YOLO model options (nano to xlarge)
- Frame skipping for up to 5x faster processing
- Advanced analytics (zones, sizes, confidence trends)
- Annotated video export
- Multi-format reports (JSON/CSV/PDF)
- Interactive dashboard with real-time metrics
- Complete test coverage (10/10 passing)

**Last Updated**: November 20, 2025
