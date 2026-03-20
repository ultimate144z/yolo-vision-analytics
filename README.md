<p align="center">
  <h1 align="center">YOLO Vision Analytics</h1>
  <p align="center">Real-time object detection, tracking, and spatial analytics powered by YOLOv8</p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white" alt="Python 3.8+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT"></a>
  <a href="https://github.com/ultimate144z/yolo-vision-analytics/actions/workflows/ci.yml"><img src="https://github.com/ultimate144z/yolo-vision-analytics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Model-YOLOv8-0033A0?logo=yolo" alt="YOLOv8"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
</p>

---

A production-grade computer vision platform that combines **YOLOv8 object detection**, **real-time tracking**, and **interactive analytics** in a single Streamlit dashboard. Designed for security monitoring, traffic analysis, and operational video intelligence.

## What It Does

| Feature | Description |
|---------|-------------|
| **Real-time Detection** | Run YOLOv8 (nano to xlarge) on video files or live webcam feeds |
| **Object Tracking** | Persistent ID tracking across frames with track history |
| **Activity Heatmaps** | Gaussian density maps showing spatial activity patterns |
| **Zone Analytics** | 3x3 grid spatial analysis with hotspot detection |
| **Confidence Trends** | Temporal confidence analysis to surface anomalies |
| **Multi-format Export** | JSON, CSV, and PDF reports with one click |
| **Annotated Video** | Optional export of processed video with bounding boxes |
| **Batch Processing** | Process multiple video files sequentially |

## Quick Start

```bash
# Clone
git clone https://github.com/ultimate144z/yolo-vision-analytics.git
cd yolo-vision-analytics

# Setup
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install
pip install -r requirements.txt

# Run
streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. Model weights are **auto-downloaded** on first use.

## Architecture

```
yolo-vision-analytics/
  src/
    detection/
      yolo_detector.py         # YOLOv8 inference + tracking
      video_processor.py       # Video I/O, async writing, batch frames
    dashboard/
      app.py                   # Streamlit application
      components.py            # Reusable UI components
    utils/
      analytics.py             # Detection statistics engine
      advanced_analytics.py    # Zone, size, confidence analyzers
      heatmap_generator.py     # Gaussian heatmap generation
      report_generator.py      # JSON/CSV/PDF export
      fps_counter.py           # Real-time FPS tracking
      logger.py                # Rotating file + console logger
  config/
    config.py                  # Centralized configuration
  tests/                       # Unit tests
  data/                        # Input/output/logs (gitignored)
  models/                      # Model weights (auto-downloaded)
  reports/                     # Generated reports
```

## Configuration

All settings are in `config/config.py` and can be adjusted from the Streamlit sidebar at runtime:

| Setting | Default | Options |
|---------|---------|---------|
| Model | `yolov8n.pt` | `n` / `s` / `m` / `l` / `x` |
| Confidence | `0.5` | `0.1` - `1.0` (adjustable live) |
| Frame Skip | `1` | `1` / `2` / `3` / `5` |
| Classes | Vehicles & People | Default / All 80 / Custom |
| Device | Auto-detect | CUDA if available, else CPU |

## Performance Optimizations

- **TensorRT auto-export** on NVIDIA GPUs (up to 5x faster inference)
- **Batch inference** for video files (configurable batch size)
- **Async video writing** (non-blocking I/O via threaded queue)
- **Frame skipping** with smart seeking (2-5x throughput gain)
- **Cached model loading** (Streamlit `@st.cache_resource`)
- **Moving-average FPS** for stable performance monitoring

See [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) for detailed benchmarks and methodology.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src      # with coverage
```

## Tech Stack

- [**Ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics) - Object detection & tracking
- [**OpenCV**](https://opencv.org/) - Video processing
- [**Streamlit**](https://streamlit.io/) - Dashboard UI
- [**Plotly**](https://plotly.com/) - Interactive charts
- [**Pandas**](https://pandas.pydata.org/) / [**NumPy**](https://numpy.org/) - Data processing
- [**ReportLab**](https://www.reportlab.com/) - PDF generation

## Contributing

Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

## License

[MIT](LICENSE)

## Links

- [Project Guide](PROJECT_GUIDE.md) - Operational usage
- [Developer Guide](DEVELOPER_GUIDE.md) - Architecture & extension points
- [Optimizations](OPTIMIZATIONS.md) - Performance methodology

---

Built by [@ultimate144z](https://github.com/ultimate144z)
