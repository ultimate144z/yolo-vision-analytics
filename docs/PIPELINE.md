# Project Pipeline Documentation

## System Architecture

### 1. Data Input Layer
```
┌─────────────────────────────────────┐
│      Input Sources                  │
│  ┌──────────┐      ┌──────────┐   │
│  │ Webcam   │      │  Video   │   │
│  │  Feed    │      │  Upload  │   │
│  └──────────┘      └──────────┘   │
└─────────────────────────────────────┘
           ↓
```

### 2. Processing Layer
```
┌─────────────────────────────────────┐
│   Video Processing Pipeline         │
│                                     │
│  ┌────────────────────────────┐   │
│  │  Frame Extraction          │   │
│  └────────────────────────────┘   │
│              ↓                     │
│  ┌────────────────────────────┐   │
│  │  YOLOv8 Detection Engine   │   │
│  │  - Object Detection        │   │
│  │  - Bounding Box Gen        │   │
│  │  - Confidence Scoring      │   │
│  └────────────────────────────┘   │
│              ↓                     │
│  ┌────────────────────────────┐   │
│  │  Object Tracking           │   │
│  │  - ID Assignment           │   │
│  │  - Trajectory Tracking     │   │
│  └────────────────────────────┘   │
└─────────────────────────────────────┘
           ↓
```

### 3. Analytics Layer
```
┌─────────────────────────────────────┐
│      Analytics Engine               │
│                                     │
│  ┌───────────┐  ┌───────────┐     │
│  │  Object   │  │ Heatmap   │     │
│  │ Counting  │  │Generator  │     │
│  └───────────┘  └───────────┘     │
│        ↓              ↓            │
│  ┌───────────┐  ┌───────────┐     │
│  │Time-Series│  │ Activity  │     │
│  │ Analysis  │  │  Zones    │     │
│  └───────────┘  └───────────┘     │
└─────────────────────────────────────┘
           ↓
```

### 4. Presentation Layer
```
┌─────────────────────────────────────┐
│    Streamlit Dashboard              │
│                                     │
│  ┌────────────────────────────┐   │
│  │  Real-time Visualization   │   │
│  │  - Live Video Feed         │   │
│  │  - Detection Overlays      │   │
│  └────────────────────────────┘   │
│              ↓                     │
│  ┌────────────────────────────┐   │
│  │  Interactive Charts        │   │
│  │  - Plotly Graphs           │   │
│  │  - Heatmaps                │   │
│  └────────────────────────────┘   │
│              ↓                     │
│  ┌────────────────────────────┐   │
│  │  Report Generator          │   │
│  │  - JSON/CSV Export         │   │
│  │  - PDF Reports             │   │
│  └────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 5. Storage Layer
```
┌─────────────────────────────────────┐
│      Data Persistence               │
│                                     │
│  ┌───────────┐  ┌───────────┐     │
│  │Detection  │  │ Processed │     │
│  │   Logs    │  │  Videos   │     │
│  │(JSON/CSV) │  │           │     │
│  └───────────┘  └───────────┘     │
└─────────────────────────────────────┘
```

## Data Flow

1. **Input**: Video source (webcam/file) → Frame buffer
2. **Detection**: Frames → YOLOv8 → Detected objects + coordinates
3. **Tracking**: Detected objects → Tracking algorithm → Object IDs + trajectories
4. **Analytics**: Detection data → Aggregation → Statistics + heatmaps
5. **Visualization**: Analytics → Streamlit → Interactive dashboard
6. **Storage**: Detection logs → File system → JSON/CSV files

## Module Dependencies

```
app.py (Dashboard)
  ├── yolo_detector.py
  │     └── YOLOv8 model
  ├── video_processor.py
  │     └── OpenCV
  ├── analytics.py
  │     └── Pandas, NumPy
  ├── heatmap_generator.py
  │     └── OpenCV, Matplotlib
  └── report_generator.py
        └── Pandas, ReportLab
```

## Key Algorithms

### Object Detection
- **Model**: YOLOv8 (You Only Look Once v8)
- **Input**: RGB frames (640x640 default)
- **Output**: Bounding boxes, class labels, confidence scores

### Heatmap Generation
- **Method**: Gaussian kernel density estimation
- **Input**: Detection coordinates
- **Output**: 2D heatmap array

### Object Counting
- **Method**: Dictionary-based aggregation
- **Tracking**: Frame-by-frame ID matching
- **Persistence**: Rolling window statistics

## Performance Considerations

- **YOLOv8 Nano**: ~45 FPS on GPU, ~10 FPS on CPU
- **Frame Resolution**: 640x640 (adjustable)
- **Memory Usage**: ~2-4 GB during processing
- **Storage**: ~100 MB per 10-minute video processed

