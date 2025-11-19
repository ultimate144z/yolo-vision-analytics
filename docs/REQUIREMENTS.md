# Project Requirements Documentation

## Functional Requirements

### FR-1: Video Input
- **FR-1.1**: System shall accept video files in MP4, AVI, MOV, MKV formats
- **FR-1.2**: System shall support webcam feed as input source
- **FR-1.3**: System shall handle videos up to 500 MB in size
- **FR-1.4**: System shall validate video format before processing

### FR-2: Object Detection
- **FR-2.1**: System shall detect objects using YOLOv8 model
- **FR-2.2**: System shall support configurable confidence threshold (default 0.5)
- **FR-2.3**: System shall track following object classes:
  - Person
  - Car
  - Bicycle
  - Motorcycle
  - Bus
  - Truck
- **FR-2.4**: System shall display bounding boxes around detected objects
- **FR-2.5**: System shall show confidence scores for each detection

### FR-3: Object Tracking & Counting
- **FR-3.1**: System shall assign unique IDs to tracked objects
- **FR-3.2**: System shall maintain object count by category
- **FR-3.3**: System shall track object trajectories across frames
- **FR-3.4**: System shall provide total detection count

### FR-4: Analytics
- **FR-4.1**: System shall generate time-series detection graphs
- **FR-4.2**: System shall create category-wise distribution charts
- **FR-4.3**: System shall compute detection frequency statistics
- **FR-4.4**: System shall identify peak activity periods

### FR-5: Heatmap Generation
- **FR-5.1**: System shall generate activity heatmaps
- **FR-5.2**: System shall identify high-traffic zones
- **FR-5.3**: System shall provide color-coded intensity visualization
- **FR-5.4**: System shall allow heatmap export as image file

### FR-6: Logging & Export
- **FR-6.1**: System shall log detections with timestamps
- **FR-6.2**: System shall export logs in JSON format
- **FR-6.3**: System shall export logs in CSV format
- **FR-6.4**: System shall generate PDF reports
- **FR-6.5**: System shall include metadata in exported files

### FR-7: Dashboard Interface
- **FR-7.1**: System shall provide web-based dashboard using Streamlit
- **FR-7.2**: System shall display real-time detection feed
- **FR-7.3**: System shall show live statistics and metrics
- **FR-7.4**: System shall provide interactive controls
- **FR-7.5**: System shall support parameter customization

## Non-Functional Requirements

### NFR-1: Performance
- **NFR-1.1**: System shall process video at minimum 10 FPS on CPU
- **NFR-1.2**: System shall process video at minimum 30 FPS on GPU
- **NFR-1.3**: System shall start detection within 5 seconds of input
- **NFR-1.4**: Dashboard shall update metrics within 1 second

### NFR-2: Usability
- **NFR-2.1**: Dashboard shall be accessible via web browser
- **NFR-2.2**: Interface shall be intuitive for non-technical users
- **NFR-2.3**: System shall provide clear error messages
- **NFR-2.4**: Documentation shall be comprehensive

### NFR-3: Reliability
- **NFR-3.1**: System shall handle corrupted video files gracefully
- **NFR-3.2**: System shall recover from detection failures
- **NFR-3.3**: System shall validate all inputs
- **NFR-3.4**: System shall log errors for debugging

### NFR-4: Maintainability
- **NFR-4.1**: Code shall follow PEP 8 style guidelines
- **NFR-4.2**: Functions shall be modular and reusable
- **NFR-4.3**: Configuration shall be externalized
- **NFR-4.4**: Code shall include docstrings and comments

### NFR-5: Scalability
- **NFR-5.1**: System shall support batch video processing
- **NFR-5.2**: System shall handle multiple concurrent users
- **NFR-5.3**: Architecture shall allow model upgrades

### NFR-6: Storage
- **NFR-6.1**: System shall use G: drive exclusively
- **NFR-6.2**: System shall not write to C: drive
- **NFR-6.3**: Virtual environment shall be on G: drive
- **NFR-6.4**: All outputs shall be stored on G: drive

## Technical Requirements

### TR-1: Software Dependencies
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV 4.8+
- Streamlit 1.28+
- Plotly 5.17+
- PyTorch 2.1+ (with CUDA optional)

### TR-2: Hardware Requirements
**Minimum**:
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 4 GB free on G: drive
- GPU: None (CPU fallback available)

**Recommended**:
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16 GB
- Storage: 10 GB free on G: drive
- GPU: NVIDIA GTX 1060 or better with CUDA support

### TR-3: Operating System
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

## User Stories

### US-1: Upload Video Detection
**As a** security analyst  
**I want to** upload recorded surveillance footage  
**So that** I can analyze past incidents for object activity

**Acceptance Criteria**:
- User can upload video via dashboard
- System displays detection results in real-time
- User can download detection report

### US-2: Live Webcam Monitoring
**As a** security operator  
**I want to** monitor live webcam feed  
**So that** I can detect objects in real-time

**Acceptance Criteria**:
- System accesses webcam successfully
- Detections appear with minimal latency
- User can start/stop monitoring

### US-3: Activity Analysis
**As a** facility manager  
**I want to** view heatmaps of activity zones  
**So that** I can optimize space utilization

**Acceptance Criteria**:
- Heatmap accurately represents activity
- High-traffic zones are clearly visible
- Heatmap can be exported

### US-4: Generate Reports
**As a** analyst  
**I want to** export detection data  
**So that** I can create presentations and reports

**Acceptance Criteria**:
- Reports include all detections with timestamps
- Multiple export formats available
- Reports are well-formatted

## Constraints

1. **Storage Constraint**: No writes to C: drive due to limited space (4 GB)
2. **Model Constraint**: YOLOv8 nano model for performance/size balance
3. **Format Constraint**: Limited to common video formats
4. **Processing Constraint**: Real-time performance depends on hardware

## Assumptions

1. Users have basic computer literacy
2. Python 3.8+ is installed on system
3. G: drive has sufficient free space
4. Internet connection available for initial setup
5. Webcam drivers are properly installed (if using webcam)

## Success Criteria

1. System successfully detects objects with >80% accuracy
2. Dashboard loads within 3 seconds
3. All reports export without errors
4. System runs for 8+ hours without crashes
5. Setup completes in under 30 minutes

