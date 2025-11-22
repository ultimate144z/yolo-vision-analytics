"""
Streamlit Dashboard Components
Reusable UI components for the security monitor dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import cv2
from PIL import Image

import config.config as config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def display_header():
    """Display dashboard header with branding"""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
    st.markdown("Real-time object detection and analytics powered by YOLOv8")
    st.markdown("---")


def display_metrics(detection_summary: Dict[str, int], col_count: int = 4):
    """
    Display detection metrics in columns
    
    Args:
        detection_summary: Dictionary with class counts
        col_count: Number of columns to display
    """
    if not detection_summary:
        st.info("No detections yet. Start processing to see metrics.")
        return
    
    # Calculate total
    total = sum(detection_summary.values())
    
    # Create columns
    cols = st.columns(col_count)
    
    # Display total first with emphasis
    with cols[0]:
        st.metric("Total Detections", total, delta=None)
        st.markdown("---")
    
    # Display class counts
    col_idx = 1
    for class_name, count in detection_summary.items():
        with cols[col_idx % col_count]:
            st.metric(class_name.capitalize(), count)
        col_idx += 1


def display_video_frame(frame: np.ndarray, caption: str = "Detection Feed"):
    """
    Display video frame in Streamlit
    
    Args:
        frame: Frame array (BGR format)
        caption: Caption for the frame
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display
    st.image(frame_rgb, caption=caption, use_column_width=True)


def plot_class_distribution(class_counts: Dict[str, int]):
    """
    Create pie chart for class distribution
    
    Args:
        class_counts: Dictionary mapping class names to counts
    """
    if not class_counts:
        st.info("No data available for visualization")
        return
    
    # Prepare data
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(
            colors=px.colors.qualitative.Set3
        )
    )])
    
    fig.update_layout(
        title="Detection Distribution by Class",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_detection_timeline(timeseries_data: Dict[str, List[int]], 
                           window_labels: List[str] = None):
    """
    Create line chart for detection frequency over time
    
    Args:
        timeseries_data: Dictionary mapping class names to frequency lists
        window_labels: Optional labels for time windows
    """
    if not timeseries_data:
        st.info("No timeline data available")
        return
    
    fig = go.Figure()
    
    for class_name, frequencies in timeseries_data.items():
        if window_labels is None:
            x_values = list(range(len(frequencies)))
        else:
            x_values = window_labels[:len(frequencies)]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=frequencies,
            mode='lines+markers',
            name=class_name.capitalize(),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Detection Frequency Over Time",
        xaxis_title="Time Window",
        yaxis_title="Detection Count",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_distribution(confidence_stats: Dict[str, Dict[str, float]]):
    """
    Create bar chart for confidence statistics
    
    Args:
        confidence_stats: Dictionary with confidence statistics per class
    """
    if not confidence_stats:
        st.info("No confidence data available")
        return
    
    classes = list(confidence_stats.keys())
    mean_conf = [stats['mean'] for stats in confidence_stats.values()]
    min_conf = [stats['min'] for stats in confidence_stats.values()]
    max_conf = [stats['max'] for stats in confidence_stats.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Mean Confidence',
        x=classes,
        y=mean_conf,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Average Confidence by Class",
        xaxis_title="Object Class",
        yaxis_title="Confidence Score",
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_heatmap(heatmap: np.ndarray, title: str = "Activity Heatmap"):
    """
    Display heatmap visualization
    
    Args:
        heatmap: 2D heatmap array
        title: Title for the heatmap
    """
    fig = px.imshow(
        heatmap,
        color_continuous_scale='hot',
        labels=dict(x="X Position", y="Y Position", color="Intensity"),
        title=title
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)


def create_settings_sidebar():
    """
    Create sidebar with settings controls
    
    Returns:
        Dictionary with selected settings
    """
    st.sidebar.markdown("### Settings")
    st.sidebar.markdown("---")
    
    settings = {}
    
    # Model selection
    st.sidebar.markdown("#### Model Selection")
    model_options = {
        "YOLOv8 Nano (Fast)": "yolov8n.pt",
        "YOLOv8 Small": "yolov8s.pt",
        "YOLOv8 Medium": "yolov8m.pt",
        "YOLOv8 Large (Accurate)": "yolov8l.pt",
        "YOLOv8 XLarge (Best)": "yolov8x.pt"
    }
    selected_model_name = st.sidebar.selectbox(
        "Choose YOLO model:",
        list(model_options.keys()),
        index=0,
        help="Nano is fastest, XLarge is most accurate but slower"
    )
    settings['model_name'] = model_options[selected_model_name]
    
    st.sidebar.markdown("---")
    
    # Video source selection
    st.sidebar.markdown("#### Video Source")
    source_type = st.sidebar.radio(
        "Select source:",
        ["Upload Video", "Use Webcam"]
    )
    settings['source_type'] = source_type
    
    if source_type == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        settings['video_file'] = uploaded_file
    else:
        # Detect available cameras
        from src.detection.video_processor import get_available_cameras
        available_cams = get_available_cameras()
        
        if available_cams:
            st.sidebar.info(f"Detected cameras: {', '.join(map(str, available_cams))}")
            default_cam = available_cams[0] if available_cams else 0
        else:
            st.sidebar.warning("No cameras detected. Enter index manually.")
            default_cam = 0
        
        webcam_index = st.sidebar.number_input(
            "Webcam Index",
            min_value=0,
            max_value=5,
            value=default_cam,
            help="System will auto-detect available cameras on connection"
        )
        settings['webcam_index'] = webcam_index
    
    st.sidebar.markdown("---")
    
    # Detection settings
    st.sidebar.markdown("#### Detection Settings")
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=config.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Lower values detect more objects but with more false positives"
    )
    settings['confidence_threshold'] = confidence
    
    # Performance optimization
    frame_skip = st.sidebar.select_slider(
        "Processing Speed",
        options=[1, 2, 3, 5],
        value=1,
        format_func=lambda x: f"Every {x} frame{'s' if x > 1 else ''}" if x > 1 else "All frames (slowest)",
        help="Skip frames for faster processing. Higher = faster but less detailed."
    )
    settings['frame_skip'] = frame_skip
    
    # Class selection
    st.sidebar.markdown("#### Tracked Classes")
    
    # Preset options
    class_preset = st.sidebar.selectbox(
        "Class Preset:",
        ["Default (Vehicles & People)", "All Classes (80)", "Custom Selection"],
        help="Choose which object classes to detect"
    )
    
    if class_preset == "Default (Vehicles & People)":
        selected_classes = config.TRACKED_CLASSES
    elif class_preset == "All Classes (80)":
        selected_classes = config.YOLO_ALL_CLASSES
    else:  # Custom Selection
        st.sidebar.markdown("**Select Classes to Track:**")
        
        # Group classes by category for better UX
        categories = {
            "People & Animals": ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird'],
            "Vehicles": ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
            "Furniture": ['chair', 'couch', 'bed', 'dining table', 'toilet'],
            "Electronics": ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster'],
            "Food & Kitchen": ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
            "Sports & Outdoors": ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
            "Other": ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'backpack',
                     'umbrella', 'handbag', 'tie', 'suitcase', 'potted plant', 'book', 'clock', 'vase', 
                     'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'refrigerator', 'sink']
        }
        
        selected_classes = []
        
        # Use expanders for each category
        for category, classes in categories.items():
            with st.sidebar.expander(f"**{category}** ({len(classes)} classes)"):
                select_all = st.checkbox(f"Select All {category}", key=f"select_all_{category}")
                
                for cls in classes:
                    default_checked = cls in config.TRACKED_CLASSES or select_all
                    if st.checkbox(cls.capitalize(), value=default_checked, key=f"track_cls_{cls}"):
                        selected_classes.append(cls)
    
    # Display selected count with color-coded feedback
    if len(selected_classes) == 0:
        st.sidebar.error(f"⚠️ No classes selected")
    elif len(selected_classes) <= 10:
        st.sidebar.success(f"✓ {len(selected_classes)} classes selected")
    else:
        st.sidebar.info(f"✓ {len(selected_classes)} classes selected")
    
    settings['tracked_classes'] = selected_classes
    
    st.sidebar.markdown("---")
    
    # Display options
    st.sidebar.markdown("#### Display Options")
    
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
    show_track_id = st.sidebar.checkbox("Show Track ID", value=True)
    
    settings['show_confidence'] = show_confidence
    settings['show_track_id'] = show_track_id
    
    st.sidebar.markdown("---")
    
    # Export settings
    st.sidebar.markdown("#### Export Options")
    
    export_formats = []
    if st.sidebar.checkbox("JSON", value=True, key="sidebar_export_json"):
        export_formats.append('json')
    if st.sidebar.checkbox("CSV", value=True, key="sidebar_export_csv"):
        export_formats.append('csv')
    if st.sidebar.checkbox("PDF", value=False, key="sidebar_export_pdf"):
        export_formats.append('pdf')
    
    settings['export_formats'] = export_formats
    
    # Video export option
    export_video = st.sidebar.checkbox(
        "Save Annotated Video", 
        value=False, 
        key="sidebar_export_video",
        help="Save processed video with detection boxes (increases processing time)"
    )
    settings['export_video'] = export_video
    
    return settings


def display_progress_bar(progress: float, text: str = "Processing..."):
    """
    Display progress bar
    
    Args:
        progress: Progress value (0-100)
        text: Progress text
    """
    progress_normalized = min(max(progress / 100.0, 0.0), 1.0)
    st.progress(progress_normalized, text=f"{text} ({progress:.1f}%)")


def display_statistics_table(stats: Dict[str, any]):
    """
    Display statistics in a formatted table
    
    Args:
        stats: Dictionary with statistics
    """
    if not stats:
        st.info("No statistics available")
        return
    
    # Convert to DataFrame for nice display
    data = []
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            data.append({'Metric': key.replace('_', ' ').title(), 'Value': value})
    
    if data:
        df = pd.DataFrame(data)
        st.table(df)


def create_download_buttons(file_paths: Dict[str, str]):
    """
    Create download buttons for exported files
    
    Args:
        file_paths: Dictionary mapping format names to file paths
    """
    if not file_paths:
        st.warning("No files available for download")
        return
    
    st.markdown("### Download Reports")
    
    cols = st.columns(len(file_paths))
    
    for idx, (format_name, file_path) in enumerate(file_paths.items()):
        with cols[idx]:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            st.download_button(
                label=f"Download {format_name.upper()}",
                data=file_data,
                file_name=file_path.name if hasattr(file_path, 'name') else f"report.{format_name}",
                mime=get_mime_type(format_name)
            )


def get_mime_type(format_name: str) -> str:
    """
    Get MIME type for file format
    
    Args:
        format_name: File format name
    
    Returns:
        MIME type string
    """
    mime_types = {
        'json': 'application/json',
        'csv': 'text/csv',
        'pdf': 'application/pdf'
    }
    return mime_types.get(format_name.lower(), 'application/octet-stream')


def display_alert(message: str, alert_type: str = "info"):
    """
    Display styled alert message
    
    Args:
        message: Alert message
        alert_type: Type of alert ('info', 'success', 'warning', 'error')
    """
    if alert_type == "info":
        st.info(message)
    elif alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    elif alert_type == "error":
        st.error(message)
    else:
        st.write(message)


def display_detection_grid(detections_by_class: Dict[str, int], grid_cols: int = 3):
    """
    Display detections in a grid layout
    
    Args:
        detections_by_class: Dictionary with detection counts per class
        grid_cols: Number of columns in grid
    """
    if not detections_by_class:
        st.info("No detections to display")
        return
    
    # Create grid
    items = list(detections_by_class.items())
    rows = [items[i:i+grid_cols] for i in range(0, len(items), grid_cols)]
    
    for row in rows:
        cols = st.columns(grid_cols)
        for idx, (class_name, count) in enumerate(row):
            with cols[idx]:
                st.markdown(f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 5px;
                ">
                    <h3 style="margin: 0;">{count}</h3>
                    <p style="margin: 5px 0 0 0;">{class_name.capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)


def create_tabs(tab_names: List[str]):
    """
    Create tabs for different dashboard sections
    
    Args:
        tab_names: List of tab names
    
    Returns:
        List of tab objects
    """
    return st.tabs(tab_names)


def display_system_info():
    """Display system information in expander"""
    with st.expander("System Information", expanded=False):
        import torch
        import cv2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Software Versions:**")
            st.write(f"- OpenCV: {cv2.__version__}")
            st.write(f"- PyTorch: {torch.__version__}")
            st.write(f"- CUDA Available: {torch.cuda.is_available()}")
        
        with col2:
            st.write("**Configuration:**")
            st.write(f"- Model: {config.YOLO_MODEL}")
            st.write(f"- Device: {config.DEVICE}")
            st.write(f"- Base Path: {config.BASE_DIR}")


def display_performance_metrics(metrics: Dict[str, any]):
    """
    Display performance metrics
    
    Args:
        metrics: Dictionary with performance metrics
    """
    st.markdown("### Performance Metrics")
    
    cols = st.columns(4)
    
    with cols[0]:
        fps = metrics.get('average_fps', 0)
        st.metric("Average FPS", f"{fps:.2f}")
    
    with cols[1]:
        frame_time = metrics.get('avg_frame_time_ms', 0)
        st.metric("Frame Time", f"{frame_time:.2f} ms")
    
    with cols[2]:
        inference_time = metrics.get('avg_inference_time_ms', 0)
        st.metric("Inference Time", f"{inference_time:.2f} ms")
    
    with cols[3]:
        total_frames = metrics.get('total_frames', 0)
        st.metric("Total Frames", total_frames)
