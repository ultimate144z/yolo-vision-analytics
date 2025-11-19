"""
Main Streamlit Dashboard Application
Real-time object detection and analytics interface
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import config.config as config
from src.detection.yolo_detector import YOLODetector
from src.detection.video_processor import VideoProcessor
from src.utils.analytics import DetectionAnalytics, PerformanceMetrics
from src.utils.heatmap_generator import HeatmapGenerator
from src.utils.report_generator import ReportGenerator
from src.utils.logger import get_logger, log_system_info
from src.utils.fps_counter import FPSCounter
from src.utils.advanced_analytics import ZoneAnalyzer, ObjectSizeAnalyzer, ConfidenceTemporalAnalyzer
from src.dashboard import components

logger = get_logger(__name__)


@st.cache_resource
def load_detector_cached(model_name: str, confidence: float, tracked_classes: tuple):
    """
    Cached detector initialization for performance
    Uses Streamlit's caching to prevent model reloading
    """
    logger.info(f"Loading detector with model: {model_name}")
    return YOLODetector(
        model_name=model_name,
        confidence_threshold=confidence,
        tracked_classes=list(tracked_classes)
    )


class SecurityMonitorApp:
    """Main application class for the security monitor dashboard"""
    
    def __init__(self):
        """Initialize application"""
        self.detector = None
        self.analytics = None
        self.heatmap_gen = None
        self.report_gen = None
        self.performance = None
        self.is_processing = False
        self.fps_counter = FPSCounter()
        self.zone_analyzer = None
        self.size_analyzer = None
        self.confidence_analyzer = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'detector_initialized' not in st.session_state:
            st.session_state.detector_initialized = False
        
        if 'detections_history' not in st.session_state:
            st.session_state.detections_history = []
        
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = None
        
        if 'heatmap_data' not in st.session_state:
            st.session_state.heatmap_data = None
        
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = None
        
        if 'advanced_analytics' not in st.session_state:
            st.session_state.advanced_analytics = None
        
        if 'output_video_path' not in st.session_state:
            st.session_state.output_video_path = None
        
        if 'stop_processing' not in st.session_state:
            st.session_state.stop_processing = False
    
    def initialize_detector(self, model_name: str, confidence_threshold: float, tracked_classes: list):
        """
        Initialize YOLO detector with caching
        
        Args:
            model_name: YOLO model name
            confidence_threshold: Confidence threshold
            tracked_classes: List of classes to track
        """
        try:
            with st.spinner("Loading YOLOv8 model..."):
                # Use cached detector for performance
                self.detector = load_detector_cached(
                    model_name,
                    confidence_threshold,
                    tuple(tracked_classes)  # Must be hashable for caching
                )
                st.session_state.detector_initialized = True
                logger.info("Detector initialized successfully")
                components.display_alert("Model loaded successfully!", "success")
        
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            components.display_alert(f"Error loading model: {str(e)}", "error")
            raise
    
    def process_video(self, video_source, settings):
        """
        Process video with object detection
        
        Args:
            video_source: Video file path or webcam index
            settings: Settings dictionary from UI
        """
        try:
            # Initialize components
            self.analytics = DetectionAnalytics()
            self.performance = PerformanceMetrics()
            self.report_gen = ReportGenerator()
            self.fps_counter.reset()
            
            # Create placeholders for live updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            metrics_row_placeholder = st.empty()
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Start performance tracking
            self.performance.start()
            
            # Setup video writer if export is enabled
            video_writer = None
            output_video_path = None
            if settings.get('export_video', False):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = config.OUTPUT_DIR / f"annotated_video_{timestamp}.mp4"
                logger.info(f"Video export enabled. Output: {output_video_path}")
            
            # Process video
            with VideoProcessor(video_source) as processor:
                video_info = processor.get_video_info()
                logger.info(f"Processing video: {video_info}")
                
                # Initialize video writer if export enabled
                if settings.get('export_video', False):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(output_video_path),
                        fourcc,
                        video_info.get('fps', 30),
                        (video_info['width'], video_info['height'])
                    )
                
                # Initialize advanced analyzers
                self.heatmap_gen = HeatmapGenerator(
                    video_info['width'],
                    video_info['height']
                )
                self.zone_analyzer = ZoneAnalyzer(
                    video_info['width'],
                    video_info['height'],
                    grid_size=(3, 3)
                )
                self.size_analyzer = ObjectSizeAnalyzer()
                self.confidence_analyzer = ConfidenceTemporalAnalyzer()
                
                frame_count = 0
                frame_skip = settings.get('frame_skip', 1)  # Process every Nth frame
                
                # Calculate actual frames to process with frame skip
                total_video_frames = video_info.get('frame_count', 0)
                frames_to_process = total_video_frames // frame_skip if frame_skip > 1 else total_video_frames
                
                logger.info(f"Total video frames: {total_video_frames}, Frame skip: {frame_skip}, Frames to process: {frames_to_process}")
                
                for frame, frame_num, timestamp in processor.frames():
                    # Skip frames if configured
                    if frame_num % frame_skip != 0:
                        continue
                    
                    frame_start_time = time.time()
                    
                    # Run detection
                    inference_start = time.time()
                    detections = self.detector.detect(frame, track=True, timestamp=timestamp)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # Add to analytics
                    self.analytics.add_detections(detections, frame_num)
                    self.heatmap_gen.add_detections(detections)
                    
                    # Advanced analytics
                    for det in detections:
                        self.zone_analyzer.add_detection(det)
                        self.size_analyzer.add_detection(det)
                        self.confidence_analyzer.add_detection(det, frame_num)
                    
                    # Draw detections
                    annotated_frame = self.detector.draw_detections(
                        frame,
                        detections,
                        show_confidence=settings['show_confidence'],
                        show_track_id=settings['show_track_id']
                    )
                    
                    # Write frame to output video if enabled
                    if video_writer is not None:
                        video_writer.write(annotated_frame)
                    
                    # Update FPS counter
                    current_fps = self.fps_counter.update()
                    
                    # Update UI every 10 frames
                    if frame_num % 10 == 0:
                        # Performance metrics row
                        with metrics_row_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("FPS", f"{current_fps:.1f}")
                            with col2:
                                st.metric("Inference", f"{inference_time:.0f}ms")
                            with col3:
                                st.metric("Detections", len(detections))
                        
                        # Update frame display
                        frame_placeholder.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_num}",
                            use_column_width=True
                        )
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            components.display_metrics(self.detector.get_detection_summary())
                        
                        # Update progress
                        progress = processor.get_progress()
                        if progress > 0:
                            progress_placeholder.progress(
                                progress / 100.0,
                                text=f"Processing: {progress:.1f}% | FPS: {current_fps:.1f}"
                            )
                    
                    # Track performance
                    frame_time = (time.time() - frame_start_time) * 1000
                    self.performance.add_frame_time(frame_time)
                    self.performance.add_inference_time(inference_time)
                    
                    frame_count += 1
                    
                    # Store detections in session
                    st.session_state.detections_history.extend(detections)
            
            # Release video writer if used
            if video_writer is not None:
                video_writer.release()
                logger.info(f"Annotated video saved: {output_video_path}")
                st.session_state.output_video_path = str(output_video_path)
            
            # Complete processing
            self.performance.stop()
            progress_placeholder.progress(1.0, text="Processing complete!")
            
            # Generate heatmap
            self.heatmap_gen.generate()
            st.session_state.heatmap_data = self.heatmap_gen.heatmap
            
            # Store analytics
            st.session_state.analytics_data = self.analytics.get_summary_report()
            st.session_state.performance_metrics = {
                'average_fps': self.fps_counter.get_average_fps(),
                'total_frames': frame_count,
                'processing_time': self.performance.get_total_time()
            }
            
            # Store advanced analytics
            st.session_state.advanced_analytics = {
                'zone_analysis': self.zone_analyzer.get_zone_summary(),
                'size_analysis': self.size_analyzer.get_size_statistics(),
                'confidence_trend': self.confidence_analyzer.get_confidence_trend()
            }
            
            st.session_state.processing_complete = True
            
            logger.info(f"Video processing complete. Processed {frame_count} frames")
            components.display_alert(
                f"Processing complete! Analyzed {frame_count} frames with "
                f"{self.analytics.total_detections} detections.",
                "success"
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            components.display_alert(f"Error: {str(e)}", "error")
            raise
    
    def process_webcam(self, settings, max_frames: int = 300):
        """
        Process webcam feed with object detection
        
        Args:
            settings: Settings dictionary from UI
            max_frames: Maximum frames to process
        """
        try:
            # Initialize components
            self.analytics = DetectionAnalytics()
            self.performance = PerformanceMetrics()
            self.fps_counter = FPSCounter()
            
            # Create placeholders
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            performance_placeholder = st.empty()
            
            # Start processing
            self.performance.start()
            
            webcam_index = settings.get('webcam_index', 0)
            
            with VideoProcessor(webcam_index) as processor:
                video_info = processor.get_video_info()
                
                # Initialize heatmap
                self.heatmap_gen = HeatmapGenerator(
                    video_info['width'],
                    video_info['height']
                )
                
                # Initialize advanced analytics
                self.zone_analyzer = ZoneAnalyzer(
                    video_info['width'],
                    video_info['height'],
                    grid_size=(3, 3)
                )
                self.size_analyzer = ObjectSizeAnalyzer()
                self.confidence_analyzer = ConfidenceTemporalAnalyzer()
                
                frame_count = 0
                st.session_state.stop_processing = False
                
                # Stop button (persistent)
                stop_col1, stop_col2 = st.columns([1, 4])
                with stop_col1:
                    if st.button("Stop Detection", type="secondary", key="stop_webcam_btn"):
                        st.session_state.stop_processing = True
                
                while frame_count < max_frames and not st.session_state.stop_processing:
                    ret, frame = processor.read_frame()
                    
                    if not ret:
                        break
                    
                    frame_start_time = time.time()
                    
                    # Detect objects
                    inference_start = time.time()
                    detections = self.detector.detect(frame, track=True)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # Log detection info every 10 frames
                    if frame_count % 10 == 0:
                        logger.info(f"Frame {frame_count}: {len(detections)} detections - {[d.class_name for d in detections]}")
                    
                    self.analytics.add_detections(detections, frame_count)
                    self.heatmap_gen.add_detections(detections)
                    
                    # Advanced analytics
                    for det in detections:
                        self.zone_analyzer.add_detection(det)
                        self.size_analyzer.add_detection(det)
                        self.confidence_analyzer.add_detection(det, frame_count)
                    
                    # Draw detections
                    annotated_frame = self.detector.draw_detections(
                        frame,
                        detections,
                        show_confidence=settings['show_confidence'],
                        show_track_id=settings['show_track_id']
                    )
                    
                    # Update display
                    frame_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live Feed - Frame {frame_count}",
                        use_column_width=True
                    )
                    
                    # Update FPS counter
                    current_fps = self.fps_counter.update()
                    
                    # Update performance metrics
                    with performance_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("FPS", f"{current_fps:.1f}")
                        with col2:
                            st.metric("Inference", f"{inference_time:.0f}ms")
                        with col3:
                            st.metric("Current Frame", len(detections))
                    
                    # Update cumulative metrics (below performance metrics)
                    with metrics_placeholder.container():
                        st.markdown("#### Session Totals")
                        summary = self.detector.get_detection_summary()
                        if summary:
                            total = sum(summary.values())
                            cols = st.columns(min(len(summary) + 1, 4))
                            cols[0].metric("Total", total)
                            for idx, (class_name, count) in enumerate(list(summary.items())[:3]):
                                cols[idx + 1].metric(class_name.capitalize(), count)
                    
                    frame_count += 1
                    
                    time.sleep(0.03)  # ~30 FPS
            
            self.performance.stop()
            
            # Generate results
            self.heatmap_gen.generate()
            st.session_state.heatmap_data = self.heatmap_gen.heatmap
            st.session_state.analytics_data = self.analytics.get_summary_report()
            st.session_state.processing_complete = True
            
            # Store advanced analytics
            st.session_state.advanced_analytics = {
                'zone_analysis': self.zone_analyzer.get_zone_statistics(),
                'size_analysis': self.size_analyzer.get_size_statistics(),
                'confidence_analysis': self.confidence_analyzer.get_confidence_trend()
            }
            
            if st.session_state.stop_processing:
                components.display_alert(
                    f"Detection stopped by user. Processed {frame_count} frames.",
                    "warning"
                )
            else:
                components.display_alert(
                    f"Webcam session complete! Processed {frame_count} frames.",
                    "success"
                )
            
        except Exception as e:
            logger.error(f"Error processing webcam: {e}")
            components.display_alert(f"Error: {str(e)}", "error")
    
    def display_analytics_tab(self):
        """Display analytics dashboard"""
        st.markdown("# Analytics Dashboard")
        st.markdown("---")
        
        if not st.session_state.processing_complete:
            st.info("Complete a detection session to view analytics")
            return
        
        analytics_data = st.session_state.analytics_data
        
        if not analytics_data:
            st.warning("No analytics data available")
            return
        
        # Summary metrics
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", analytics_data.get('total_detections', 0))
        with col2:
            st.metric("Total Frames", analytics_data.get('total_frames', 0))
        with col3:
            unique_tracks = analytics_data.get('tracking_statistics', {}).get('unique_track_ids', 0)
            st.metric("Unique Objects", unique_tracks)
        with col4:
            avg_track = analytics_data.get('tracking_statistics', {}).get('average_track_length', 0)
            st.metric("Avg Track Length", f"{avg_track:.1f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            if 'class_distribution' in analytics_data:
                components.plot_class_distribution(analytics_data['class_distribution'])
        
        with col2:
            # Confidence statistics
            if 'confidence_statistics' in analytics_data:
                components.plot_confidence_distribution(analytics_data['confidence_statistics'])
        
        # Detection frequency
        if self.analytics:
            st.subheader("Detection Frequency Over Time")
            frequency_data = self.analytics.get_detection_frequency(window_size=30)
            components.plot_detection_timeline(frequency_data)
        
        # Performance metrics
        if self.performance:
            st.markdown("---")
            components.display_performance_metrics(self.performance.get_metrics())
    
    def display_heatmap_tab(self):
        """Display heatmap visualization"""
        st.markdown("# Activity Heatmap")
        st.markdown("---")
        
        if not st.session_state.processing_complete:
            st.info("Complete a detection session to view heatmap")
            return
        
        heatmap_data = st.session_state.heatmap_data
        
        if heatmap_data is None or not self.heatmap_gen:
            st.warning("No heatmap data available")
            return
        
        # Display heatmap
        components.display_heatmap(heatmap_data, "Detection Activity Heatmap")
        
        # Heatmap statistics
        st.subheader("Heatmap Statistics")
        stats = self.heatmap_gen.get_statistics()
        components.display_statistics_table(stats)
        
        # Save heatmap
        if st.button("Save Heatmap"):
            try:
                output_path = config.OUTPUT_DIR / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.heatmap_gen.save(str(output_path))
                st.success(f"Heatmap saved to: {output_path}")
            except Exception as e:
                st.error(f"Failed to save heatmap: {e}")
    
    def display_export_tab(self):
        """Display export options"""
        st.markdown("# Export Reports")
        st.markdown("---")
        
        if not st.session_state.processing_complete:
            st.info("Complete a detection session to export reports")
            return
        
        if not st.session_state.detections_history:
            st.warning("No detection data to export")
            return
        
        st.write("Export your detection data and analytics in various formats:")
        
        # Video export section (if available)
        if st.session_state.output_video_path:
            st.markdown("### Annotated Video")
            video_path = Path(st.session_state.output_video_path)
            if video_path.exists():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"Annotated video saved: `{video_path.name}`")
                    st.info(f"📂 Location: `{video_path.parent}`")
                    file_size_mb = video_path.stat().st_size / (1024 * 1024)
                    st.caption(f"File size: {file_size_mb:.2f} MB")
                with col2:
                    # Provide download button
                    with open(video_path, 'rb') as video_file:
                        st.download_button(
                            label="Download Video",
                            data=video_file,
                            file_name=video_path.name,
                            mime="video/mp4",
                            key="download_annotated_video"
                        )
            else:
                st.warning("Video file not found. It may have been moved or deleted.")
            
            st.markdown("---")
        
        # Data export formats selection
        st.markdown("### Data Reports")
        col1, col2, col3 = st.columns(3)
        
        export_json = col1.checkbox("JSON", value=True, key="export_tab_json")
        export_csv = col2.checkbox("CSV", value=True, key="export_tab_csv")
        export_pdf = col3.checkbox("PDF", value=False, key="export_tab_pdf")
        
        # Export button
        if st.button("📤 Generate Reports", type="primary"):
            try:
                with st.spinner("Generating reports..."):
                    formats = []
                    if export_json:
                        formats.append('json')
                    if export_csv:
                        formats.append('csv')
                    if export_pdf:
                        formats.append('pdf')
                    
                    if not formats:
                        st.warning("Please select at least one export format")
                        return
                    
                    # Initialize report generator if not exists
                    if self.report_gen is None:
                        self.report_gen = ReportGenerator()
                    
                    # Save heatmap for PDF
                    heatmap_path = None
                    if export_pdf and self.heatmap_gen:
                        heatmap_path = config.OUTPUT_DIR / "temp_heatmap.png"
                        self.heatmap_gen.save(str(heatmap_path), include_colorbar=False)
                    
                    # Generate reports
                    exported_files = self.report_gen.export_full_report(
                        detections=st.session_state.detections_history,
                        summary_data=st.session_state.analytics_data,
                        heatmap_path=str(heatmap_path) if heatmap_path else None,
                        formats=formats
                    )
                    
                    st.success(f"Reports generated successfully!")
                    
                    # Display download buttons
                    st.markdown("---")
                    components.create_download_buttons(exported_files)
            
            except Exception as e:
                logger.error(f"Export failed: {e}")
                st.error(f"Export failed: {e}")
    
    def run(self):
        """Run the main application"""
        # Display header
        components.display_header()
        
        # Log system info
        log_system_info(logger)
        
        # Sidebar settings
        settings = components.create_settings_sidebar()
        
        # System info
        components.display_system_info()
        
        # Initialize detector
        try:
            self.initialize_detector(
                settings.get('model_name', config.YOLO_MODEL),
                settings['confidence_threshold'],
                settings['tracked_classes']
            )
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = components.create_tabs([
            "Detection", "Analytics", "Heatmap", "Export"
        ])
        
        # Detection Tab
        with tab1:
            st.header("Real-Time Object Detection")
            
            if settings['source_type'] == "Upload Video":
                if settings['video_file'] is not None:
                    # Save uploaded file to temp location on G: drive
                    temp_dir = config.DATA_DIR / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    
                    temp_path = temp_dir / settings['video_file'].name
                    
                    with open(temp_path, 'wb') as f:
                        f.write(settings['video_file'].getbuffer())
                    
                    if st.button("Start Detection", type="primary"):
                        self.process_video(str(temp_path), settings)
                else:
                    st.info("👆 Please upload a video file from the sidebar")
            
            else:  # Webcam
                if st.button("Start Webcam Detection", type="primary"):
                    self.process_webcam(settings, max_frames=600)
        
        # Analytics Tab
        with tab2:
            self.display_analytics_tab()
        
        # Heatmap Tab
        with tab3:
            self.display_heatmap_tab()
        
        # Export Tab
        with tab4:
            self.display_export_tab()


def main():
    """Main entry point"""
    app = SecurityMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
