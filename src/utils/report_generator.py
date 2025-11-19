"""
Report Generation Module
Exports detection data in multiple formats (JSON, CSV, PDF)
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, TYPE_CHECKING
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    import warnings
    warnings.warn("ReportLab not available. PDF export will be disabled.")

import config.config as config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.detection.yolo_detector import Detection

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate reports from detection data in various formats
    All reports saved to G: drive
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Output directory for reports (on G: drive)
        """
        self.output_dir = output_dir or config.EXPORTS_DIR
        
        # Ensure on G: drive
        if not str(self.output_dir).startswith('G:') and not str(self.output_dir).startswith('g:'):
            self.output_dir = config.EXPORTS_DIR
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Report generator initialized. Output: {self.output_dir}")
    
    def _generate_filename(self, prefix: str, extension: str) -> Path:
        """
        Generate timestamped filename
        
        Args:
            prefix: Filename prefix
            extension: File extension
        
        Returns:
            Full file path
        """
        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        filename = f"{prefix}_{timestamp}.{extension}"
        return self.output_dir / filename
    
    def export_json(self, data: dict, filename: str = None) -> Path:
        """
        Export data as JSON
        
        Args:
            data: Data to export
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            output_path = self._generate_filename("detection_report", "json")
        else:
            output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"JSON report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_csv(self, detections: List['Detection'], filename: str = None) -> Path:
        """
        Export detections as CSV
        
        Args:
            detections: List of Detection objects
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            output_path = self._generate_filename("detections", "csv")
        else:
            output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Frame Number', 'Timestamp', 'Class Name', 'Confidence',
                    'BBox X1', 'BBox Y1', 'BBox X2', 'BBox Y2', 'Track ID'
                ])
                
                # Write data
                for det in detections:
                    writer.writerow([
                        det.frame_number,
                        det.timestamp,
                        det.class_name,
                        f"{det.confidence:.4f}",
                        det.bbox[0],
                        det.bbox[1],
                        det.bbox[2],
                        det.bbox[3],
                        det.track_id
                    ])
            
            logger.info(f"CSV report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def export_dataframe_csv(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Export pandas DataFrame as CSV
        
        Args:
            df: DataFrame to export
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            output_path = self._generate_filename("analytics", "csv")
        else:
            output_path = self.output_dir / filename
        
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"DataFrame CSV saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export DataFrame CSV: {e}")
            raise
    
    def export_pdf(self, 
                   summary_data: dict,
                   detections: List['Detection'] = None,
                   heatmap_path: str = None,
                   filename: str = None) -> Path:
        """
        Export comprehensive PDF report
        
        Args:
            summary_data: Summary statistics dictionary
            detections: Optional list of detections for details
            heatmap_path: Optional path to heatmap image
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available. Cannot export PDF.")
            raise ImportError("ReportLab is required for PDF export")
        
        if filename is None:
            output_path = self._generate_filename("detection_report", "pdf")
        else:
            output_path = self.output_dir / filename
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("YOLOv8 Security Monitor Report", title_style))
            story.append(Spacer(1, 0.3 * inch))
            
            # Metadata
            metadata = [
                ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Model:', config.YOLO_MODEL],
                ['Confidence Threshold:', f"{config.CONFIDENCE_THRESHOLD:.2f}"]
            ]
            
            meta_table = Table(metadata, colWidths=[2*inch, 4*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.5 * inch))
            
            # Summary Section
            story.append(Paragraph("Detection Summary", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))
            
            summary_items = []
            if 'total_detections' in summary_data:
                summary_items.append(['Total Detections:', str(summary_data['total_detections'])])
            if 'total_frames' in summary_data:
                summary_items.append(['Total Frames:', str(summary_data['total_frames'])])
            if 'class_distribution' in summary_data:
                story.append(Paragraph("Class Distribution:", styles['Heading3']))
                for class_name, count in summary_data['class_distribution'].items():
                    summary_items.append([f"  {class_name}:", str(count)])
            
            if summary_items:
                summary_table = Table(summary_items, colWidths=[3*inch, 3*inch])
                summary_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(summary_table)
            
            story.append(Spacer(1, 0.3 * inch))
            
            # Tracking Statistics
            if 'tracking_statistics' in summary_data:
                story.append(Paragraph("Tracking Statistics", styles['Heading2']))
                story.append(Spacer(1, 0.2 * inch))
                
                track_stats = summary_data['tracking_statistics']
                track_items = [
                    ['Unique Tracked Objects:', str(track_stats.get('unique_track_ids', 0))],
                    ['Average Track Length:', f"{track_stats.get('average_track_length', 0):.2f} frames"]
                ]
                
                track_table = Table(track_items, colWidths=[3*inch, 3*inch])
                track_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(track_table)
                story.append(Spacer(1, 0.3 * inch))
            
            # Heatmap
            if heatmap_path and Path(heatmap_path).exists():
                story.append(PageBreak())
                story.append(Paragraph("Activity Heatmap", styles['Heading2']))
                story.append(Spacer(1, 0.2 * inch))
                
                img = Image(str(heatmap_path), width=6*inch, height=4.5*inch)
                story.append(img)
            
            # Detailed Detections (first 100)
            if detections and len(detections) > 0:
                story.append(PageBreak())
                story.append(Paragraph("Detection Details (Sample)", styles['Heading2']))
                story.append(Spacer(1, 0.2 * inch))
                
                det_data = [['Frame', 'Class', 'Confidence', 'Track ID']]
                for det in detections[:100]:  # Limit to first 100
                    det_data.append([
                        str(det.frame_number) if det.frame_number else 'N/A',
                        det.class_name,
                        f"{det.confidence:.3f}",
                        str(det.track_id) if det.track_id else 'N/A'
                    ])
                
                det_table = Table(det_data, colWidths=[1*inch, 2*inch, 1.5*inch, 1.5*inch])
                det_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(det_table)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            raise
    
    def export_full_report(self,
                          detections: List['Detection'],
                          summary_data: dict,
                          heatmap_path: str = None,
                          formats: List[str] = None) -> Dict[str, Path]:
        """
        Export complete report in multiple formats
        
        Args:
            detections: List of Detection objects
            summary_data: Summary statistics dictionary
            heatmap_path: Optional path to heatmap image
            formats: List of formats to export ('json', 'csv', 'pdf')
        
        Returns:
            Dictionary mapping format names to file paths
        """
        if formats is None:
            formats = config.EXPORT_FORMATS
        
        exported_files = {}
        
        # JSON export
        if 'json' in formats:
            try:
                # Prepare JSON-serializable data
                json_data = {
                    'metadata': {
                        'generated': datetime.now().isoformat(),
                        'model': config.YOLO_MODEL,
                        'confidence_threshold': config.CONFIDENCE_THRESHOLD
                    },
                    'summary': summary_data,
                    'detections': [
                        {
                            'frame_number': det.frame_number,
                            'timestamp': det.timestamp,
                            'class_name': det.class_name,
                            'confidence': det.confidence,
                            'bbox': det.bbox,
                            'track_id': det.track_id
                        }
                        for det in detections
                    ]
                }
                
                path = self.export_json(json_data)
                exported_files['json'] = path
                
            except Exception as e:
                logger.error(f"JSON export failed: {e}")
        
        # CSV export
        if 'csv' in formats:
            try:
                path = self.export_csv(detections)
                exported_files['csv'] = path
                
            except Exception as e:
                logger.error(f"CSV export failed: {e}")
        
        # PDF export
        if 'pdf' in formats and REPORTLAB_AVAILABLE:
            try:
                path = self.export_pdf(summary_data, detections, heatmap_path)
                exported_files['pdf'] = path
                
            except Exception as e:
                logger.error(f"PDF export failed: {e}")
        
        logger.info(f"Exported report in {len(exported_files)} format(s)")
        return exported_files
    
    def create_batch_summary(self, batch_results: List[dict]) -> Path:
        """
        Create summary report for batch processing
        
        Args:
            batch_results: List of result dictionaries from batch processing
        
        Returns:
            Path to summary file
        """
        output_path = self._generate_filename("batch_summary", "json")
        
        summary = {
            'generated': datetime.now().isoformat(),
            'total_videos': len(batch_results),
            'results': batch_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Batch summary saved: {output_path}")
        return output_path
