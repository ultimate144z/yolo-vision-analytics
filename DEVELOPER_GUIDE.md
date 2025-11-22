# YOLO Vision Analytics – Developer Architecture Guide

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models YOLOv8](https://img.shields.io/badge/Models-YOLOv8-navy)](https://github.com/ultralytics/ultralytics)
[![Framework Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Status Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)](#status)
[![Tests 100% Passing](https://img.shields.io/badge/Tests-100%25%20Passing-success)](tests/)
[![Security Local Execution](https://img.shields.io/badge/Security-Local%20Data%20Only-critical)](#security--privacy)
[![Architecture Modular](https://img.shields.io/badge/Architecture-Modular%20Layered-lightgrey)](#architecture-overview)

Version: 2.0.0  
Last Updated: November 22, 2025

---
> Purpose: Technical reference for maintainers and contributors. Covers architecture, data contracts, extensibility, performance levers, and quality standards. End‑user workflow lives in `PROJECT_GUIDE.md`; optimization rationale in `OPTIMIZATIONS.md`.
---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Module Map](#module-map)
3. [Data Flow](#data-flow)
4. [Core Abstractions](#core-abstractions)
5. [Configuration Layer](#configuration-layer)
6. [Analytics Subsystem](#analytics-subsystem)
7. [Extensibility Points](#extensibility-points)
8. [Performance Considerations](#performance-considerations)
9. [Testing & Quality Assurance](#testing--quality-assurance)
10. [Logging & Observability](#logging--observability)
11. [Deployment & Environment](#deployment--environment)
12. [Security & Privacy](#security--privacy)
13. [Release & Versioning](#release--versioning)
14. [Contribution Standards](#contribution-standards)
15. [Glossary](#glossary)
16. [Status](#status)

---
## Architecture Overview
Layered modular design optimizing isolation and testability:

| Layer | Purpose | Directory |
|-------|---------|-----------|
| Ingestion | Frame acquisition (file/webcam) | `src/detection/video_processor.py` |
| Detection | Model inference | `src/detection/yolo_detector.py` |
| Analytics | Temporal & spatial post‑processing | `src/utils/analytics.py`, `src/utils/advanced_analytics.py` |
| Visualization | Streamlit UI & charts | `src/dashboard/` |
| Reporting | Structured exports | `src/utils/report_generator.py` |
| Observability | Logging, metrics | `src/utils/logger.py` |

Principles: single responsibility, immutable intermediate data, explicit configuration, minimal hidden state, predictable performance.

---
## Module Map
| Module | Role | Key Elements |
|--------|------|--------------|
| `video_processor.py` | Frame generator | context manager, `frames()` |
| `yolo_detector.py` | Inference wrapper | `YOLODetector`, `detect()`, `draw_detections()` |
| `analytics.py` | Base aggregation | `DetectionAnalytics` |
| `advanced_analytics.py` | Extended insights | `ZoneAnalyzer`, `ObjectSizeAnalyzer`, `ConfidenceTemporalAnalyzer` |
| `heatmap_generator.py` | Density heatmap | `generate()` |
| `report_generator.py` | Export pipeline | `export_json/csv/pdf` |
| `components.py` | UI composition | sidebar + charts |
| `app.py` | Orchestration | detector caching, loops |
| `logger.py` | Logging setup | `get_logger()` |

Deprecated: ETA estimation (removed due to instability with frame skip). Do not reintroduce without adaptive predictive model.

---
## Data Flow
```
Capture → Frame Iterator → YOLO Inference → Detection Records
      → Analytics Accumulators → UI Rendering → Exports (Reports/Video/Heatmap)
```

Structures:
- Detection: `{frame:int,class:str,confidence:float,bbox:(x1,y1,x2,y2)}`
- Summary: `{total_detections:int,class_distribution:dict,temporal:list}`
- Zones: `{(gx,gy):count}`
- Sizes: `[{area:int,width:int,height:int}]`
- Confidence trend: `{frame_number:[float]}`

---
## Core Abstractions
### YOLODetector
Encapsulates model loading and inference. Returns normalized detection objects filtered by confidence and class list.

### VideoProcessor
Manages capture lifecycle; yields `(frame, frame_num, timestamp)`. Frame skipping handled by caller to keep processor pure.

### Analytics Classes
Stateless interfaces over cumulative internal state (maps/lists). All output functions return serializable dicts for downstream export.

---
## Configuration Layer
File: `config/config.py`
Defines thresholds, tracked classes, frame sizing. New configuration entries must include docstrings and reference updates in documentation. Avoid adding transient runtime flags; prefer explicit structured settings.

---
## Analytics Subsystem
| Analyzer | Purpose | Outputs |
|----------|---------|---------|
| ZoneAnalyzer | Spatial hotspot density | top zones, per‑class counts |
| ObjectSizeAnalyzer | Relative size grouping | bucket counts, mean area |
| ConfidenceTemporalAnalyzer | Confidence stability | mean, std, low frames |

Buckets & thresholds internal; promote to config only if operational tuning emerges.

---
## Extensibility Points
1. New analytics module → add class under `src/utils`, invoke post‑detection, expose UI toggle.
2. Additional export format → extend `report_generator.py`; integrate with export tab.
3. Alternate vision model → adapter preserving detection record schema.
4. Tracking integration → tracker wrapper adding `track_id` field.

Contracts: maintain detection fields; avoid mutable globals; include tests for new transformations.

---
## Performance Considerations
Levers: model tier, frame skip, resolution (future), device (GPU/CPU). Caching limited to detector instance via `@st.cache_resource`. FPS reported as moving average; inference latency surfaced; ETA intentionally absent.

---
## Testing & Quality Assurance
Current tests: detector init, model load, blank frame behavior, summary generation, config paths.
Recommended: analytics correctness, export artifact integrity, performance baselines (non‑blocking). Tests must be deterministic; mock external downloads if added.

Coverage badge placeholder—formal measurement scheduled next milestone.

---
## Logging & Observability
Structured logging to `data/logs/`. Avoid per‑frame verbose output in long sessions (consider sampling). Future enhancements: Prometheus metrics adapter, anomaly hooks.

---
## Deployment & Environment
Local:
```bash
streamlit run src/dashboard/app.py
```
Remote/headless:
```bash
streamlit run src/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```
GPU optional; auto fallback to CPU. Ensure G: drive capacity for large batches.

---
## Security & Privacy
All processing local; no outbound data beyond initial model weight fetch. No PII extraction. Potential future enhancements: checksum validation, sandboxed inference process.

---
## Release & Versioning
Semantic versioning (MAJOR/MINOR/PATCH). Release checklist: tests pass, docs updated, no large binaries committed, optimization benchmarks validated.

---
## Contribution Standards
Style: type hints, explicit error handling, no silent excepts. Commits follow conventional prefixes. PRs supply rationale + risk + test evidence. Review focuses on data contract stability and performance impact.

---
## Glossary
| Term | Definition |
|------|------------|
| Detection | Model output with class, confidence, bbox |
| Frame Sampling | Reducing processed frames by skip factor |
| Bounding Box | Pixel rectangle encapsulating object |
| Zone | Grid cell in spatial partition |
| Confidence Trend | Temporal confidence statistics |
| Throughput | Effective frames processed per second |
| Aggregation | Combining per‑frame detections into summaries |

---
## Status
Production ready. Next roadmap items: coverage integration, optional tracking adapter, metrics endpoint.

---
This document evolves through PRs; structural changes require linked issue detailing motivation and impact.
