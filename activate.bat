@echo off
echo ========================================
echo YOLOv8 Security Monitor
echo ========================================
echo.

cd /d G:\yolov8_security_monitor
call venv\Scripts\activate.bat

echo Environment activated!
echo.
echo Quick commands:
echo   streamlit run src/dashboard/app.py  - Start dashboard
echo   pytest tests/                       - Run tests
echo   deactivate                          - Exit environment
echo.
cmd /k
