@echo off
echo ====================================
echo Boston House Price Predictor
echo Streamlit Web Application
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\streamlit.exe" (
    echo Installing required dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK
)

REM Check if model exists
if not exist "model.pkl" (
    echo Warning: Model file not found!
    echo Training a new model...
    ".venv\Scripts\python.exe" train_model.py
    if errorlevel 1 (
        echo Error: Failed to train model
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit application...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start Streamlit with full path
".venv\Scripts\streamlit.exe" run app.py

echo.
echo Application stopped.
pause
