@echo off
echo Starting Handwritten Text Recognition System...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama is not installed! Please install Ollama first.
    pause
    exit /b 1
)

:: Check if virtual environment exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Start Ollama in a new window
start "Ollama Service" cmd /k "ollama serve"

:: Wait for Ollama to start
echo Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

:: Pull the required model if not already present
echo Checking for required model...
ollama list | findstr "llama3.2-vision" >nul
if errorlevel 1 (
    echo Pulling llama3.2-vision model...
    ollama pull llama3.2-vision
)

:: Start Streamlit in a new window
echo Starting Streamlit application...
start "Streamlit App" cmd /k "streamlit run ocr.py"

echo System started successfully!
echo.
echo Please keep both windows open while using the application.
echo The application will be available at http://localhost:8501
echo.
echo Press any key to exit this window...
pause >nul 