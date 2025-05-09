@echo off
echo Starting Handwritten Text Recognition System...

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Start Ollama server in a new window
start cmd /k "echo Starting Ollama server... && ollama serve"

:: Wait for Ollama server to start
timeout /t 5 /nobreak

:: Start Streamlit app in the same window (after venv activation)
echo Starting Streamlit app...
streamlit run ocr.py

:: Keep the window open
pause

echo System is starting up...
echo Please wait for both windows to fully load...
echo.
echo 1. Ollama server should be running on http://localhost:11434
echo 2. Streamlit app will open in your default browser
echo.
echo Press any key to close this window...
pause > nul 