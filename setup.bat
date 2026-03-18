@echo off
REM Startup script for Spotify Intelligence Copilot (Windows)

echo 🎵 Spotify Intelligence Copilot — Startup Script
echo ==================================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🐍 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/upgrade requirements
echo 📚 Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Copy .env if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env from template...
    copy .env.example .env
    echo    Edit .env with your settings before running!
)

echo.
echo ✅ Setup complete!
echo.
echo 🚀 To start the system (2 terminals):
echo.
echo    Terminal 1 (API Backend):
echo    ^> uvicorn api.main:app --reload
echo.
echo    Terminal 2 (Streamlit Dashboard):
echo    ^> streamlit run app.py
echo.
echo 📍 Access points:
echo    API Docs: http://localhost:8000/docs
echo    Dashboard: http://localhost:8501
echo.
echo 🐳 Or with Docker Compose:
echo    ^> docker-compose up --build
echo.
pause
