#!/bin/bash
# Startup script for Spotify Intelligence Copilot

set -e

echo "🎵 Spotify Intelligence Copilot — Startup Script"
echo "=================================================="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    VENV_ACTIVATE="source .venv/bin/activate"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    VENV_ACTIVATE="source .venv/bin/activate"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    VENV_ACTIVATE=".venv\\Scripts\\activate"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
eval $VENV_ACTIVATE

# Install/upgrade requirements
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "   Edit .env with your settings before running!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the system (2 terminals):"
echo ""
echo "   Terminal 1 (API Backend):"
echo "   $ uvicorn api.main:app --reload"
echo ""
echo "   Terminal 2 (Streamlit Dashboard):"
echo "   $ streamlit run app.py"
echo ""
echo "📍 Access points:"
echo "   API Docs: http://localhost:8000/docs"
echo "   Dashboard: http://localhost:8501"
echo ""
echo "🐳 Or with Docker Compose:"
echo "   $ docker-compose up --build"
echo ""
