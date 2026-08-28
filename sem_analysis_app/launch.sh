#!/bin/bash
# SEM Particle Analysis App Launcher (macOS/Linux)

echo "🔬 Starting SEM Particle Analysis Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not found. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting Gradio app..."
echo "📱 Open your browser to: http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch the app (module lives one directory up, so run from the repo root)
cd "$(dirname "$0")/.."
python3 -m sem_analysis_app
