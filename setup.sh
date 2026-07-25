#!/bin/bash
set -e

echo "🚀 Setting up your free local chatbot with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or: brew install uv"
    exit 1
fi

# Check if Python is available through uv
if ! uv python find &> /dev/null; then
    echo "📦 Installing Python via uv..."
    uv python install 3.11
fi

# Create virtual environment and install dependencies
echo "📦 Creating venv & installing deps..."
uv venv
uv pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "⚠️  Ollama is not installed. You need to install it first:"
    echo "   macOS:   brew install ollama"
    echo "   Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "   Windows: https://ollama.com/download"
    echo ""
    echo "After installing Ollama, run: ollama pull llama3.2"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Make sure Ollama is running: ollama serve"
echo "  2. Download a model (if not already): ollama pull llama3.2"
echo "  3. Start the chatbot: uv run uvicorn main:app --reload"
echo "  4. Open http://localhost:8000 in your browser"
echo ""
