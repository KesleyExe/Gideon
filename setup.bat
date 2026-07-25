@echo off
echo Setting up your free local chatbot with uv...

uv --version >nul 2>&1
if errorlevel 1 (
    echo uv is not installed. Install it first:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

echo Creating venv and installing dependencies...
uv venv
uv pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. Install Ollama from https://ollama.com/download
echo   2. Open a new terminal and run: ollama serve
echo   3. Download a model: ollama pull llama3.2
echo   4. Start the chatbot: uv run uvicorn main:app --reload
echo   5. Open http://localhost:8000 in your browser
echo.
pause
