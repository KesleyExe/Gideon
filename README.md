# 🤖 Local Chatbot — Free & Private

A complete end-to-end chatbot that runs **entirely on your computer**. No API keys, no subscriptions, no data ever leaves your machine.

Built with [uv](https://docs.astral.sh/uv/) for lightning-fast Python dependency management.

## ✨ Features

- 🆓 **100% Free** — Uses Ollama to run open-source LLMs locally
- 🔒 **Private** — All conversations stay on your device
- ⚡ **Streaming** — Real-time token-by-token responses
- 🎨 **Beautiful UI** — Dark mode, responsive, mobile-friendly
- 🧠 **Multiple Models** — Switch between any Ollama model
- 💾 **Conversation Memory** — Maintains context across messages

## 🚀 Quick Start

### 1. Install uv (if you haven't)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS/Linux
# Or: brew install uv
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download from [ollama.com/download](https://ollama.com/download)

### 3. Download a Model

```bash
ollama pull llama3.2        # Fast, good quality (2GB)
ollama pull mistral         # Alternative option (4GB)
ollama pull llama3          # Best quality, larger (4.7GB)
```

### 4. Start Ollama Server

```bash
ollama serve
```

Leave this terminal running.

### 5. Install & Run the Chatbot

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
uv run uvicorn main:app --reload
```

**Windows:**
```cmd
setup.bat
uv run uvicorn main:app --reload
```

Or manually:
```bash
uv venv
uv pip install -r requirements.txt
uv run uvicorn main:app --reload
```

### 6. Open in Browser

Go to: **http://localhost:8000**

---

## 📁 Project Structure

```
chatbot-project/
├── main.py              # FastAPI backend
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Chat UI (single file, no build step)
├── setup.sh             # macOS/Linux setup script
└── setup.bat            # Windows setup script
```

## 🔧 Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- 4GB+ RAM (8GB+ recommended)
- ~2-5GB disk space for the model

## 🛠️ Customization

### Change the default model
Edit `main.py` and change:
```python
DEFAULT_MODEL = "llama3.2"
```

### Add system prompt
In `main.py`, modify the `messages` list to include a system message:
```python
messages = [{"role": "system", "content": "You are a helpful coding assistant."}]
```

### Change the port
```bash
uv run uvicorn main:app --reload --port 3000
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not found" | Make sure `ollama serve` is running in another terminal |
| "Model not found" | Run `ollama pull llama3.2` to download the model |
| Slow responses | Use a smaller model like `llama3.2` or enable GPU acceleration |
| Port already in use | Change port: `uv run uvicorn main:app --port 8001` |

## 📜 License

MIT — Use it however you want!
