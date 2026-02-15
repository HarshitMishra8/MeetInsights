# MeetInsights (Meeting Summarizer)

Lightweight meeting transcription and summarization UI using Whisper (local) and an Ollama LLM.

Important: the original MIT `LICENSE` file has been removed at your request — this repository now has no license file.

## What this repository contains
- `main.py` — Gradio app that converts audio -> transcript (via `whisper.cpp`) and summarizes via an Ollama model.
- `whisper.cpp/` — local whisper.cpp sources and helper scripts (models are excluded from git).
- `requirements.txt` — Python dependencies.

## Quick setup (macOS, Apple Silicon)

1. Install system packages (Homebrew):

```bash
brew install pkg-config freetype libpng zlib jpeg ffmpeg
brew install python@3.11
```

2. Create and activate a Python 3.11 venv (recommended):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-3.11
source .venv-3.11/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Whisper model binaries are large and are gitignored. Download models using the helper script included in `whisper.cpp` or obtain ggml models manually:

```bash
# from repo root
./whisper.cpp/download-ggml-model.sh   # or run the provided script in whisper.cpp/models
```

4. Build `whisper.cpp` if `whisper.cpp/main` is not present:

```bash
cd whisper.cpp
make
cd ..
# optionally symlink binary the app expects
ln -s ./whisper.cpp/build/bin/main ./whisper.cpp/main || true
```

5. Configure Ollama or your LLM endpoint:
- By default `main.py` points to `http://localhost:11434` (set `OLLAMA_SERVER_URL` in the environment or edit `main.py` to change).
- Ensure your Ollama server is running and has models available.

6. Run the app:

```bash
source .venv-3.11/bin/activate
python main.py
```

The Gradio UI will start and show a link in the terminal.

## Notes
- Large files (models, virtualenvs) are intentionally excluded via `.gitignore`.
- If you want to include large model files in the repo, consider using Git LFS or a separate storage mechanism — I can help set that up.

## Troubleshooting
- If `pip install -r requirements.txt` fails building native packages, ensure `pkg-config`, `freetype`, `libpng`, and a compatible Python version (3.11) are installed.
- If `ffmpeg` is missing, install via Homebrew: `brew install ffmpeg`.

---
If you'd like, I can also add a script to download or verify required models automatically. Tell me which option you prefer.
