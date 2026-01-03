# ☢️ Non-Toxic Me (Pro Voice Edition)

An AI-powered toxicity detection pipeline that uses a fine-tuned **BERT** model to analyze text. This version features **Speech-to-Text (STT)** for voice input and **Text-to-Speech (TTS)** for narrated feedback.

## 🚀 Features
- **Fine-Tuned BERT:** Custom-trained for multi-label toxicity classification (Toxic, Obscene, Threat, Insult, etc.).
- **Voice Interface:** Powered by OpenAI's Whisper (STT) for transcribing microphone input.
- **Audio Feedback:** Uses gTTS to read out the toxicity status automatically.
- **Smart UI:** Detailed label breakdowns only appear when toxicity is detected, keeping the interface clean for positive interactions.

## 🧠 The Model
The core of this project is a **BERT-base** model fine-tuned on toxicity datasets.
- **Base Model:** `bert-base-uncased`
- **Labels:** Toxic, Very Toxic, Obscene, Threat, Insult, Identity Hate.
- **Training:** The fine-tuning process, loss curves, and evaluation metrics are documented in the `/notebooks` directory.


---

## 🛠️ Installation & Setup

This project uses **uv** for ultra-fast, reproducible dependency management.

### 1. Prerequisites
Ensure you have **FFmpeg** installed on your system (required for audio processing):
- **Windows:** `winget install ffmpeg` or `choco install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### 2. Install `uv`
If you don't have `uv` installed:
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

# Linux/macOS/Bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

### 3. Setup Project

Clone the repository and sync the environment:

```bash
git clone [https://github.com/YOUR_USERNAME/NonToxicMe.git](https://github.com/YOUR_USERNAME/NonToxicMe.git)
cd NonToxicMe
uv sync

```

---

## 🏃 Running the App

To start the Gradio interface:

```bash
uv run app.py

```

Once running, the app will be available at `http://127.0.0.1:7860`.

---

## 📂 Project Structure

```text
NonToxicMe/
├── notebooks/          # BERT fine-tuning code & training logs
├── app.py              # Main Gradio application logic
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Deterministic dependency lockfile
└── README.md           # Documentation

```
