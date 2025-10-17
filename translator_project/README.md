# Speech-to-Multilingual Translator (Streamlit)

**What this is**
A simple prototype app that:
1. Accepts an uploaded audio file (wav/mp3).
2. Transcribes speech using Whisper (if installed).
3. Translates the transcription into one or more target languages using Hugging Face transformers (or a cloud API).
4. Optionally generates spoken audio (MP3) using gTTS.

**Files**
- app.py — Streamlit frontend.
- translator.py — Core functions (transcribe, translate, tts).
- requirements.txt — Packages to install.
- LICENSE — MIT license.

**How to run (example)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Notes**
- This project is a starting point. Whisper and some HF models are large and require GPU for fast processing.
- If you don't want to install large models locally, swap the translation step for a cloud API (DeepL, Google Translate, or OpenAI).

**Support**
If you want, I can:
- Replace Streamlit with Flask.
- Add an example Dockerfile.
- Add optional batch translation or CLI mode.
