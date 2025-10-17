import os
import tempfile

def transcribe_with_whisper(audio_path, model_name='base'):
    """Transcribe audio using Whisper if available. Returns text."""
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Whisper is not installed. Install it or use another STT.") from e

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result.get('text', '')

def translate_text_hf(text, src_lang='en', tgt_lang='fr'):
    """Translate using Hugging Face M2M100 model if available."""
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    except Exception as e:
        raise RuntimeError("Transformers or model not available. Install transformers and sentencepiece.") from e

    model_name = 'facebook/m2m100_418M'
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors='pt')
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return out

def text_to_speech_gtts(text, lang='en', out_path=None):
    """Generate TTS using gTTS. Returns path to mp3 file."""
    try:
        from gtts import gTTS
    except Exception as e:
        raise RuntimeError("gTTS not installed.") from e

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
    tts = gTTS(text, lang=lang)
    tts.save(out_path)
    return out_path
