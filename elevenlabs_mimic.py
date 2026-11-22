#!/usr/bin/env python3
"""
ElevenLabs Mimic Server

Mimics the ElevenLabs API endpoints using local Whisper (STT) and XTTS (TTS).
Allows OpenClaw or other clients to use local speech services instead of
ElevenLabs cloud.

Endpoints:
  POST /v1/text-to-speech/{voice_id}  - Text-to-Speech (local XTTS)
  GET  /v1/voices                      - List available voices
  POST /v1/speech-to-text/             - Speech-to-Text (local Whisper)

Usage:
    PYTHONPATH=. python elevenlabs_mimic.py

Environment variables:
    WHISPER_MODEL    - Whisper model size (tiny, base, small, medium, large-v3-turbo). Default: base
    WHISPER_DEVICE   - Device (auto, cpu, cuda, mps). Default: auto
    LANGUAGE         - Language code or None for auto-detect. Default: None
    PORT             - Server port. Default: 8767
    HOST             - Server host. Default: 0.0.0.0
    VOICE_SAMPLE     - Path to voice sample WAV for XTTS cloning. Default: None
"""

import os
import io
import json
import time
import numpy as np
from typing import Optional

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Logging
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8767"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
LANGUAGE = os.getenv("LANGUAGE", None)
VOICE_SAMPLE = os.getenv("VOICE_SAMPLE", None)

app = FastAPI(title="ElevenLabs Mimic", version="0.1.0")

# Global instances
stt_model = None
tts_model = None
tts_backend = "mock"


def load_models():
    """Load Whisper STT and XTTS TTS models."""
    global stt_model, tts_model, tts_backend

    # Load Whisper STT
    logger.info(f"Loading Whisper STT model '{WHISPER_MODEL}'...")
    try:
        from faster_whisper import WhisperModel
        import torch

        if WHISPER_DEVICE == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        elif WHISPER_DEVICE == "cuda":
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        stt_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        logger.info(f"✅ Whisper STT loaded on {device}")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper: {e}")
        stt_model = None

    # Load TTS (try XTTS first, then mock)
    logger.info("Loading TTS model...")
    try:
        from TTS.api import TTS
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts_backend = "xtts"
        logger.info("✅ XTTS TTS loaded")
    except Exception as e:
        logger.warning(f"XTTS not available: {e}")
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            tts_model = CBModel.from_pretrained(device="cpu")
            tts_backend = "chatterbox"
            logger.info("✅ Chatterbox TTS loaded")
        except Exception as e2:
            logger.warning(f"Chatterbox not available: {e2}")
            tts_backend = "mock"
            logger.warning("⚠️ Using mock TTS (silence)")


@app.on_event("startup")
async def startup():
    load_models()


# ===================
# ElevenLabs-compatible endpoints
# ===================

@app.get("/v1/voices")
async def list_voices():
    """List available voices (ElevenLabs-compatible)."""
    voices = [
        {
            "voice_id": "default",
            "name": "Default Voice",
            "category": "premade",
            "description": "Default local voice",
        }
    ]
    if VOICE_SAMPLE:
        voices.append({
            "voice_id": "cloned",
            "name": "Cloned Voice",
            "category": "cloned",
            "description": f"Voice cloned from {VOICE_SAMPLE}",
        })
    return {"voices": voices}


@app.post("/v1/text-to-speech/{voice_id}")
async def text_to_speech(voice_id: str, request: dict = None):
    """Text-to-Speech endpoint (ElevenLabs-compatible).

    Accepts JSON body with 'text' field.
    Returns audio as WAV/PCM stream.
    """
    if request is None:
        request = {}

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    logger.info(f"TTS request: voice={voice_id}, text='{text[:50]}...'")

    if tts_backend == "xtts":
        try:
            if VOICE_SAMPLE and voice_id == "cloned":
                wav = tts_model.tts(text=text, speaker_wav=VOICE_SAMPLE, language="en")
            else:
                # Use a default speaker from XTTS's built-in speakers
                wav = tts_model.tts(text=text, speaker="Claribel Dervla", language="en")
            audio_array = np.array(wav, dtype=np.float32)

            # Convert to 16-bit PCM WAV
            import struct
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Build WAV file in memory
            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)  # XTTS sample rate
                wf.writeframes(audio_int16.tobytes())
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="audio/wav")
        except Exception as e:
            logger.error(f"XTTS error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    elif tts_backend == "chatterbox":
        try:
            if VOICE_SAMPLE:
                audio = tts_model.generate(text, audio_prompt=VOICE_SAMPLE)
            else:
                audio = tts_model.generate(text)
            audio_array = audio.cpu().numpy().astype(np.float32)
            audio_int16 = (audio_array * 32767).astype(np.int16)

            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_int16.tobytes())
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="audio/wav")
        except Exception as e:
            logger.error(f"Chatterbox error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    else:
        # Mock: return silence
        buffer = io.BytesIO()
        import wave
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 24000 * 2)  # 1 sec silence
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    """Speech-to-Text endpoint (ElevenLabs-compatible).

    Accepts audio file upload.
    Returns JSON with transcript.
    """
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    # Read audio file
    audio_bytes = await file.read()

    # Convert to numpy array
    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)  # Convert to mono
        audio_float32 = audio_array.astype(np.float32)
    except Exception as e:
        logger.error(f"Audio read error: {e}")
        # Try raw PCM
        audio_float32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    logger.info(f"STT request: {len(audio_float32)} samples at {sample_rate if 'sample_rate' in dir() else '?'}Hz")

    # Transcribe
    try:
        segments, info = stt_model.transcribe(
            audio_float32,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        transcript = ""

    return JSONResponse({
        "text": transcript,
        "language_code": info.language if 'info' in dir() else (LANGUAGE or "en"),
        "language_probability": getattr(info, "language_probability", 1.0) if 'info' in dir() else 1.0,
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "stt": "whisper" if stt_model else "none",
        "tts": tts_backend,
        "stt_model": WHISPER_MODEL,
    }


def main():
    logger.info(f"Starting ElevenLabs Mimic server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
