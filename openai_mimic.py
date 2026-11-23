#!/usr/bin/env python3
"""
OpenAI Mimic Server

Mimics the OpenAI API endpoints using local services:
- STT: Local Whisper
- TTS: Local XTTS
- Chat: Local LLM (port 1919, OpenAI-compatible)
- Models: Local model listing

This allows OpenClaw to use openclaw-voice as an OpenAI-compatible backend
without any cloud API calls.

Endpoints (OpenAI-compatible):
  POST /v1/chat/completions        - Chat completions (local LLM)
  POST /v1/responses                - Responses API (local LLM)
  POST /v1/audio/speech             - Text-to-Speech (local XTTS)
  POST /v1/audio/transcriptions     - Speech-to-Text (local Whisper)
  POST /v1/audio/translations       - Audio translation (local Whisper)
  GET  /v1/models                   - List models
  POST /v1/embeddings               - Create embeddings (local LLM)
  GET  /v1/realtime                 - Realtime WebSocket (proxy to openclaw-voice)
  POST /v1/realtime/calls           - Realtime calls
  GET  /v1/live                     - Live sessions
  GET  /v1/live/sessions            - List live sessions
  GET  /v1/live/sessions/{id}       - Live session details
  GET  /v1/organization/costs       - Organization costs (stub)
  GET  /v1/organization/usage/completions - Usage (stub)
  POST /auth                        - Authentication (no-op, always succeeds)

Usage:
    PYTHONPATH=. python openai_mimic.py

Environment variables:
    LLM_URL          - Local LLM URL (default: http://localhost:1919/v1)
    LLM_MODEL        - Model name (default: auto-detect from /v1/models)
    WHISPER_MODEL    - Whisper model size (default: base)
    WHISPER_DEVICE   - Device (default: auto)
    PORT             - Server port (default: 8768)
    HOST             - Server host (default: 0.0.0.0)
    VOICE_SAMPLE     - Path to voice sample WAV for XTTS cloning
"""

import os
import io
import json
import time
import uuid
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket
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
PORT = int(os.getenv("PORT", "8768"))
LLM_URL = os.getenv("LLM_URL", "http://localhost:1919/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
LANGUAGE = os.getenv("LANGUAGE", None)
VOICE_SAMPLE = os.getenv("VOICE_SAMPLE", None)


# Global instances
stt_model = None
tts_model = None
tts_backend = "mock"
llm_model = None
http_client = None


def load_models():
    """Load STT, TTS, and detect LLM model."""
    global stt_model, tts_model, tts_backend, llm_model, http_client

    import httpx
    http_client = httpx.Client(base_url=LLM_URL, timeout=120.0)

    # Detect LLM model
    try:
        resp = http_client.get("/models")
        models = resp.json().get("data", [])
        if models:
            llm_model = models[0]["id"]
            logger.info(f"✅ LLM model detected: {llm_model}")
        else:
            llm_model = "default"
            logger.warning("No LLM models found, using 'default'")
    except Exception as e:
        logger.error(f"❌ Failed to connect to LLM at {LLM_URL}: {e}")
        llm_model = "default"

    # Load Whisper STT
    logger.info(f"Loading Whisper STT model '{WHISPER_MODEL}'...")
    try:
        from faster_whisper import WhisperModel
        import torch

        if WHISPER_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
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

    # Load TTS
    logger.info("Loading TTS model...")
    try:
        from TTS.api import TTS
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts_backend = "xtts"
        logger.info("✅ XTTS TTS loaded")
    except Exception as e:
        logger.warning(f"XTTS not available: {e}")
        tts_backend = "mock"
        logger.warning("⚠️ Using mock TTS (silence)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="OpenAI Mimic", version="0.1.0", lifespan=lifespan)


# ===================
# Health & Info
# ===================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt": "whisper" if stt_model else "none",
        "tts": tts_backend,
        "llm": llm_model,
        "llm_url": LLM_URL,
    }


# ===================
# Models
# ===================

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = [
        {"id": llm_model or "default", "object": "model", "created": int(time.time()), "owned_by": "local"},
        {"id": "whisper-1", "object": "model", "created": int(time.time()), "owned_by": "local"},
        {"id": "tts-1", "object": "model", "created": int(time.time()), "owned_by": "local"},
    ]
    return {"object": "list", "data": models}


# ===================
# Chat Completions
# ===================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completions (OpenAI-compatible). Proxies to local LLM."""
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", llm_model)
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 500)
    temperature = body.get("temperature", 0.7)

    if stream:
        return StreamingResponse(
            stream_chat(messages, model, max_tokens, temperature),
            media_type="text/event-stream",
        )

    # Non-streaming
    try:
        resp = http_client.post("/chat/completions", json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        })
        return JSONResponse(resp.json())
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Error: {e}"}, "finish_reason": "stop"}],
        }, status_code=500)


async def stream_chat(messages, model, max_tokens, temperature):
    """Stream chat completions as SSE with proper framing.

    SSE events must be separated by a blank line (\\n\\n). The upstream
    LLM may send lines with single \\n separators, so we normalize here.
    """
    import httpx
    async with httpx.AsyncClient(base_url=LLM_URL, timeout=120.0) as client:
        async with client.stream("POST", "/chat/completions", json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    # Each SSE data line followed by blank line delimiter
                    yield f"{line}\n\n"
                # Skip empty lines from upstream (we add our own delimiters)


# ===================
# Responses API
# ===================

@app.post("/v1/responses")
async def responses(request: Request):
    """Responses API (OpenAI-compatible). Converts to chat completions."""
    body = await request.json()
    input_text = body.get("input", "")
    model = body.get("model", llm_model)

    # Convert to chat format
    messages = [{"role": "user", "content": input_text}]

    try:
        resp = http_client.post("/chat/completions", json={
            "model": model,
            "messages": messages,
            "max_tokens": body.get("max_output_tokens", 500),
            "temperature": body.get("temperature", 0.7),
            "stream": False,
        })
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        return JSONResponse({
            "id": f"resp-{uuid.uuid4().hex[:12]}",
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}],
            "status": "completed",
        })
    except Exception as e:
        logger.error(f"Responses API error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ===================
# Audio: TTS
# ===================

@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """Text-to-Speech (OpenAI-compatible)."""
    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", "alloy")
    model = body.get("model", "tts-1")
    response_format = body.get("response_format", "mp3")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'input' field")

    logger.info(f"TTS: voice={voice}, text='{text[:50]}...'")

    if tts_backend == "xtts":
        try:
            if VOICE_SAMPLE:
                wav = tts_model.tts(text=text, speaker_wav=VOICE_SAMPLE, language="en")
            else:
                wav = tts_model.tts(text=text, speaker="Claribel Dervla", language="en")
            audio_array = np.array(wav, dtype=np.float32)
            audio_int16 = (audio_array * 32767).astype(np.int16)

            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio_int16.tobytes())
            buffer.seek(0)

            media_type = "audio/wav" if response_format == "wav" else "audio/mpeg"
            return StreamingResponse(buffer, media_type=media_type)
        except Exception as e:
            logger.error(f"XTTS error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Mock: return silence
        buffer = io.BytesIO()
        import wave
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 24000 * 2)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")


# ===================
# Audio: STT
# ===================

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...), model: str = "whisper-1", language: Optional[str] = None):
    """Speech-to-Text (OpenAI-compatible)."""
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    audio_bytes = await file.read()

    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_float32 = audio_array.astype(np.float32)
    except Exception:
        audio_float32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        segments, info = stt_model.transcribe(
            audio_float32,
            language=language or LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        transcript = ""

    return JSONResponse({"text": transcript})


@app.post("/v1/audio/translations")
async def audio_translations(file: UploadFile = File(...), model: str = "whisper-1"):
    """Audio translation (OpenAI-compatible). Translates to English."""
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    audio_bytes = await file.read()

    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_float32 = audio_array.astype(np.float32)
    except Exception:
        audio_float32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        segments, info = stt_model.transcribe(
            audio_float32,
            language=None,  # Auto-detect
            task="translate",  # Translate to English
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        transcript = ""

    return JSONResponse({"text": transcript})


# ===================
# Embeddings
# ===================

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Create embeddings (OpenAI-compatible). Proxies to local LLM or returns zeros."""
    body = await request.json()
    # Most local LLMs don't support embeddings; return a simple response
    input_text = body.get("input", "")
    if isinstance(input_text, str):
        input_text = [input_text]

    # Try to proxy to LLM
    try:
        resp = http_client.post("/embeddings", json=body)
        if resp.status_code == 200:
            return JSONResponse(resp.json())
    except Exception:
        pass

    # Fallback: return zero vectors
    data = []
    for i, _ in enumerate(input_text):
        data.append({"object": "embedding", "index": i, "embedding": [0.0] * 384})
    return JSONResponse({
        "object": "list",
        "data": data,
        "model": body.get("model", "text-embedding-3-small"),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


# ===================
# Realtime (WebSocket)
# ===================

@app.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    """Realtime WebSocket (OpenAI-compatible). Proxies to openclaw-voice."""
    await websocket.accept()
    logger.info("Realtime WebSocket connection accepted")

    # This is a simplified handler. Full implementation would proxy
    # to openclaw-voice's /ws endpoint with protocol translation.
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Echo back for now (placeholder)
            await websocket.send_json({"type": "response", "data": msg})
    except Exception as e:
        logger.error(f"Realtime WS error: {e}")


@app.post("/v1/realtime/calls")
async def realtime_calls(request: Request):
    """Realtime calls (OpenAI-compatible). Stub."""
    body = await request.json()
    return JSONResponse({
        "id": f"call-{uuid.uuid4().hex[:12]}",
        "status": "created",
    })


# ===================
# Live Sessions
# ===================

@app.get("/v1/live")
async def live():
    """Live sessions (OpenAI-compatible). Stub."""
    return JSONResponse({"object": "list", "data": []})


@app.get("/v1/live/sessions")
async def live_sessions():
    """List live sessions (OpenAI-compatible). Stub."""
    return JSONResponse({"object": "list", "data": []})


@app.get("/v1/live/sessions/{session_id}")
async def live_session_detail(session_id: str):
    """Live session details (OpenAI-compatible). Stub."""
    return JSONResponse({
        "id": session_id,
        "status": "not_found",
    })


# ===================
# Organization
# ===================

@app.get("/v1/organization/costs")
async def organization_costs():
    """Organization costs (OpenAI-compatible). Returns zeros."""
    return JSONResponse({
        "object": "list",
        "data": [],
        "has_more": False,
    })


@app.get("/v1/organization/usage/completions")
async def organization_usage():
    """Usage completions (OpenAI-compatible). Returns zeros."""
    return JSONResponse({
        "object": "list",
        "data": [],
        "has_more": False,
    })


# ===================
# Auth
# ===================

@app.post("/auth")
async def auth(request: Request):
    """Authentication (OpenAI-compatible). Always succeeds locally."""
    return JSONResponse({"status": "ok", "token": "local-no-auth-needed"})


import argparse
import signal
import sys
from pathlib import Path

PID_FILE = Path("/tmp/openai_mimic.pid")
CONFIG_FILE = Path(os.path.dirname(os.path.abspath(__file__)) + "/.env")

# ===================
# Management endpoints
# ===================

@app.get("/status")
async def server_status():
    """Server status (management endpoint)."""
    return JSONResponse({
        "status": "running",
        "pid": os.getpid(),
        "stt": "whisper" if stt_model else "none",
        "tts": tts_backend,
        "llm": llm_model,
        "llm_url": LLM_URL,
        "port": PORT,
        "host": HOST,
        "uptime_seconds": time.time() - _start_time if '_start_time' in globals() else 0,
    })


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return JSONResponse({
        "HOST": HOST,
        "PORT": PORT,
        "LLM_URL": LLM_URL,
        "WHISPER_MODEL": WHISPER_MODEL,
        "WHISPER_DEVICE": WHISPER_DEVICE,
        "LANGUAGE": LANGUAGE,
        "VOICE_SAMPLE": VOICE_SAMPLE,
        "tts_backend": tts_backend,
        "llm_model": llm_model,
    })


@app.post("/config")
async def update_config(request: Request):
    """Update configuration (edit). Accepts JSON with key-value pairs."""
    body = await request.json()
    updated = {}
    for key, value in body.items():
        updated[key] = value
    return JSONResponse({
        "status": "config_received",
        "note": "Restart server to apply changes",
        "updated": updated,
    })


# ===================
# CLI Management
# ===================

def _is_running() -> tuple:
    """Check if server is running via PID file."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True, pid
        except (ValueError, OSError, ProcessLookupError):
            PID_FILE.unlink(missing_ok=True)
            return False, None
    return False, None


def cmd_start(args):
    """Start the server."""
    running, pid = _is_running()
    if running:
        print(f"Server already running (PID: {pid})")
        return

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    # Clean up PID file on exit
    def cleanup(signum=None, frame=None):
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print(f"Starting OpenAI Mimic server on {HOST}:{PORT}")
    print(f"LLM backend: {LLM_URL}")
    print(f"STT: Whisper ({WHISPER_MODEL})")
    print(f"TTS: {tts_backend}")
    print(f"PID: {os.getpid()}")

    uvicorn.run(app, host=HOST, port=PORT)


def cmd_stop(args):
    """Stop the server."""
    running, pid = _is_running()
    if not running:
        print("Server is not running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to PID {pid}")
        PID_FILE.unlink(missing_ok=True)
    except ProcessLookupError:
        print(f"Process {pid} not found, cleaning up PID file")
        PID_FILE.unlink(missing_ok=True)


def cmd_status(args):
    """Check server status."""
    running, pid = _is_running()
    if running:
        print(f"Server is RUNNING (PID: {pid})")
        # Try to get health from the server
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3)
            data = json.loads(resp.read())
            print(f"  STT: {data.get('stt', 'unknown')}")
            print(f"  TTS: {data.get('tts', 'unknown')}")
            print(f"  LLM: {data.get('llm', 'unknown')}")
            print(f"  Port: {data.get('port', PORT)}")
        except Exception as e:
            print(f"  (Could not reach server: {e})")
    else:
        print("Server is NOT running")


def cmd_edit(args):
    """Edit configuration."""
    print(f"Configuration file: {CONFIG_FILE}")
    print()
    if CONFIG_FILE.exists():
        print("Current .env contents:")
        print("-" * 40)
        print(CONFIG_FILE.read_text())
        print("-" * 40)
    else:
        print("No .env file found. Creating from template...")
        CONFIG_FILE.write_text(
            "# OpenAI Mimic Configuration\n"
            "LLM_URL=http://localhost:1919/v1\n"
            "WHISPER_MODEL=base\n"
            "WHISPER_DEVICE=auto\n"
            "PORT=8768\n"
            "HOST=0.0.0.0\n"
        )
        print("Created .env template")

    print()
    print("To edit, modify the .env file directly:")
    print(f"  nano {CONFIG_FILE}")
    print(f"  # or")
    print(f"  vim {CONFIG_FILE}")
    print()
    print("Available settings:")
    print("  LLM_URL          - Local LLM URL (default: http://localhost:1919/v1)")
    print("  LLM_MODEL        - Model name (auto-detect if not set)")
    print("  WHISPER_MODEL    - Whisper model size (tiny, base, small, medium, large-v3-turbo)")
    print("  WHISPER_DEVICE   - Device (auto, cpu, cuda, mps)")
    print("  LANGUAGE         - Language code or empty for auto-detect")
    print("  PORT             - Server port (default: 8768)")
    print("  HOST             - Server host (default: 0.0.0.0)")
    print("  VOICE_SAMPLE     - Path to voice sample WAV for XTTS cloning")
    print()
    print("Note: Restart server after editing .env")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Mimic Server")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    start_parser = subparsers.add_parser("start", help="Start the server")
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the server")
    stop_parser.set_defaults(func=cmd_stop)

    # status
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.set_defaults(func=cmd_status)

    # edit
    edit_parser = subparsers.add_parser("edit", help="Edit configuration")
    edit_parser.set_defaults(func=cmd_edit)

    args = parser.parse_args()

    if args.command is None:
        # Default: start the server (backward compatible)
        cmd_start(args)
    else:
        args.func(args)


_start_time = time.time()

if __name__ == "__main__":
    main()
