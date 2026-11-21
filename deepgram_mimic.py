#!/usr/bin/env python3
"""
Deepgram Mimic Server

Receives audio via WebSocket (Deepgram protocol), runs local Whisper + VAD,
and returns transcripts in Deepgram's JSON format.

This allows the OpenClaw Android app to use local Whisper STT instead of
Deepgram's cloud service. The app sends audio to this server thinking it's
talking to Deepgram, but gets back transcripts from your local Whisper model.

Usage:
    PYTHONPATH=. python deepgram_mimic.py

Environment variables:
    WHISPER_MODEL    - Whisper model size (tiny, base, small, medium, large-v3-turbo). Default: small
    WHISPER_DEVICE   - Device (auto, cpu, cuda, mps). Default: auto
    LANGUAGE         - Language code or None for auto-detect. Default: None
    PORT             - Server port. Default: 8766
    HOST             - Server host. Default: 0.0.0.0
    LOG_LEVEL        - Logging level. Default: INFO
"""

import asyncio
import json
import os
import time
from typing import Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# Logging (use loguru if available, fallback to stdlib)
try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8766"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
LANGUAGE = os.getenv("LANGUAGE", None)  # None = auto-detect

# Audio parameters (Deepgram standard)
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

# VAD parameters
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.0"))
INTERIM_INTERVAL = float(os.getenv("INTERIM_INTERVAL", "0.5"))
MIN_AUDIO_LENGTH = 0.1  # Minimum 100ms of audio to transcribe


class DeepgramMimic:
    """Deepgram Mimic Server - local STT with Deepgram-compatible API."""

    def __init__(self):
        self.model = None
        self.vad_model = None
        self.vad_utils = None
        self._load_models()

    def _load_models(self):
        """Load Whisper and Silero VAD models."""
        # Load Whisper
        logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on device '{WHISPER_DEVICE}'...")
        try:
            from faster_whisper import WhisperModel
            import torch

            if WHISPER_DEVICE == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "cpu"
                    compute_type = "int8"
                else:
                    device = "cpu"
                    compute_type = "int8"
            elif WHISPER_DEVICE == "cuda":
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu" if WHISPER_DEVICE in ("cpu", "mps") else WHISPER_DEVICE
                compute_type = "int8"

            self.model = WhisperModel(
                WHISPER_MODEL,
                device=device if device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._whisper_backend = "faster-whisper"
            logger.info(f"✅ faster-whisper loaded on {device}")
        except ImportError:
            logger.warning("faster-whisper not available, trying openai-whisper...")
            try:
                import whisper
                import torch

                if WHISPER_DEVICE == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = "cpu" if WHISPER_DEVICE == "mps" else WHISPER_DEVICE

                self.model = whisper.load_model(WHISPER_MODEL, device=device)
                self._whisper_backend = "openai-whisper"
                logger.info(f"✅ openai-whisper loaded on {device}")
            except ImportError:
                logger.error("❌ No Whisper backend available. Install faster-whisper or openai-whisper.")
                self._whisper_backend = "mock"
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
            self._whisper_backend = "mock"

        # Load Silero VAD
        logger.info("Loading Silero VAD model...")
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.vad_model = model
            # utils: [get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks]
            self.vad_iterator_cls = utils[3]  # VADIterator
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.warning(f"⚠️ Silero VAD not available: {e}")
            self.vad_model = None

    def pcm_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 numpy array."""
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio_array.astype(np.float32) / 32768.0

    def has_speech(self, audio_float32: np.ndarray) -> bool:
        """Check if audio contains speech using energy-based VAD."""
        if len(audio_float32) == 0:
            return False
        # Simple energy-based VAD
        energy = np.abs(audio_float32).max()
        return energy > SILENCE_THRESHOLD

    def transcribe_audio(self, audio_float32: np.ndarray, is_final: bool = True) -> Optional[dict]:
        """Run Whisper transcription and return Deepgram-compatible JSON."""
        if len(audio_float32) < int(SAMPLE_RATE * MIN_AUDIO_LENGTH):
            return None

        # Run Whisper
        transcript_text = ""
        try:
            if self._whisper_backend == "faster-whisper":
                segments, info = self.model.transcribe(
                    audio_float32,
                    language=LANGUAGE,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                transcript_text = " ".join([seg.text for seg in segments]).strip()
            elif self._whisper_backend == "openai-whisper":
                result = self.model.transcribe(audio_float32, language=LANGUAGE)
                transcript_text = result["text"].strip()
            else:
                # Mock mode
                transcript_text = "[Mock transcription - install whisper for real STT]"
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

        if not transcript_text:
            return None

        # Build Deepgram-compatible response
        duration = len(audio_float32) / SAMPLE_RATE

        # Build words array
        words = []
        words_list = transcript_text.split()
        word_duration = duration / len(words_list) if words_list else 0

        for i, word in enumerate(words_list):
            words.append({
                "word": word,
                "start": round(i * word_duration, 3),
                "end": round((i + 1) * word_duration, 3),
                "confidence": 0.95,
            })

        response = {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": round(duration, 3),
            "start": 0.0,
            "is_final": is_final,
            "speech_final": is_final,
            "channel": {
                "alternatives": [
                    {
                        "transcript": transcript_text,
                        "confidence": 0.95,
                        "words": words,
                    }
                ]
            },
        }

        return response

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket connection from OpenClaw Android app."""
        path = getattr(websocket, "path", None) or getattr(getattr(websocket, "request", None), "path", "/")
        logger.info(f"New connection from {websocket.remote_address} (path: {path})")

        # Parse query parameters (Deepgram protocol)
        params = {}
        if "?" in path:
            query_string = path.split("?")[1]
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value

        logger.info(f"Connection params: {params}")

        # Audio buffer
        audio_buffer = bytearray()
        last_speech_time = time.time()
        last_interim_time = 0.0

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary audio data (Deepgram sends raw PCM)
                    audio_buffer.extend(message)

                    # Convert to float32 for processing
                    audio_float32 = self.pcm_to_float32(bytes(audio_buffer))

                    current_time = time.time()
                    speech_detected = self.has_speech(audio_float32)

                    if speech_detected:
                        last_speech_time = current_time

                        # Send interim transcript every INTERIM_INTERVAL
                        if current_time - last_interim_time > INTERIM_INTERVAL:
                            interim_result = self.transcribe_audio(audio_float32, is_final=False)
                            if interim_result:
                                await websocket.send(json.dumps(interim_result))
                            last_interim_time = current_time
                    else:
                        # Silence detected - check if enough silence to finalize
                        if (
                            current_time - last_speech_time > SILENCE_DURATION
                            and len(audio_buffer) > 0
                        ):
                            # Send final transcript
                            final_result = self.transcribe_audio(audio_float32, is_final=True)
                            if final_result:
                                await websocket.send(json.dumps(final_result))

                            # Clear buffer
                            audio_buffer.clear()
                            logger.info("Speech segment completed, buffer cleared")

                elif isinstance(message, str):
                    # JSON control message
                    try:
                        control_msg = json.loads(message)
                        msg_type = control_msg.get("type", "")

                        if msg_type == "CloseStream":
                            logger.info("Received CloseStream, sending final transcript if any")
                            if len(audio_buffer) > 0:
                                audio_float32 = self.pcm_to_float32(bytes(audio_buffer))
                                final_result = self.transcribe_audio(audio_float32, is_final=True)
                                if final_result:
                                    await websocket.send(json.dumps(final_result))
                            # Send a final empty result to signal completion
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "Results",
                                        "channel_index": [0, 1],
                                        "duration": 0.0,
                                        "start": 0.0,
                                        "is_final": True,
                                        "speech_final": True,
                                        "channel": {
                                            "alternatives": [{"transcript": "", "confidence": 0.0, "words": []}]
                                        },
                                    }
                                )
                            )
                            break

                        elif msg_type == "Finalize":
                            logger.info("Received Finalize request")
                            if len(audio_buffer) > 0:
                                audio_float32 = self.pcm_to_float32(bytes(audio_buffer))
                                final_result = self.transcribe_audio(audio_float32, is_final=True)
                                if final_result:
                                    await websocket.send(json.dumps(final_result))
                                audio_buffer.clear()

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON control message: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
        finally:
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def run(self):
        """Run the Deepgram Mimic server."""
        logger.info(f"Starting Deepgram Mimic server on {HOST}:{PORT}")
        async with websockets.serve(self.handle_client, HOST, PORT):
            logger.info(f"Server ready. Point OpenClaw streaming STT to: ws://{HOST}:{PORT}/v1/listen")
            await asyncio.Future()  # Run forever


def main():
    mimic = DeepgramMimic()
    asyncio.run(mimic.run())


if __name__ == "__main__":
    main()
