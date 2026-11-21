"""
Tests for Deepgram Mimic Server.
"""

import asyncio
import json
import socket
import subprocess
import sys
import time
import os

import pytest
import numpy as np


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def mimic_server():
    """Start the Deepgram Mimic server for testing."""
    port = 8798
    if is_port_in_use(port):
        pytest.skip(f"Port {port} already in use")

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["WHISPER_MODEL"] = "tiny"
    env["WHISPER_DEVICE"] = "cpu"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))

    proc = subprocess.Popen(
        [sys.executable, "deepgram_mimic.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    max_wait = 60
    for i in range(max_wait):
        if is_port_in_use(port):
            break
        time.sleep(1)
    else:
        proc.terminate()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        pytest.fail(f"Server did not start in time. stderr: {stderr[-500:]}")

    yield f"ws://127.0.0.1:{port}"

    proc.terminate()
    proc.wait(timeout=10)


class TestDeepgramMimicConnection:
    """Test WebSocket connection to Deepgram Mimic server."""

    @pytest.mark.asyncio
    async def test_connect(self, mimic_server):
        import websockets

        async with websockets.connect(f"{mimic_server}/v1/listen") as ws:
            assert ws is not None

    @pytest.mark.asyncio
    async def test_close_stream(self, mimic_server):
        import websockets

        async with websockets.connect(f"{mimic_server}/v1/listen") as ws:
            await ws.send(json.dumps({"type": "CloseStream"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert msg["type"] == "Results"
            assert msg["is_final"] is True
            assert msg["speech_final"] is True

    @pytest.mark.asyncio
    async def test_audio_silence(self, mimic_server):
        import websockets

        async with websockets.connect(f"{mimic_server}/v1/listen") as ws:
            # Send silence
            silence = np.zeros(16000, dtype=np.int16).tobytes()
            await ws.send(silence)
            await asyncio.sleep(1.5)
            await ws.send(json.dumps({"type": "CloseStream"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert msg["type"] == "Results"
            assert msg["is_final"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
