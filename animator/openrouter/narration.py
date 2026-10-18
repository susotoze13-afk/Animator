"""STEP 3 (OpenRouter): speak a scene's beats as one take and time each beat.

Reuses the Gemini backend's pause locator (../gemini/beat_timing.py): the TTS
endpoint returns raw PCM, so beat windows come out of the same silence analysis.
"""
from __future__ import annotations

import subprocess
from typing import List, Tuple

from ..gemini.beat_timing import PCM_RATE, locate_beats, pcm_to_wav
from .client import OpenRouterClient, OpenRouterError
from .settings import DEFAULT_TTS_MODEL, DEFAULT_VOICE


def to_mono_24k(pcm: bytes, rate: int, channels: int) -> bytes:
    """Normalise OpenRouter PCM to the 24 kHz mono the beat locator expects."""
    if rate == PCM_RATE and channels == 1:
        return pcm
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-i", "pipe:0",
        "-f", "s16le", "-ar", str(PCM_RATE), "-ac", "1", "-acodec", "pcm_s16le", "pipe:1",
    ]
    result = subprocess.run(command, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise OpenRouterError(
            f"ffmpeg could not resample narration audio: {result.stderr.decode(errors='replace')[:300]}"
        )
    return result.stdout


def generate_beat_narration(
    client: OpenRouterClient,
    beats: List[str],
    *,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
) -> Tuple[bytes, List[List[float]]]:
    """Synthesize a scene's beats as one take and locate each beat in it.

    Returns (wav_bytes, windows) where windows[i] is the [start, end] in seconds
    of beat i inside the WAV, so drawing can be locked to speech.
    """
    text = " ".join(beat.strip() for beat in beats)
    pcm, rate, channels = client.speech_pcm(model, text, voice)
    pcm24 = to_mono_24k(pcm, rate, channels)
    return pcm_to_wav(pcm24), locate_beats(pcm24, beats)
