"""Defaults for the OpenRouter backend.

OpenRouter is OpenAI-compatible and serves text, vision, image and speech
models behind one base URL, so the whole --prompt pipeline can run on an
OpenRouter key instead of a Google AI Studio key.

Model slugs change often; list what your key can use with
`python -m animator.openrouter.discover` and override any of these in .env.
"""
from __future__ import annotations

import os

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Text/vision model for the storyboard and region detection (structured JSON).
DEFAULT_TEXT_MODEL = os.getenv("OPENROUTER_TEXT_MODEL", "google/gemini-3.8-flash")
DEFAULT_REGION_MODEL = os.getenv("OPENROUTER_REGION_MODEL", "google/gemini-3.1-flash-lite")
# Image model (Nano Banana family).
DEFAULT_IMAGE_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL", "google/gemini-3.1-flash-image")
# Text-to-speech model and voice (Google voices; 'Kore' matches the Gemini default).
DEFAULT_TTS_MODEL = os.getenv("OPENROUTER_TTS_MODEL", "google/gemini-2.5-flash-preview-tts")
DEFAULT_VOICE = os.getenv("OPENROUTER_VOICE", "Kore")

MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", os.getenv("GEMINI_MAX_RETRIES", "5")))
RETRY_DELAY_SEC = float(os.getenv("OPENROUTER_RETRY_DELAY_SEC", os.getenv("GEMINI_RETRY_DELAY_SEC", "8")))

# Optional attribution headers OpenRouter shows on its dashboard.
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Animator")
