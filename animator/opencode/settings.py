"""Defaults for the OpenCode backend.

The backend runs the whole --prompt pipeline on services the user's OpenCode is
already set up with, so no Google AI Studio or OpenRouter key is needed:

  * text + vision (storyboard, region detection): Pollinations' OpenAI-compatible
    chat API — the same service and API key OpenCode's `image` tool uses. (The
    `opencode` CLI itself is a full agent, not a plain chat endpoint, so it is not
    used as the model transport.)
  * images: Pollinations.
  * narration: the OS text-to-speech (Windows SAPI).

Everything is overridable in .env.
"""
from __future__ import annotations

import os
from pathlib import Path

# Text/vision models on Pollinations. `openai` is the built-in default that
# works with a plain (possibly zero-balance) key; vision is supported.
DEFAULT_TEXT_MODEL = os.getenv("OPENCODE_TEXT_MODEL", "openai")
DEFAULT_REGION_MODEL = os.getenv("OPENCODE_REGION_MODEL", "openai")

# Images.
DEFAULT_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "black-forest-labs/flux.1.1-pro")
POLLINATIONS_BASE_URL = os.getenv("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai")
POLLINATIONS_LEGACY_URL = os.getenv("POLLINATIONS_LEGACY_URL", "https://image.pollinations.ai")
# Free, anonymous OpenAI-compatible text endpoint. It ignores response_format
# and does not accept images; used when the keyed endpoint is out of pollen or
# the key is rejected, so text (storyboard) still works at zero balance.
POLLINATIONS_TEXT_URL = os.getenv("POLLINATIONS_TEXT_URL", "https://text.pollinations.ai/openai")
# The anonymous legacy endpoint is rate-limited (about one image per 45s per
# IP), so free generations are serialized and paced. It ignores the paid model
# slugs and serves its own free model.
POLLINATIONS_FREE_MODEL = os.getenv("POLLINATIONS_FREE_MODEL", "flux")
FREE_IMAGE_MIN_INTERVAL_SEC = float(os.getenv("POLLINATIONS_FREE_MIN_INTERVAL_SEC", "46"))
# Free-tier models verified to work on a zero-balance Pollinations key; tried in
# turn when the requested model is out of balance.
POLLINATIONS_FALLBACK_MODELS = (
    "black-forest-labs/flux.1.1-pro",
    "black-forest-labs/flux.2-klein-4b",
    "black-forest-labs/flux.1-schnell",
    "tongyi-mai/z-image-turbo",
)
# 16:9 with both sides a multiple of 32 (FLUX rejects anything else).
IMAGE_WIDTH = int(os.getenv("POLLINATIONS_IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("POLLINATIONS_IMAGE_HEIGHT", "576"))
# Pollinations key: env first, then the token the OpenCode image shim keeps.
POLLINATIONS_TOKEN_FILE = Path(
    os.getenv("POLLINATIONS_TOKEN_FILE", str(Path.home() / ".config" / "opencode" / "pollinations-shim" / "token.txt"))
)

# Narration: OS text-to-speech (Windows SAPI). The Gemini voice slugs are
# ignored; set a Windows voice here (e.g. "Microsoft Zira Desktop", "Microsoft Irina Desktop").
DEFAULT_TTS_MODEL = os.getenv("OPENCODE_TTS_MODEL", "sapi")
DEFAULT_VOICE = os.getenv("OPENCODE_SAPI_VOICE", "Microsoft Zira Desktop")
SAPI_RATE = int(os.getenv("OPENCODE_SAPI_RATE", "0"))

MAX_RETRIES = int(os.getenv("OPENCODE_MAX_RETRIES", "4"))
RETRY_DELAY_SEC = float(os.getenv("OPENCODE_RETRY_DELAY_SEC", "6"))
CHAT_TIMEOUT_SEC = float(os.getenv("OPENCODE_CHAT_TIMEOUT_SEC", "300"))
IMAGE_TIMEOUT_SEC = float(os.getenv("POLLINATIONS_TIMEOUT_SEC", "300"))


def pollinations_token() -> str:
    """Env key if set, else the OpenCode shim's token.txt, else ''."""
    token = (os.getenv("POLLINATIONS_API_KEY") or "").strip()
    if token:
        return token
    try:
        return POLLINATIONS_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def chat_url() -> str:
    return os.getenv("POLLINATIONS_CHAT_URL", POLLINATIONS_BASE_URL.rstrip("/") + "/v1/chat/completions")
