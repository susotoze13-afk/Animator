"""Pick the generation backend for --prompt.

Three backends expose the same four stage functions behind a single
`StoryboardBackend`, so the pipeline in ../gemini/pipeline.py stays
backend-agnostic:

    gemini      Google AI Studio key (text, vision, image, speech)
    openrouter  OpenRouter key (text, vision, image, speech)
    opencode    the user's OpenCode harness + Pollinations + OS text-to-speech,
                so no Google/OpenRouter key is needed at all

Selection order:
    1. explicit argument to select_backend()
    2. $ANIMATOR_BACKEND ('gemini', 'openrouter' or 'opencode')
    3. auto: OpenRouter when $OPENROUTER_API_KEY is set, else Gemini when a
       Google key is set, else OpenCode
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StoryboardBackend:
    """Everything the pipeline needs, for one provider."""

    name: str
    client: Any
    text_model: str
    region_model: str
    image_model: str
    tts_model: str
    default_voice: str
    generate_storyboard: Callable
    generate_scene_image: Callable
    generate_beat_narration: Callable
    detect_region_plan: Callable


def gemini_backend(*, api_key: str | None = None, client: Any = None) -> StoryboardBackend:
    """Gemini backend for an explicit key/client (the web API passes a per-user key)."""
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - deps are in requirements.txt
        raise SystemExit("the Gemini backend needs the gemini extra: pip install -r requirements.txt") from exc

    if client is None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("needs GEMINI_API_KEY (or GOOGLE_API_KEY) for the Gemini backend")
        client = genai.Client(api_key=api_key)

    from .gemini import detect_regions, images, narration, storyboard
    from .gemini.settings import (
        DEFAULT_IMAGE_MODEL,
        DEFAULT_REGION_MODEL,
        DEFAULT_TEXT_MODEL,
        DEFAULT_TTS_MODEL,
        DEFAULT_VOICE,
    )

    return StoryboardBackend(
        name="gemini",
        client=client,
        text_model=DEFAULT_TEXT_MODEL,
        region_model=DEFAULT_REGION_MODEL,
        image_model=DEFAULT_IMAGE_MODEL,
        tts_model=DEFAULT_TTS_MODEL,
        default_voice=DEFAULT_VOICE,
        generate_storyboard=lambda client, idea, **kw: storyboard.generate_storyboard(client, idea, **kw),
        generate_scene_image=lambda client, scene, **kw: images.generate_scene_image(client, scene, **kw),
        generate_beat_narration=lambda client, beats, **kw: narration.generate_beat_narration(client, beats, **kw),
        detect_region_plan=lambda client, **kw: detect_regions.detect_region_plan(google_client=client, **kw),
    )


def openrouter_backend(*, api_key: str | None = None) -> StoryboardBackend:
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("needs OPENROUTER_API_KEY for the OpenRouter backend (https://openrouter.ai/keys)")

    from .openrouter import detect_regions, images, narration, storyboard
    from .openrouter.client import OpenRouterClient
    from .openrouter.settings import (
        DEFAULT_IMAGE_MODEL,
        DEFAULT_REGION_MODEL,
        DEFAULT_TEXT_MODEL,
        DEFAULT_TTS_MODEL,
        DEFAULT_VOICE,
    )

    return StoryboardBackend(
        name="openrouter",
        client=OpenRouterClient(api_key),
        text_model=DEFAULT_TEXT_MODEL,
        region_model=DEFAULT_REGION_MODEL,
        image_model=DEFAULT_IMAGE_MODEL,
        tts_model=DEFAULT_TTS_MODEL,
        default_voice=DEFAULT_VOICE,
        generate_storyboard=storyboard.generate_storyboard,
        generate_scene_image=images.generate_scene_image,
        generate_beat_narration=narration.generate_beat_narration,
        detect_region_plan=detect_regions.detect_region_plan,
    )


async def _opencode_region_plan(client, **kwargs):
    """Region detection, degraded when vision is unavailable.

    The free tier cannot see images, so when vision fails we return None and
    the renderer draws the frame sequentially over the narration instead of
    aborting the whole video.
    """
    import logging

    from .openrouter import detect_regions

    try:
        return await detect_regions.detect_region_plan(client, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a missing plan is recoverable
        logging.getLogger(__name__).warning(
            "region detection unavailable (%s); drawing the frame sequentially", exc
        )
        return None


def opencode_backend(*, client: Any = None) -> StoryboardBackend:
    """Backend driven by the user's OpenCode install.

    Text/vision run on Pollinations (the key OpenCode's `image` tool already
    has, with a free legacy text fallback); images come from Pollinations'
    free endpoint when the pollen balance is empty; narration from the OS
    text-to-speech. The per-stage modules are shared with the OpenRouter
    backend — they only ever call the client interface `OpenCodeClient` mirrors.
    """
    from .opencode.client import OpenCodeClient
    from .opencode.settings import (
        DEFAULT_IMAGE_MODEL,
        DEFAULT_REGION_MODEL,
        DEFAULT_TEXT_MODEL,
        DEFAULT_TTS_MODEL,
        DEFAULT_VOICE,
    )
    from .opencode import storyboard as oc_storyboard
    from .openrouter import images, narration

    return StoryboardBackend(
        name="opencode",
        client=client or OpenCodeClient(),
        text_model=DEFAULT_TEXT_MODEL,
        region_model=DEFAULT_REGION_MODEL,
        image_model=DEFAULT_IMAGE_MODEL,
        tts_model=DEFAULT_TTS_MODEL,
        default_voice=DEFAULT_VOICE,
        generate_storyboard=oc_storyboard.generate_storyboard,
        generate_scene_image=images.generate_scene_image,
        generate_beat_narration=narration.generate_beat_narration,
        detect_region_plan=_opencode_region_plan,
    )


def select_backend(name: str | None = None) -> StoryboardBackend:
    chosen = (name or os.environ.get("ANIMATOR_BACKEND") or "").strip().lower()
    if not chosen:
        if os.environ.get("OPENROUTER_API_KEY"):
            chosen = "openrouter"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            chosen = "gemini"
        else:
            chosen = "opencode"
    if chosen == "gemini":
        return gemini_backend()
    if chosen == "openrouter":
        return openrouter_backend()
    if chosen == "opencode":
        return opencode_backend()
    raise SystemExit(f"unknown ANIMATOR_BACKEND {chosen!r}; use 'gemini', 'openrouter' or 'opencode'")

