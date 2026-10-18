"""Free local image source: Stable Diffusion WebUI (AUTOMATIC1111/Forge).

The `opencode` backend prefers this when a local SD WebUI is reachable, so
whiteboard frames cost nothing and do not depend on Pollinations pollen. It
talks to the standard `/sdapi/v1/txt2img` endpoint used by the
Kolyamba_News_animation project.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE = (
    "blackboard, chalkboard, dark background, wood, wooden table, photo, "
    "photorealistic, 3d render, shading, shadow, gradient, color, color fill, "
    "texture, letters, words, caption, text, frame, border, clutter, pattern, "
    "signature, watermark, low quality"
)
# SD 1.5 renders any quoted caption as gibberish, so captions/headings are
# stripped and the frame is steered hard toward black marker on white paper.
_STYLE_PREFIX = (
    "whiteboard sketch, black marker on white paper, high contrast, "
    "generous white space, no color, no shading, no text, no letters, "
)


def sanitize_prompt(prompt: str) -> str:
    """Turn the shared whiteboard prompt into something SD 1.5 handles well."""
    text = re.sub(r'—\s*caption\s*"[^"]*"', "", prompt, flags=re.IGNORECASE)
    text = re.sub(r'caption\s*"[^"]*"', "", text, flags=re.IGNORECASE)
    text = re.sub(r'Heading at the top center:\s*"[^"]*"\.?', "", text, flags=re.IGNORECASE)
    text = text.replace("flat solid color fills", "flat 2D")
    text = re.sub(r"\s+", " ", text).strip()
    if not text.lower().startswith(_STYLE_PREFIX.split(",")[0]):
        text = _STYLE_PREFIX + text
    return text


def _settings():
    from ..opencode import settings

    return settings


def sd_available(url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """True when an SD WebUI API answers at `url`."""
    base = (url or _settings().SD_API_URL).rstrip("/")
    try:
        response = httpx.get(f"{base}/sdapi/v1/options", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def generate_image(
    prompt: str,
    width: int,
    height: int,
    *,
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    sampler: Optional[str] = None,
    seed: Optional[int] = None,
    timeout: Optional[float] = None,
) -> bytes:
    """Render one PNG frame through the local SD WebUI."""
    settings = _settings()
    base = settings.SD_API_URL.rstrip("/")
    payload = {
        "prompt": sanitize_prompt(prompt),
        "negative_prompt": negative_prompt or settings.SD_NEGATIVE_PROMPT or DEFAULT_NEGATIVE,
        "width": int(width),
        "height": int(height),
        "steps": int(steps if steps is not None else settings.SD_STEPS),
        "cfg_scale": float(cfg_scale if cfg_scale is not None else settings.SD_CFG),
        "sampler_name": sampler or settings.SD_SAMPLER,
        "seed": -1 if seed is None else int(seed),
        "batch_size": 1,
        "n_iter": 1,
        "send_images": True,
        "save_images": False,
    }
    if settings.SD_MODEL:
        payload["override_settings"] = {"sd_model_checkpoint": settings.SD_MODEL}
        payload["override_settings_restore_afterwards"] = True

    logger.info("generating %dx%d frame via SD WebUI at %s", width, height, base)
    with httpx.Client(timeout=timeout or settings.SD_TIMEOUT_SEC) as client:
        response = client.post(f"{base}/sdapi/v1/txt2img", json=payload)
        response.raise_for_status()
        data = response.json()

    images = data.get("images") or []
    if not images:
        raise RuntimeError("SD WebUI returned no images")
    return base64.b64decode(images[0])
