"""STEP 4 (OpenRouter): detect a region plan from a finished whiteboard image."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List

from PIL import Image

from ..gemini.schemas import DetectedRegion, DetectionResponse
from ..prompts import load_prompt
from ..regions import Region, SnippetRegionPlan
from .client import OpenRouterClient
from .settings import DEFAULT_REGION_MODEL


def build_region_prompt(transcript_text: str, idea: str, beats: List[str] | None = None) -> str:
    prompt = load_prompt("region_detection").format(idea=idea, transcript_text=transcript_text)
    if not beats:
        return prompt
    numbered = "\n".join(f"  {index}. {text}" for index, text in enumerate(beats, start=1))
    return f"{prompt}\n\n{load_prompt('region_beats').format(numbered=numbered)}"


def to_region(detected: DetectedRegion) -> Region:
    return Region(
        label=detected.label,
        role=detected.role,
        object=detected.object,
        reveal_order=detected.reveal_order,
        box=detected.box,
        expected_visual=detected.object,
        annotation=detected.annotation,
        reveal=detected.reveal,
        beat=detected.beat or None,
    )


async def detect_region_plan(
    client: OpenRouterClient,
    *,
    image_path: Path,
    transcript_text: str,
    idea: str = "",
    model: str = DEFAULT_REGION_MODEL,
    beats: List[str] | None = None,
) -> SnippetRegionPlan:
    idea = idea or "(untitled)"
    with Image.open(image_path) as image:
        size = list(image.size)
        mime_type = Image.MIME.get(image.format, "image/png")

    data_url = "data:{0};base64,{1}".format(
        mime_type, base64.b64encode(image_path.read_bytes()).decode("ascii"),
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": build_region_prompt(transcript_text, idea, beats)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]
    parsed = await client.chat_json_async(model, messages, DetectionResponse)
    if parsed is None or not parsed.regions:
        raise RuntimeError("Region detection returned no regions")

    return SnippetRegionPlan(
        idea=idea,
        global_style_notes="",
        regions=[to_region(detected) for detected in parsed.regions],
        image_size=size,
    )
