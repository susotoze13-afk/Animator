"""STEP 1 for the opencode backend: a script the free text model can produce.

The only anonymous Pollinations text model is a reasoning model capped at about
1500 completion tokens. Asking it for the full StoryboardPlan JSON makes it
reason past that cap and return prose, so the script is requested as plain
"CAPTION: sentence" lines and assembled into the shared StoryboardPlan here.
If the model is unavailable, a small template still lets the pipeline render.

Gemini/OpenRouter keep the richer shared storyboard prompt.
"""
from __future__ import annotations

import logging
import math
import re
from typing import List, Optional, Tuple

from ..gemini.schemas import Beat, StoryboardPlan, StoryboardScene
from .client import OpenCodeClient
from .settings import DEFAULT_TEXT_MODEL

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You write short whiteboard explainer scripts. Reply with plain lines only: "
    "CAPTION: one short sentence that says the caption words out loud. "
    "No JSON, no markdown, no numbering, no extra text."
)
_LAYOUT = (
    "Spread the icons across a wide 16:9 frame and connect them left to right "
    "with thin hand-drawn arrows."
)


def build_prompt(idea: str, num_scenes: Optional[int]) -> str:
    count = max(6, (num_scenes or 3) * 4)
    return (
        f'Write a whiteboard explainer about "{idea}" as {count} lines. '
        "Each line: a 1-3 word CAPTION, a colon, then one short sentence that says those "
        "caption words out loud. Cover the topic from start to finish; the first line names "
        "the topic and the last line is the takeaway."
    )


def _parse_lines(text: str) -> List[Tuple[str, str]]:
    beats: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        match = re.match(r"^([^:]{2,48}):\s*(.+)$", line)
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group(1)).strip().strip('"').upper()
        narration = re.sub(r"\s+", " ", match.group(2)).strip().strip('"')
        # Drop chain-of-thought leakage ("1. "CAPTION" : ... Count: ...").
        label = re.sub(r"[^A-Za-z0-9 \-]", "", label).strip()
        lowered = narration.lower()
        if any(marker in lowered for marker in ("count:", "caption", "word ok", "let's ", "we need")):
            continue
        if len(label) < 2 or len(label) > 40 or len(narration) < 5:
            continue
        beats.append((label, narration))
    return beats


def _plan_from_beats(idea: str, beats: List[Tuple[str, str]], num_scenes: Optional[int]) -> StoryboardPlan:
    scene_count = max(1, num_scenes or 3)
    per_scene = max(2, math.ceil(len(beats) / scene_count))
    scenes: List[StoryboardScene] = []
    for start in range(0, len(beats), per_scene):
        chunk = beats[start:start + per_scene]
        if not chunk:
            continue
        scenes.append(
            StoryboardScene(
                title=chunk[0][0].title(),
                beats=[
                    Beat(
                        label=label,
                        visual=f"simple doodle icon of {label.lower()}",
                        narration=narration,
                    )
                    for label, narration in chunk
                ],
                layout=_LAYOUT,
            )
        )
    if not scenes:
        raise RuntimeError("Storyboard assembly produced no scenes")
    return StoryboardPlan(title=idea.strip().rstrip("?").title() or "Explainer", scenes=scenes)


def _template(idea: str, num_scenes: Optional[int]) -> StoryboardPlan:
    topic = idea.strip().rstrip("?").title() or "Explainer"
    scene_count = max(1, num_scenes or 3)
    scenes: List[StoryboardScene] = []
    for index in range(scene_count):
        if index == 0:
            title = "What It Is"
            beats = [
                ("TOPIC", f"{topic} is the subject of this video."),
                ("QUESTION", "This video answers one clear question about it."),
                ("WHY IT MATTERS", "It matters because it affects everyday life."),
            ]
        elif index == scene_count - 1:
            title = "Takeaway"
            beats = [
                ("RESULT", "The main steps fit together into a clear result."),
                ("TAKEAWAY", f"The key thing to remember about {topic.lower()} is the big picture."),
            ]
        else:
            title = f"Step {index}"
            beats = [
                ("STEP", "First, the important parts are set up."),
                ("CHANGE", "Then they interact and change."),
                ("NEXT", "Finally, the next step follows."),
            ]
        scenes.append(
            StoryboardScene(
                title=title,
                beats=[
                    Beat(label=label, visual=f"simple doodle icon of {label.lower()}", narration=narration)
                    for label, narration in beats
                ],
                layout=_LAYOUT,
            )
        )
    return StoryboardPlan(title=topic, scenes=scenes)


def generate_storyboard(
    client: OpenCodeClient,
    idea: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    num_scenes: Optional[int] = None,
) -> StoryboardPlan:
    prompt = build_prompt(idea, num_scenes)
    for _ in range(2):
        try:
            text = client.chat_text(
                model,
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            beats = _parse_lines(text)
            if len(beats) >= 2:
                return _plan_from_beats(idea, beats, num_scenes)
            logger.warning("storyboard model returned no usable lines: %r", text[:200])
        except Exception as exc:  # noqa: BLE001 - one retry, then the template
            logger.warning("storyboard text attempt failed: %s", exc)
    logger.warning("storyboard model unavailable; using the built-in template")
    return _template(idea, num_scenes)
