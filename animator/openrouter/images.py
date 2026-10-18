"""STEP 2 (OpenRouter): draw one storyboard scene as a 16:9 whiteboard image."""
from __future__ import annotations

from ..gemini.schemas import StoryboardScene
from ..prompts import load_prompt
from .client import OpenRouterClient
from .settings import DEFAULT_IMAGE_MODEL


def build_image_prompt(scene: StoryboardScene) -> str:
    elements = "\n".join(
        f'{index}. {beat.visual.strip()} — caption "{beat.label.strip().upper()}"'
        for index, beat in enumerate(scene.beats, start=1)
    )
    frame = load_prompt("scene_image").format(
        title=scene.title.strip().upper(),
        count=len(scene.beats),
        elements=elements,
        layout=scene.layout.strip(),
    )
    return f"{frame}\n{load_prompt('image_style')}"


def generate_scene_image(
    client: OpenRouterClient,
    scene: StoryboardScene,
    *,
    model: str = DEFAULT_IMAGE_MODEL,
) -> bytes:
    """Render one frame of separate flat doodle icons. Returns image bytes."""
    return client.generate_image(model, build_image_prompt(scene), aspect_ratio="16:9")
