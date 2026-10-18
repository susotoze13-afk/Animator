"""Local Stable Diffusion WebUI image source (free, no API key)."""
from .client import generate_image, sanitize_prompt, sd_available

__all__ = ["generate_image", "sanitize_prompt", "sd_available"]
