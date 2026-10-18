"""List the OpenRouter models your key can use: `python -m animator.openrouter.discover`.

Model slugs move around, so when a run fails with "model does not exist", look
up the current ones here and put them in .env (OPENROUTER_TEXT_MODEL, ...).

    python -m animator.openrouter.discover                 # text/vision models
    python -m animator.openrouter.discover --type image
    python -m animator.openrouter.discover --type speech   # shows supported voices
    python -m animator.openrouter.discover --query gemini
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import httpx

from .settings import BASE_URL


def fetch_models(output_modalities: str | None = None, timeout: float = 60.0) -> List[Dict[str, Any]]:
    params = {"output_modalities": output_modalities} if output_modalities else None
    response = httpx.get(f"{BASE_URL.rstrip('/')}/models", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json().get("data", [])


def _matches(model: Dict[str, Any], query: str | None) -> bool:
    if not query:
        return True
    haystack = f"{model.get('id', '')} {model.get('name', '')}".lower()
    return query.lower() in haystack


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m animator.openrouter.discover")
    parser.add_argument("--type", choices=["text", "image", "speech"], default="text",
                        help="which modality to list (default: text)")
    parser.add_argument("--query", default=None, help="substring filter on id/name")
    args = parser.parse_args(argv)

    modality = None if args.type == "text" else args.type
    try:
        models = fetch_models(modality)
    except httpx.HTTPError as exc:
        print(f"discover: could not reach OpenRouter: {exc}", file=sys.stderr)
        return 1

    rows = [m for m in models if _matches(m, args.query)]
    if not rows:
        print("discover: no models matched.", file=sys.stderr)
        return 1

    for model in sorted(rows, key=lambda m: m.get("id", "")):
        voices = model.get("supported_voices")
        suffix = f"   voices: {', '.join(voices)}" if voices else ""
        print(f"{model.get('id', '?')}{suffix}")
    print(f"\n{len(rows)} {args.type} model(s). Override with OPENROUTER_*_MODEL in .env.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
