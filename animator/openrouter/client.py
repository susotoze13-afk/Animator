"""HTTP client for OpenRouter's OpenAI-compatible API.

Covers the three shapes the pipeline needs: structured chat (JSON schema),
image generation and text-to-speech. Retries mirror the Gemini backend:
429 and 5xx are worth another attempt, everything else fails fast.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .settings import APP_TITLE, BASE_URL, MAX_RETRIES, RETRY_DELAY_SEC, SITE_URL

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Defaults for a PCM speech response when the Content-Type omits them.
DEFAULT_PCM_RATE = 24000
DEFAULT_PCM_CHANNELS = 1
_PCM_RATE_RE = re.compile(r"rate=(\d+)")
_PCM_CHANNELS_RE = re.compile(r"channels=(\d+)")


class OpenRouterError(RuntimeError):
    """Any non-retryable OpenRouter failure, with the response body attached."""


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


def _headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": APP_TITLE,
    }
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    return headers


def _json_schema_format(schema: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": schema.model_json_schema(),
        },
    }


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ``` fences and prose around it."""
    if not text:
        raise OpenRouterError("model returned empty content where JSON was expected")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped[:4].lower() == "json":
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError as exc:
                raise OpenRouterError(f"could not parse JSON from model reply: {exc}") from exc
        raise OpenRouterError("model reply contained no JSON object")


def _content_of(data: Dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"unexpected chat response: {json.dumps(data)[:500]}") from exc
    if isinstance(content, list):  # some providers return content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _parse_pcm_headers(content_type: str) -> tuple[int, int]:
    rate = _PCM_RATE_RE.search(content_type or "")
    channels = _PCM_CHANNELS_RE.search(content_type or "")
    return (
        int(rate.group(1)) if rate else DEFAULT_PCM_RATE,
        int(channels.group(1)) if channels else DEFAULT_PCM_CHANNELS,
    )


class OpenRouterClient:
    """Thin wrapper over the OpenRouter endpoints the backend uses."""

    def __init__(self, api_key: str, *, base_url: str = BASE_URL, timeout: float = 300.0) -> None:
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise(self, path: str, response: httpx.Response) -> None:
        hint = ""
        if response.status_code == 404 or "does not exist" in response.text:
            hint = " (check the model slug with `python -m animator.openrouter.discover`)"
        raise OpenRouterError(f"{path} failed ({response.status_code}){hint}: {response.text[:500]}")

    # --- structured chat -------------------------------------------------
    def chat_json(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        schema: Type[T],
        *,
        temperature: Optional[float] = None,
    ) -> T:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": _json_schema_format(schema),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            data = self._post_json("/chat/completions", payload)
        except OpenRouterError as exc:
            if "response_format" not in str(exc) and "json_schema" not in str(exc):
                raise
            logger.warning("model %s rejected json_schema; retrying with json_object", model)
            data = self._post_json("/chat/completions", self._fallback_payload(payload))
        parsed = extract_json(_content_of(data))
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise OpenRouterError(f"model reply did not match {schema.__name__}: {exc}") from exc

    async def chat_json_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        schema: Type[T],
        *,
        temperature: Optional[float] = None,
    ) -> T:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": _json_schema_format(schema),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            data = await self._post_json_async("/chat/completions", payload)
        except OpenRouterError as exc:
            if "response_format" not in str(exc) and "json_schema" not in str(exc):
                raise
            logger.warning("model %s rejected json_schema; retrying with json_object", model)
            data = await self._post_json_async("/chat/completions", self._fallback_payload(payload))
        parsed = extract_json(_content_of(data))
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise OpenRouterError(f"model reply did not match {schema.__name__}: {exc}") from exc

    @staticmethod
    def _fallback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Models without native structured output still get told to return JSON."""
        fallback = dict(payload)
        fallback.pop("response_format", None)
        fallback["messages"] = [
            {"role": "system", "content": "Respond with a single valid JSON object and nothing else."},
            *payload["messages"],
        ]
        return fallback

    # --- images ----------------------------------------------------------
    def generate_image(self, model: str, prompt: str, *, aspect_ratio: Optional[str] = None) -> bytes:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        data = self._post_json("/images", payload)
        items = data.get("data") or []
        if not items:
            raise OpenRouterError(f"image response contained no data: {json.dumps(data)[:500]}")
        item = items[0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            response = httpx.get(item["url"], timeout=self.timeout)
            response.raise_for_status()
            return response.content
        raise OpenRouterError("image response had neither b64_json nor url")

    # --- speech ----------------------------------------------------------
    def speech(self, model: str, text: str, voice: str, *, response_format: str = "pcm") -> httpx.Response:
        payload = {"model": model, "input": text, "voice": voice, "response_format": response_format}
        return self._post_raw("/audio/speech", payload)

    def speech_pcm(self, model: str, text: str, voice: str) -> tuple[bytes, int, int]:
        """Return (pcm_bytes, sample_rate, channels) for raw PCM speech."""
        response = self.speech(model, text, voice, response_format="pcm")
        rate, channels = _parse_pcm_headers(response.headers.get("content-type", ""))
        return response.content, rate, channels

    # --- transport -------------------------------------------------------
    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._post_raw(path, payload)
        try:
            return response.json()
        except ValueError as exc:
            raise OpenRouterError(f"{path} returned non-JSON: {response.text[:500]}") from exc

    def _post_raw(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        url = self._url(path)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = httpx.post(url, json=payload, headers=_headers(self.api_key), timeout=self.timeout)
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    raise OpenRouterError(f"{path} request failed: {exc}") from exc
                logger.warning("%s request error (%s); retry %s/%s", path, exc, attempt, MAX_RETRIES)
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            if response.status_code == 200:
                return response
            if attempt < MAX_RETRIES and _retryable(response.status_code):
                logger.warning("%s %s; retry %s/%s", path, response.status_code, attempt, MAX_RETRIES)
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            self._raise(path, response)
        raise OpenRouterError(f"{path} exhausted retries")  # pragma: no cover

    async def _post_json_async(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._url(path)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.post(url, json=payload, headers=_headers(self.api_key))
                except httpx.HTTPError as exc:
                    if attempt == MAX_RETRIES:
                        raise OpenRouterError(f"{path} request failed: {exc}") from exc
                    logger.warning("%s request error (%s); retry %s/%s", path, exc, attempt, MAX_RETRIES)
                    await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                if response.status_code == 200:
                    return response.json()
                if attempt < MAX_RETRIES and _retryable(response.status_code):
                    logger.warning("%s %s; retry %s/%s", path, response.status_code, attempt, MAX_RETRIES)
                    await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                self._raise(path, response)
        raise OpenRouterError(f"{path} exhausted retries")  # pragma: no cover
