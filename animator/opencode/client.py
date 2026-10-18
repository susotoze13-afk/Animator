"""Client for the OpenCode backend.

Three transports behind one interface that mirrors
`openrouter.client.OpenRouterClient`, so the per-stage modules (storyboard,
images, narration, region detection) are shared unchanged:

  * text/vision: Pollinations' OpenAI-compatible chat API, using the same key
    the user's OpenCode `image` tool is configured with. (The `opencode` CLI is
    a full agent — it loads skills and ignores `-m` — so it is deliberately not
    used as a chat transport.)
  * images: Pollinations HTTP — the keyed API when a key is available, the free
    legacy endpoint otherwise.
  * speech: the OS text-to-speech (Windows SAPI), rendered to one PCM take so
    the pause locator can time the beats.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from ..openrouter.client import extract_json
from .settings import (
    CHAT_TIMEOUT_SEC,
    DEFAULT_VOICE,
    FREE_IMAGE_MIN_INTERVAL_SEC,
    FREE_TEXT_MIN_INTERVAL_SEC,
    IMAGE_HEIGHT,
    IMAGE_SOURCE,
    IMAGE_TIMEOUT_SEC,
    IMAGE_WIDTH,
    MAX_RETRIES,
    POLLINATIONS_BASE_URL,
    POLLINATIONS_FALLBACK_MODELS,
    POLLINATIONS_FREE_MODEL,
    POLLINATIONS_LEGACY_URL,
    POLLINATIONS_TEXT_URL,
    RETRY_DELAY_SEC,
    SAPI_RATE,
    SD_API_URL,
    SD_HEIGHT,
    SD_WIDTH,
    chat_url,
    pollinations_token,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_IMAGE_STATUS = (402, 429)
# The anonymous free endpoint allows roughly one image per 45s per IP, so the
# free path is serialized and spaced out across scenes.
_free_image_lock = threading.Lock()
_last_free_image_at = [0.0]
_free_text_lock = threading.Lock()
_last_free_text_at = [0.0]
_SYSTEM_JSON = (
    "Respond with a single valid JSON object and nothing else — no prose, "
    "no markdown, no code fences. It must match this JSON schema:\n"
)
_SYSTEM_JSON_COMPACT = (
    "Return ONLY compact JSON matching this schema: no reasoning, no prose, "
    "no markdown, no code fences. Start with '{' and end with '}'.\n"
)


def _compact_schema(obj: Any) -> Any:
    """Drop descriptions/titles so the free reasoning model does not overthink."""
    if isinstance(obj, dict):
        return {
            key: _compact_schema(value)
            for key, value in obj.items()
            if key not in ("description", "title", "default", "examples")
        }
    if isinstance(obj, list):
        return [_compact_schema(item) for item in obj]
    return obj


class OpenCodeError(RuntimeError):
    """Any OpenCode backend failure."""


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


class OpenCodeClient:
    """One client, three transports, one shape the stage modules expect."""

    def __init__(self, *, token: Optional[str] = None, timeout: float = CHAT_TIMEOUT_SEC) -> None:
        self.token = token if token is not None else pollinations_token()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            raise OpenCodeError(
                "no Pollinations key found; set POLLINATIONS_API_KEY or keep OpenCode's "
                "pollinations-shim/token.txt in place"
            )
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    # --- structured chat -------------------------------------------------
    def chat_json(self, model: str, messages: List[Dict[str, Any]], schema: Type[T],
                  *, temperature: Optional[float] = None) -> T:
        try:
            data = self._post_chat(self._payload(model, messages, schema, temperature))
        except OpenCodeError as exc:
            data = self._free_chat_fallback(exc, model, messages, schema, temperature)
        return self._validate(data, schema)

    async def chat_json_async(self, model: str, messages: List[Dict[str, Any]], schema: Type[T],
                              *, temperature: Optional[float] = None) -> T:
        try:
            data = await self._post_chat_async(self._payload(model, messages, schema, temperature))
        except OpenCodeError as exc:
            if self._has_image(messages) or not self._key_unavailable(exc):
                raise
            logger.warning("keyed chat unavailable (%s); using the free text endpoint", exc)
            data = await self._post_chat_async(
                self._payload(model, messages, schema, temperature, legacy=True),
                url=POLLINATIONS_TEXT_URL,
            )
        return self._validate(data, schema)

    def chat_text(self, model: str, messages: List[Dict[str, Any]], *,
                  temperature: Optional[float] = None) -> str:
        """Plain-text chat (no JSON schema), with the same free-endpoint fallback."""
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            return self._content(self._post_chat(payload))
        except OpenCodeError as exc:
            if self._has_image(messages) or not self._key_unavailable(exc):
                raise
            logger.warning("keyed chat unavailable (%s); using the free text endpoint", exc)
            return self._content(self._post_chat(payload, url=POLLINATIONS_TEXT_URL))

    @staticmethod
    def _looks_like_balance(text: str) -> bool:
        lowered = text.lower()
        return (
            "enough credits" in lowered
            or "insufficient balance" in lowered
            or "not enough pollen" in lowered
            or ("top up" in lowered and "pollinations" in lowered)
        )

    @staticmethod
    def _content(data: Dict[str, Any]) -> str:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenCodeError(f"unexpected chat response: {json.dumps(data)[:500]}") from exc
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            # The free endpoint sometimes returns only a reasoning trace.
            content = message.get("reasoning") if isinstance(message, dict) else None
        if not content:
            raise OpenCodeError(f"empty chat reply: {json.dumps(data)[:500]}")
        if OpenCodeClient._looks_like_balance(content):
            raise OpenCodeError(f"chat balance error: {content[:300]}")
        return content

    def _free_chat_fallback(self, exc: OpenCodeError, model: str, messages, schema,
                            temperature) -> Dict[str, Any]:
        # Vision needs the keyed endpoint; the free text endpoint hangs on
        # images, so only text can fall back.
        if self._has_image(messages) or not self._key_unavailable(exc):
            raise exc
        logger.warning("keyed chat unavailable (%s); using the free text endpoint", exc)
        return self._post_chat(
            self._payload(model, messages, schema, temperature, legacy=True),
            url=POLLINATIONS_TEXT_URL,
        )

    def _payload(self, model: str, messages: List[Dict[str, Any]], schema: Type[T],
                 temperature: Optional[float], *, legacy: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._with_schema(messages, schema, compact=legacy),
        }
        # The free legacy endpoint hangs on response_format; the JSON schema
        # instruction in the system prompt is what actually pins the shape.
        if not legacy:
            payload["response_format"] = {"type": "json_object"}
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    @staticmethod
    def _has_image(messages: List[Dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in content
            ):
                return True
        return False

    @staticmethod
    def _key_unavailable(exc: BaseException) -> bool:
        text = str(exc)
        return any(token in text for token in ("401", "402", "UNAUTHORIZED", "INSUFFICIENT_BALANCE", "balance"))

    @staticmethod
    def _with_schema(messages: List[Dict[str, Any]], schema: Type[T], *, compact: bool = False) -> List[Dict[str, Any]]:
        raw = schema.model_json_schema()
        if compact:
            raw = _compact_schema(raw)
        prefix = _SYSTEM_JSON_COMPACT if compact else _SYSTEM_JSON
        instruction = prefix + json.dumps(raw)
        return [{"role": "system", "content": instruction}, *messages]

    @staticmethod
    def _validate(data: Dict[str, Any], schema: Type[T]) -> T:
        content = OpenCodeClient._content(data)
        parsed = OpenCodeClient._parse_json(OpenCodeClient._normalize_json_text(content))
        try:
            return schema.model_validate(parsed)
        except ValidationError as first:
            # Free chat models are not held to the JSON schema, so the usual
            # slip is a bounding box in pixel space (e.g. 1024) instead of the
            # documented 0-1000, or a positional list where named fields are
            # expected. Repair those and retry once before giving up.
            OpenCodeClient._repair_reply(parsed)
            try:
                return schema.model_validate(parsed)
            except ValidationError as exc:
                raise OpenCodeError(
                    f"model reply did not match {schema.__name__}: {exc}; "
                    f"raw: {json.dumps(parsed, ensure_ascii=False)[:800]}"
                ) from first

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        """Undo the typographic quotes free models often emit."""
        return (
            text.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\ufeff", "")
        )

    @staticmethod
    def _parse_json(text: str) -> Any:
        try:
            return extract_json(text)
        except Exception as first:  # noqa: BLE001 - try a lenient repair
            start, end = text.find("{"), text.rfind("}")
            snippet = text[start:end + 1] if start != -1 and end > start else text
            candidates = [snippet, re.sub(r",\s*([}\]])", r"\1", snippet)]
            for candidate in candidates:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            raise OpenCodeError(f"could not parse JSON: {first}; raw={text[:400]!r}") from first

    _COORD_KEYS = ("ymin", "xmin", "ymax", "xmax")
    _INT_KEYS = ("reveal_order", "beat")

    @classmethod
    def _repair_reply(cls, obj: Any) -> None:
        """Coerce common off-schema shapes in place; no-op for other schemas.

        Clamps bounding-box coordinates into 0-1000, converts a positional
        four-value ``box`` list into named fields, and rounds integer fields a
        free model may have emitted as floats.
        """
        if isinstance(obj, dict):
            box = obj.get("box")
            if (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)
            ):
                obj["box"] = dict(zip(cls._COORD_KEYS, box))
            for key, value in list(obj.items()):
                if key in cls._COORD_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                    obj[key] = max(0, min(1000, int(round(value))))
                elif key in cls._INT_KEYS and isinstance(value, float):
                    obj[key] = int(round(value))
                else:
                    cls._repair_reply(value)
        elif isinstance(obj, list):
            for item in obj:
                cls._repair_reply(item)

    # --- images (Pollinations) ------------------------------------------
    @staticmethod
    def _size(aspect_ratio: Optional[str]) -> Tuple[int, int]:
        if aspect_ratio == "16:9":
            return IMAGE_WIDTH, IMAGE_HEIGHT
        return 1024, 1024

    def generate_image(self, model: str, prompt: str, *, aspect_ratio: Optional[str] = None) -> bytes:
        width, height = self._size(aspect_ratio)
        seed = random.randint(0, 999_999)
        if IMAGE_SOURCE in ("auto", "sdwebui"):
            # A local SD WebUI is free and follows the whiteboard style far
            # better than the free Pollinations tier, so try it first.
            try:
                from ..sdwebui.client import generate_image as sd_generate, sd_available

                if sd_available():
                    sd_w, sd_h = (SD_WIDTH, SD_HEIGHT) if aspect_ratio == "16:9" else (512, 512)
                    return sd_generate(prompt, sd_w, sd_h, seed=seed)
                if IMAGE_SOURCE == "sdwebui":
                    raise OpenCodeError(f"SD WebUI is not reachable at {SD_API_URL}")
            except OpenCodeError:
                raise
            except Exception as exc:
                if IMAGE_SOURCE == "sdwebui":
                    raise OpenCodeError(f"SD WebUI failed: {exc}") from exc
                logger.warning("SD WebUI unavailable (%s); using Pollinations", exc)

        errors: List[str] = []
        if self.token:
            for candidate in dict.fromkeys([model, *POLLINATIONS_FALLBACK_MODELS]):
                try:
                    return self._image_new(prompt, candidate, width, height, seed)
                except OpenCodeError as exc:
                    errors.append(f"{candidate}: {exc}")
                    logger.warning("pollinations %s failed: %s", candidate, exc)
                    if "balance" in str(exc).lower():
                        # Out of pollen: no keyed model will work, so go
                        # straight to the free endpoint instead of trying the
                        # rest of the chain.
                        break
        # Keyed models need pollen; the anonymous legacy endpoint is the free
        # path and is rate-limited, so it is serialized and paced.
        try:
            return self._image_free(prompt, width, height, seed)
        except OpenCodeError as exc:
            errors.append(f"free: {exc}")
            raise OpenCodeError("image generation failed: " + "; ".join(errors[-3:])) from exc

    def _image_new(self, prompt: str, model: str, width: int, height: int, seed: int) -> bytes:
        url = (
            f"{POLLINATIONS_BASE_URL}/image/{quote(prompt, safe='')}"
            f"?model={quote(model, safe='')}&width={width}&height={height}&seed={seed}"
        )
        return self._fetch_image(url, {"Authorization": f"Bearer {self.token}"})

    def _image_free(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        url = (
            f"{POLLINATIONS_LEGACY_URL}/prompt/{quote(prompt, safe='')}"
            f"?width={width}&height={height}&model={quote(POLLINATIONS_FREE_MODEL, safe='')}"
            f"&nologo=true&seed={seed}"
        )
        with _free_image_lock:
            for attempt in range(1, 4):
                wait = FREE_IMAGE_MIN_INTERVAL_SEC - (time.monotonic() - _last_free_image_at[0])
                if wait > 0:
                    time.sleep(wait)
                try:
                    return self._fetch_image(url, {}, retries=1)
                except OpenCodeError as exc:
                    _last_free_image_at[0] = time.monotonic()
                    message = str(exc)
                    if attempt < 3 and (
                        "FREE_LIMIT" in message
                        or "429" in message
                        or "404" in message
                        or "request failed" in message
                        or "timed out" in message
                        or " 5" in message
                    ):
                        logger.warning("free image not ready; retry %s/3 (%s)", attempt, message[:120])
                        continue
                    raise
        raise OpenCodeError("free image generation exhausted retries")  # pragma: no cover

    @staticmethod
    def _fetch_image(url: str, headers: Dict[str, str], *, retries: int = 3) -> bytes:
        for attempt in range(1, retries + 1):
            try:
                response = httpx.get(url, headers=headers, timeout=IMAGE_TIMEOUT_SEC, follow_redirects=True)
            except httpx.HTTPError as exc:
                if attempt >= retries:
                    raise OpenCodeError(f"image request failed: {exc}") from exc
                continue
            content_type = response.headers.get("content-type", "")
            if response.status_code == 200 and content_type.startswith("image/"):
                return response.content
            body = response.text[:200].replace("\n", " ")
            # A zero balance is not worth retrying within this call — the
            # caller falls through to the free endpoint instead.
            retryable = (
                response.status_code >= 500
                or (
                    response.status_code in _RETRYABLE_IMAGE_STATUS
                    and "balance" not in body.lower()
                    and "FREE_LIMIT" not in body
                )
            )
            if attempt < retries and retryable:
                continue
            raise OpenCodeError(f"image HTTP {response.status_code}: {body}")
        raise OpenCodeError("image request exhausted retries")

    # --- speech (Windows SAPI) ------------------------------------------
    def speech_pcm(self, model: str, text: str, voice: str) -> Tuple[bytes, int, int]:
        """Synthesize one take and return raw PCM so the beat locator can run."""
        if os.name != "nt":
            raise OpenCodeError(
                "the OpenCode backend narrates with Windows SAPI; run on Windows or pick another backend"
            )
        with wave.open(io.BytesIO(self._sapi_wav(text, voice)), "rb") as handle:
            rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise OpenCodeError(f"SAPI returned {width * 8}-bit audio; expected 16-bit PCM")
        return frames, rate, channels

    @staticmethod
    def _sapi_wav(text: str, voice: str) -> bytes:
        handle, name = tempfile.mkstemp(prefix="animator-sapi-", suffix=".wav")
        os.close(handle)  # SAPI needs to open the path itself
        out = Path(name)
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $s.SelectVoice($env:ANIMATOR_VOICE) } catch {}; "
            "$s.Rate = [int]$env:ANIMATOR_RATE; "
            "$s.SetOutputToWaveFile($env:ANIMATOR_OUT); "
            "$s.Speak($env:ANIMATOR_TEXT); "
            "$s.Dispose()"
        )
        env = dict(
            os.environ,
            ANIMATOR_TEXT=text,
            ANIMATOR_OUT=str(out),
            ANIMATOR_VOICE=voice or DEFAULT_VOICE,
            ANIMATOR_RATE=str(SAPI_RATE),
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, timeout=180,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise OpenCodeError(f"SAPI synthesis failed: {exc}") from exc
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            detail = (proc.stderr or "").strip()[:300]
            raise OpenCodeError(f"SAPI synthesis failed: {detail or 'no audio produced'}")
        try:
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)

    # --- transport -------------------------------------------------------
    @staticmethod
    def _free_text_delay() -> float:
        return max(0.0, FREE_TEXT_MIN_INTERVAL_SEC - (time.monotonic() - _last_free_text_at[0]))

    def _post_chat(self, payload: Dict[str, Any], *, url: Optional[str] = None) -> Dict[str, Any]:
        url = url or chat_url()
        free = url == POLLINATIONS_TEXT_URL
        attempts = 3 if free else MAX_RETRIES
        for attempt in range(1, attempts + 1):
            try:
                if free:
                    with _free_text_lock:
                        wait = self._free_text_delay()
                        if wait > 0:
                            time.sleep(wait)
                        response = httpx.post(url, json=payload, headers={}, timeout=self.timeout)
                        _last_free_text_at[0] = time.monotonic()
                else:
                    response = httpx.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
            except httpx.HTTPError as exc:
                if free:
                    _last_free_text_at[0] = time.monotonic()
                if attempt >= attempts:
                    raise OpenCodeError(f"chat request failed: {exc}") from exc
                logger.warning("chat request error (%s); retry %s/%s", exc, attempt, attempts)
                if not free:
                    time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            if response.status_code == 200:
                return response.json()
            body = response.text
            if free and "FREE_LIMIT" in body and attempt < attempts:
                logger.warning("free text limit hit; retry %s/%s", attempt, attempts)
                continue
            if not free and attempt < attempts and _retryable(response.status_code):
                logger.warning("chat %s; retry %s/%s", response.status_code, attempt, attempts)
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            raise OpenCodeError(f"chat failed ({response.status_code}): {body[:500]}")
        raise OpenCodeError("chat exhausted retries")  # pragma: no cover

    async def _post_chat_async(self, payload: Dict[str, Any], *, url: Optional[str] = None) -> Dict[str, Any]:
        url = url or chat_url()
        free = url == POLLINATIONS_TEXT_URL
        attempts = 3 if free else MAX_RETRIES
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, attempts + 1):
                if free:
                    wait = self._free_text_delay()
                    if wait > 0:
                        await asyncio.sleep(wait)
                try:
                    response = await client.post(
                        url, json=payload, headers={} if free else self._headers(),
                    )
                except httpx.HTTPError as exc:
                    if free:
                        _last_free_text_at[0] = time.monotonic()
                    if attempt >= attempts:
                        raise OpenCodeError(f"chat request failed: {exc}") from exc
                    logger.warning("chat request error (%s); retry %s/%s", exc, attempt, attempts)
                    if not free:
                        await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                if free:
                    _last_free_text_at[0] = time.monotonic()
                if response.status_code == 200:
                    return response.json()
                body = response.text
                if free and "FREE_LIMIT" in body and attempt < attempts:
                    logger.warning("free text limit hit; retry %s/%s", attempt, attempts)
                    continue
                if not free and attempt < attempts and _retryable(response.status_code):
                    logger.warning("chat %s; retry %s/%s", response.status_code, attempt, attempts)
                    await asyncio.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                raise OpenCodeError(f"chat failed ({response.status_code}): {body[:500]}")
        raise OpenCodeError("chat exhausted retries")  # pragma: no cover
