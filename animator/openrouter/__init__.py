"""OpenRouter backend: the --prompt pipeline on an OpenRouter key.

Same five steps as the Gemini backend (see ../gemini/__init__.py), but every
call goes to https://openrouter.ai/api/v1:

    chat/completions   storyboard + region detection (structured JSON, vision)
    images             one whiteboard frame per scene
    audio/speech       narration (raw PCM), beat windows recovered from pauses
"""
