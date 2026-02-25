"""
Ollama client wrapper.

Supports qwen3, llama3, mistral (all locally served via Ollama).
Falls back to the ollama Python package if available; otherwise uses the
raw HTTP API so there is no hard dependency on the SDK.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from config import AVAILABLE_MODELS, DEFAULT_MODEL, OLLAMA_HOST


# ── Low-level HTTP helpers ────────────────────────────────────────────────────

def _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{OLLAMA_HOST}/{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Make sure `ollama serve` is running."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Generate a completion using a local Ollama model.
    Returns the response text as a string.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model!r} not in AVAILABLE_MODELS: {AVAILABLE_MODELS}"
        )

    payload: Dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    result = _post("api/generate", payload)
    return result.get("response", "")


def list_local_models() -> List[str]:
    """Return list of model names currently available in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        return [f"[error] {exc}"]
