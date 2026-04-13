"""Model registries, BedrockChatClient, and unified client factory.

This module centralises provider configuration that was previously
duplicated across experiment scripts.
"""

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import boto3

# ---------------------------------------------------------------------------
# Model registries — friendly_name -> (model_id, input_$/Mtok, output_$/Mtok)
# ---------------------------------------------------------------------------

BEDROCK_MODELS = {
    "claude-sonnet-4.6":  ("us.anthropic.claude-sonnet-4-6",                  3.00, 15.00),
    "claude-sonnet-4.5":  ("us.anthropic.claude-sonnet-4-5-20250929-v1:0",    3.00, 15.00),
    "claude-sonnet-4":    ("us.anthropic.claude-sonnet-4-20250514-v1:0",      3.00, 15.00),
    "claude-haiku-4.5":   ("us.anthropic.claude-haiku-4-5-20251001-v1:0",     1.00,  5.00),
    "claude-haiku-3.5":   ("us.anthropic.claude-3-5-haiku-20241022-v1:0",     0.80,  4.00),
    "claude-haiku-3":     ("us.anthropic.claude-3-haiku-20240307-v1:0",       0.25,  1.25),
    "claude-opus-4.6":    ("us.anthropic.claude-opus-4-6-v1",                 5.00, 25.00),
    "claude-opus-4":      ("us.anthropic.claude-opus-4-20250514-v1:0",       15.00, 75.00),
}

OPENROUTER_MODELS = {
    "kimi-k2.5":       ("moonshotai/kimi-k2.5",       0.50,  2.80),
    "minimax-m2.5":    ("minimax/minimax-m2.5",       0.30,  1.10),
    "minimax-m2-her":  ("minimax/minimax-m2-her",     0.30,  1.10),
    "minimax-m2":      ("minimax/minimax-m2",         0.30,  1.10),
}

OPENAI_MODELS = {
    "gpt-5.3-codex":  ("gpt-5.3-codex",   2.00, 10.00),
    "gpt-5.2":        ("gpt-5.2",          1.75, 14.00),
    "gpt-5":          ("gpt-5",            1.25, 10.00),
    "gpt-5-mini":     ("gpt-5-mini",       0.25,  2.00),
    "gpt-4.1":        ("gpt-4.1",          2.00,  8.00),
    "gpt-4.1-mini":   ("gpt-4.1-mini",     0.40,  1.60),
    "gpt-4.1-nano":   ("gpt-4.1-nano",     0.10,  0.40),
}

PROVIDER_REGISTRIES = {
    "bedrock":    BEDROCK_MODELS,
    "openrouter": OPENROUTER_MODELS,
    "openai":     OPENAI_MODELS,
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def resolve_model_id(name_or_id: str, provider: str = "bedrock") -> str:
    """Accept either a friendly alias or a full model ID for any provider.

    If *name_or_id* is a key in the provider's registry the corresponding
    full model ID is returned; otherwise the value is passed through as-is.
    """
    registry = PROVIDER_REGISTRIES.get(provider, BEDROCK_MODELS)
    if name_or_id in registry:
        return registry[name_or_id][0]
    return name_or_id


def get_pricing(model_id: str, provider: str = None) -> Tuple[float, float]:
    """Return ``(input_price_per_Mtok, output_price_per_Mtok)`` for a model.

    Searches the specified provider registry (or all registries when
    *provider* is ``None``).  Falls back to ``(3.0, 15.0)`` if unknown.
    """
    registries = [PROVIDER_REGISTRIES[provider]] if provider else PROVIDER_REGISTRIES.values()
    for registry in registries:
        for _, (mid, ip, op) in registry.items():
            if mid == model_id:
                return ip, op
    return 3.0, 15.0


def list_models(provider: str = None) -> Dict[str, dict]:
    """Return a dict of ``friendly_name -> {id, input_price, output_price}``."""
    if provider:
        registry = PROVIDER_REGISTRIES.get(provider, {})
    else:
        registry = {}
        for reg in PROVIDER_REGISTRIES.values():
            registry.update(reg)
    return {
        name: {"id": mid, "input_price": ip, "output_price": op}
        for name, (mid, ip, op) in registry.items()
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible response dataclasses for Bedrock
# ---------------------------------------------------------------------------


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _ChatCompletion:
    choices: List[_Choice]
    usage: _Usage
    model: str
    id: str = ""


# ---------------------------------------------------------------------------
# Internal plumbing — mimics openai.chat.completions
# ---------------------------------------------------------------------------


class _Completions:
    def __init__(self, wrapper: "BedrockChatClient"):
        self._w = wrapper

    def create(
        self,
        model: str = None,
        messages: list = None,
        temperature: float = 1.0,
        max_tokens: int = 100,
        **kwargs,
    ) -> _ChatCompletion:
        model_id = resolve_model_id(model) if model else self._w._default_model

        system_parts: list = []
        bedrock_msgs: list = []

        for msg in (messages or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": content})
            else:
                bedrock_msgs.append({"role": role, "content": [{"text": content}]})

        request: dict = {
            "modelId": model_id,
            "messages": bedrock_msgs,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": max(float(temperature), 0.0),
            },
        }
        if system_parts:
            request["system"] = system_parts

        response = self._call_with_retries(request)

        content_blocks = (
            response.get("output", {}).get("message", {}).get("content", [])
        )
        text = "\n".join(b["text"] for b in content_blocks if "text" in b)

        usage_data = response.get("usage", {})
        in_tok = usage_data.get("inputTokens", 0)
        out_tok = usage_data.get("outputTokens", 0)

        with self._w._lock:
            self._w.total_input_tokens += in_tok
            self._w.total_output_tokens += out_tok

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=text))],
            usage=_Usage(
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
            model=model_id,
        )

    def _call_with_retries(self, request: dict, max_retries: int = 6) -> dict:
        for attempt in range(max_retries):
            try:
                return self._w._client.converse(**request)
            except self._w._client.exceptions.ThrottlingException:
                wait = min(2 ** attempt, 64)
                time.sleep(wait)
            except Exception as e:
                err = str(e)
                if "ThrottlingException" in err or "TooManyRequests" in err:
                    wait = min(2 ** attempt, 64)
                    time.sleep(wait)
                else:
                    raise
        return self._w._client.converse(**request)


class _Chat:
    def __init__(self, wrapper: "BedrockChatClient"):
        self.completions = _Completions(wrapper)


# ---------------------------------------------------------------------------
# Public Bedrock client
# ---------------------------------------------------------------------------


class BedrockChatClient:
    """Drop-in replacement for ``openai.OpenAI``.

    Supports ``client.chat.completions.create(...)`` with the same signature
    used by the experiment scripts.
    """

    def __init__(
        self,
        api_key: str,
        region: str = "us-east-1",
        default_model: str = None,
    ):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._default_model = (
            resolve_model_id(default_model)
            if default_model
            else "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        self._lock = threading.Lock()

        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

        self.chat = _Chat(self)

    def get_usage_summary(self) -> dict:
        """Return a dict summarising token usage and estimated cost."""
        ip, op = get_pricing(self._default_model)
        input_cost = (self.total_input_tokens / 1_000_000) * ip
        output_cost = (self.total_output_tokens / 1_000_000) * op
        return {
            "model": self._default_model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }

    def reset_usage(self):
        """Zero out accumulated token counters."""
        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0


# ---------------------------------------------------------------------------
# Unified client factory
# ---------------------------------------------------------------------------


def create_client(
    provider: str,
    api_key: str,
    base_url: str = None,
    bedrock_region: str = "us-east-1",
    model: str = None,
):
    """Create an LLM client and resolve the model ID in one step.

    Returns:
        ``(client, resolved_model)`` — *client* exposes the standard
        ``client.chat.completions.create(...)`` interface regardless of
        the back-end provider.
    """
    import openai as _openai

    if provider == "bedrock":
        resolved = resolve_model_id(model, provider="bedrock") if model else model
        client = BedrockChatClient(
            api_key=api_key,
            region=bedrock_region,
            default_model=resolved,
        )
        return client, resolved

    if provider == "openrouter":
        resolved = resolve_model_id(model, provider="openrouter") if model else model
        client = _openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        return client, resolved

    if provider == "vllm":
        client = _openai.OpenAI(api_key=api_key, base_url=base_url)
        return client, model

    # Default: openai-compatible
    resolved = resolve_model_id(model, provider="openai") if model else model
    url = base_url or OPENAI_BASE_URL
    client = _openai.OpenAI(api_key=api_key, base_url=url)
    return client, resolved
