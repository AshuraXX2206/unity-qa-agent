"""Registry of known free-tier providers + a factory to build one from config.

Updated 2026-06-24. Adding a new OpenAI-compatible host is a one-line entry —
no new code needed. ``kind`` selects the adapter:

- ``anthropic``     → :class:`AnthropicProvider`
- ``gemini``        → :class:`GeminiProvider`
- ``openai_compat`` → :class:`OpenAICompatProvider` (parameterised by ``base_url``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.llm.base import LLMProvider


@dataclass(frozen=True)
class ProviderSpec:
    name: str               # registry key
    label: str              # human label shown in the setup wizard
    kind: str               # "anthropic" | "gemini" | "openai_compat"
    env_key: str            # env var to read the key from as a fallback
    base_url: str = ""      # required for openai_compat
    key_url: str = ""       # where the user gets a free key
    supports_vision: bool = True
    free_tier: bool = True


PROVIDERS: Dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini", label="Google Gemini (free tier, multimodal)",
        kind="gemini", env_key="GEMINI_API_KEY",
        key_url="https://aistudio.google.com/apikey",
    ),
    "groq": ProviderSpec(
        name="groq", label="Groq (free tier, very fast)",
        kind="openai_compat", env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        key_url="https://console.groq.com/keys",
        supports_vision=True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter", label="OpenRouter (many free models)",
        kind="openai_compat", env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        key_url="https://openrouter.ai/keys",
        supports_vision=True,
    ),
    "cerebras": ProviderSpec(
        name="cerebras", label="Cerebras (free tier, fast)",
        kind="openai_compat", env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        key_url="https://cloud.cerebras.ai/",
        supports_vision=False,
    ),
    "mistral": ProviderSpec(
        name="mistral", label="Mistral (free tier)",
        kind="openai_compat", env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys/",
        supports_vision=True,
    ),
    "together": ProviderSpec(
        name="together", label="Together AI (free credits)",
        kind="openai_compat", env_key="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        key_url="https://api.together.ai/settings/api-keys",
        supports_vision=True,
    ),
    "anthropic": ProviderSpec(
        name="anthropic", label="Anthropic Claude (paid — most capable agent)",
        kind="anthropic", env_key="ANTHROPIC_API_KEY",
        key_url="https://console.anthropic.com/settings/keys",
        free_tier=False,
    ),
}


def make_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> LLMProvider:
    """Instantiate the provider named *provider_name*.

    ``base_url`` overrides the registry default (lets a user point at any
    OpenAI-compatible host not listed in :data:`PROVIDERS`).
    """
    spec = PROVIDERS.get(provider_name)
    kind = spec.kind if spec else "openai_compat"
    key = api_key or (os.getenv(spec.env_key, "") if spec else "")
    url = base_url or (spec.base_url if spec else "")
    vision = spec.supports_vision if spec else True

    if kind == "anthropic":
        from agent.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=key, model=model)
    if kind == "gemini":
        from agent.llm.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=key, model=model)

    # openai_compat (default for any unknown name, given a base_url)
    from agent.llm.openai_compat_provider import OpenAICompatProvider
    if not url:
        raise ValueError(
            f"Provider '{provider_name}' is OpenAI-compatible but no base_url was given"
        )
    return OpenAICompatProvider(
        api_key=key, base_url=url, model=model, supports_vision=vision,
    )
