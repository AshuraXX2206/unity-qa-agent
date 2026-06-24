"""Provider-agnostic LLM layer for the Unity QA agent.

Exposes a single :class:`LLMProvider` interface implemented by native Anthropic
and Gemini adapters plus a generic OpenAI-compatible adapter that covers most
free-tier providers (Groq, OpenRouter, Cerebras, Mistral, Together, …).
"""

from agent.llm.base import LLMProvider, LLMResponse, ModelInfo, ToolCall
from agent.llm.registry import PROVIDERS, ProviderSpec, make_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ModelInfo",
    "ToolCall",
    "PROVIDERS",
    "ProviderSpec",
    "make_provider",
]
