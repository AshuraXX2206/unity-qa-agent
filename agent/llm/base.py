"""Core types and the abstract base class every LLM provider implements.

The agentic loop in :mod:`agent.qa_agent` speaks a single *neutral* message
format and lets each provider translate it to/from its own wire shape. A
message is a dict ``{"role": "user"|"assistant", "content": [block, ...]}``
where each block is one of:

- ``{"type": "text", "text": str}``
- ``{"type": "image", "data": <base64 png str>}``
- ``{"type": "tool_use", "id": str, "name": str, "input": dict}``   (assistant)
- ``{"type": "tool_result", "tool_use_id": str, "content": [block, ...]}`` (user)

Keeping the neutral format tiny means a new provider only has to map these few
block kinds, not the full surface of any one vendor SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMResponse:
    """One assistant turn, normalised across providers."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = ""           # "end_turn" | "tool_use" | "refusal" | ...
    raw: Any = None                 # provider-native response, for debugging


@dataclass
class ModelInfo:
    """A model advertised by a provider's ``/models`` endpoint."""

    id: str
    display_name: str = ""
    created_at: str = ""            # ISO 8601 or epoch string; "" if unknown
    supports_vision: bool = False
    supports_tools: bool = True

    def sort_key(self) -> str:
        """Sortable recency key — newest models sort last."""
        return self.created_at or self.id


class LLMProvider(ABC):
    """Abstract interface for a chat model with tool use and (optional) vision."""

    #: Whether this provider/model can accept image content blocks.
    supports_vision: bool = True
    #: Whether this provider/model can use function/tool calling.
    supports_tools: bool = True

    def __init__(self, api_key: str, model: str = "") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Query the provider's ``/models`` endpoint for available models."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        """Run one assistant turn.

        Parameters
        ----------
        system:
            System prompt text.
        messages:
            Conversation in the neutral format described in the module docstring.
        tools:
            Neutral tool schemas (see :mod:`agent.agent_tools`): each is
            ``{"name", "description", "input_schema": <JSON Schema>}``.
        """
