"""Native Anthropic (Claude) provider.

Uses the official ``anthropic`` SDK. Model discovery goes through the Models API
(``client.models.list()``); the agentic loop uses adaptive thinking and the
manual tool-use protocol.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.llm.base import LLMProvider, LLMResponse, ModelInfo, ToolCall

log = logging.getLogger(__name__)

# Safe fallback if the Models API can't be reached. Kept current as of 2026-06.
_FALLBACK_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    supports_vision = True
    supports_tools = True

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY", ""), model)
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ── model discovery ───────────────────────────────────────────────

    def list_models(self) -> List[ModelInfo]:
        client = self._ensure_client()
        out: List[ModelInfo] = []
        for m in client.models.list():
            caps = getattr(m, "capabilities", None) or {}
            vision = _cap(caps, "image_input")
            out.append(ModelInfo(
                id=m.id,
                display_name=getattr(m, "display_name", m.id),
                created_at=str(getattr(m, "created_at", "")),
                supports_vision=vision if caps else True,
                supports_tools=True,
            ))
        return out

    # ── chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        client = self._ensure_client()
        model = self._model or _FALLBACK_MODEL

        anthropic_messages = [self._to_anthropic_message(m) for m in messages]
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

        in_tok = out_tok = 0
        # Loop to transparently continue past pause_turn (server-tool iteration cap).
        for _ in range(4):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system,
                    thinking={"type": "adaptive"},
                    tools=anthropic_tools,
                    messages=anthropic_messages,
                )
            except Exception as exc:  # noqa: BLE001 — surface as a turn, don't crash the loop
                log.error("Anthropic chat failed: %s", exc)
                return LLMResponse(text=f"[provider error: {exc}]", stop_reason="error")

            usage = getattr(resp, "usage", None)
            if usage is not None:
                in_tok += getattr(usage, "input_tokens", 0) or 0
                out_tok += getattr(usage, "output_tokens", 0) or 0

            if resp.stop_reason != "pause_turn":
                break
            # Resume: feed the paused assistant turn back and continue.
            anthropic_messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "refusal":
            return LLMResponse(
                text="[model refused this request]",
                stop_reason="refusal",
                usage={"input_tokens": in_tok, "output_tokens": out_tok},
                raw=resp,
            )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            raw=resp,
        )

    # ── neutral → anthropic translation ───────────────────────────────

    @staticmethod
    def _to_anthropic_message(msg: Dict[str, Any]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for block in msg["content"]:
            content.append(AnthropicProvider._to_anthropic_block(block))
        return {"role": msg["role"], "content": content}

    @staticmethod
    def _to_anthropic_block(block: Dict[str, Any]) -> Dict[str, Any]:
        bt = block["type"]
        if bt == "text":
            return {"type": "text", "text": block["text"]}
        if bt == "image":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": block["data"],
                },
            }
        if bt == "tool_use":
            return {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": block["input"],
            }
        if bt == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": block["tool_use_id"],
                "content": [
                    AnthropicProvider._to_anthropic_block(b) for b in block["content"]
                ],
            }
        raise ValueError(f"Unknown neutral block type: {bt}")


def _cap(caps: Any, name: str) -> bool:
    """Safely read ``capabilities[name]['supported']`` from the Models API dict."""
    try:
        return bool(caps[name]["supported"])
    except (KeyError, TypeError):
        return False
