"""Generic OpenAI-compatible provider.

Most free-tier LLM hosts (Groq, OpenRouter, Cerebras, Mistral, Together,
DeepInfra, …) expose the OpenAI ``/v1/chat/completions`` + ``/v1/models``
surface, so a single adapter parameterised by ``base_url`` covers all of them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx

from agent.llm.base import LLMProvider, LLMResponse, ModelInfo, ToolCall

log = logging.getLogger(__name__)

_TIMEOUT = 60.0


class OpenAICompatProvider(LLMProvider):
    supports_tools = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "",
        supports_vision: bool = True,
        extra_headers: Dict[str, str] | None = None,
    ) -> None:
        super().__init__(api_key, model)
        self._base_url = base_url.rstrip("/")
        self.supports_vision = supports_vision
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }

    # ── model discovery ───────────────────────────────────────────────

    def list_models(self) -> List[ModelInfo]:
        url = f"{self._base_url}/models"
        resp = httpx.get(url, headers=self._headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        out: List[ModelInfo] = []
        for m in data:
            mid = m.get("id", "")
            if not mid:
                continue
            out.append(ModelInfo(
                id=mid,
                display_name=m.get("name", mid),
                created_at=str(m.get("created", "")),
                # The OpenAI /models payload rarely advertises modality; assume
                # the provider-level default and let model_selector refine.
                supports_vision=self.supports_vision,
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
        oa_messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            oa_messages.extend(self._to_openai_messages(m))

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": oa_messages,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.error("OpenAI-compat chat failed: %s", exc)
            return LLMResponse(text=f"[provider error: {exc}]", stop_reason="error")

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "")

        tool_calls: List[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), input=args))

        stop_reason = "tool_use" if finish == "tool_calls" else (
            "end_turn" if finish == "stop" else finish
        )
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=data,
        )

    # ── neutral → openai translation ──────────────────────────────────

    def _to_openai_messages(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """One neutral message may expand into several OpenAI messages.

        OpenAI splits a tool result into its own ``role: "tool"`` message, and
        keeps assistant tool calls in a dedicated ``tool_calls`` field, so a
        single neutral turn can map to multiple wire messages.
        """
        role = msg["role"]
        text_parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for block in msg["content"]:
            bt = block["type"]
            if bt == "text":
                text_parts.append({"type": "text", "text": block["text"]})
            elif bt == "image":
                if self.supports_vision:
                    text_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{block['data']}"},
                    })
                else:
                    text_parts.append({"type": "text", "text": "[screenshot omitted: model has no vision]"})
            elif bt == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                })
            elif bt == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": self._flatten_result_content(block["content"]),
                })

        out: List[Dict[str, Any]] = []
        if role == "assistant" and tool_calls:
            out.append({
                "role": "assistant",
                "content": self._collapse_text(text_parts),
                "tool_calls": tool_calls,
            })
        elif text_parts:
            out.append({"role": role, "content": text_parts})
        out.extend(tool_results)
        return out

    def _flatten_result_content(self, blocks: List[Dict[str, Any]]) -> Any:
        """OpenAI ``role:tool`` content. Images can't ride here on most hosts,
        so describe them as text and rely on the next user turn for vision."""
        parts: List[Dict[str, Any]] = []
        for b in blocks:
            if b["type"] == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image":
                parts.append({"type": "text", "text": "[screenshot captured — see next message]"})
        return self._collapse_text(parts)

    @staticmethod
    def _collapse_text(parts: List[Dict[str, Any]]) -> Any:
        """Use a plain string when there's only text, else the parts array."""
        if parts and all(p.get("type") == "text" for p in parts):
            return "\n".join(p["text"] for p in parts)
        return parts or ""
