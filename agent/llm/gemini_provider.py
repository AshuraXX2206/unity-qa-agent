"""Native Google Gemini provider (free tier).

Uses ``google-generativeai``. Model discovery via ``genai.list_models()``; chat
via the ``generateContent`` surface with function declarations and inline image
parts.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List

from agent.llm.base import LLMProvider, LLMResponse, ModelInfo, ToolCall

log = logging.getLogger(__name__)

_FALLBACK_MODEL = "gemini-2.5-flash"


class GeminiProvider(LLMProvider):
    supports_vision = True
    supports_tools = True

    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__(api_key or os.getenv("GEMINI_API_KEY", ""), model)
        self._configured = False

    def _ensure_configured(self) -> Any:
        import google.generativeai as genai
        if not self._configured:
            genai.configure(api_key=self._api_key)
            self._configured = True
        return genai

    # ── model discovery ───────────────────────────────────────────────

    def list_models(self) -> List[ModelInfo]:
        genai = self._ensure_configured()
        out: List[ModelInfo] = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            name = m.name.split("/")[-1]  # "models/gemini-2.5-flash" → "gemini-2.5-flash"
            out.append(ModelInfo(
                id=name,
                display_name=getattr(m, "display_name", name),
                created_at=getattr(m, "version", "") or name,
                supports_vision=True,  # all current Gemini chat models are multimodal
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
        genai = self._ensure_configured()
        model_name = self._model or _FALLBACK_MODEL

        gemini_tools = None
        if tools:
            gemini_tools = [{
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": _clean_schema(t["input_schema"]),
                    }
                    for t in tools
                ]
            }]

        model = genai.GenerativeModel(
            model_name,
            system_instruction=system or None,
            tools=gemini_tools,
        )

        contents = [self._to_gemini_content(m) for m in messages]

        try:
            resp = model.generate_content(contents)
        except Exception as exc:  # noqa: BLE001
            log.error("Gemini chat failed: %s", exc)
            return LLMResponse(text=f"[provider error: {exc}]", stop_reason="error")

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        n = 0
        try:
            parts = resp.candidates[0].content.parts
        except (AttributeError, IndexError):
            parts = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", ""):
                n += 1
                tool_calls.append(ToolCall(
                    id=f"call_{n}",
                    name=fc.name,
                    input=dict(fc.args) if fc.args else {},
                ))
            elif getattr(part, "text", ""):
                text_parts.append(part.text)

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=resp,
        )

    # ── neutral → gemini translation ──────────────────────────────────

    @staticmethod
    def _to_gemini_content(msg: Dict[str, Any]) -> Dict[str, Any]:
        role = "model" if msg["role"] == "assistant" else "user"
        parts: List[Dict[str, Any]] = []
        for block in msg["content"]:
            bt = block["type"]
            if bt == "text":
                parts.append({"text": block["text"]})
            elif bt == "image":
                parts.append({"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64decode(block["data"]),
                }})
            elif bt == "tool_use":
                parts.append({"function_call": {
                    "name": block["name"],
                    "args": block["input"],
                }})
            elif bt == "tool_result":
                text = " ".join(
                    b["text"] for b in block["content"] if b["type"] == "text"
                )
                # Gemini's function_response is keyed by name, but the neutral
                # block only carries an id; the model matches on the preceding
                # function_call, so a generic key is fine here.
                parts.append({"function_response": {
                    "name": block.get("name", "tool"),
                    "response": {"result": text or "[image observation in next turn]"},
                }})
        return {"role": role, "parts": parts}


def _clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON-Schema keys Gemini's function declarations reject."""
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "$schema"):
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        elif isinstance(v, dict):
            out[k] = _clean_schema(v)
        else:
            out[k] = v
    return out
