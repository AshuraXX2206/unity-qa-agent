"""Generic OpenAI-compatible provider.

Most free-tier LLM hosts (Groq, OpenRouter, Cerebras, Mistral, Together,
DeepInfra, …) expose the OpenAI ``/v1/chat/completions`` + ``/v1/models``
surface, so a single adapter parameterised by ``base_url`` covers all of them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from agent.llm.base import LLMProvider, LLMResponse, ModelInfo, ToolCall

log = logging.getLogger(__name__)

_TIMEOUT = 60.0

# Free tiers (esp. Groq) rate-limit aggressively but tell you exactly how long to
# wait. Auto-retry a bounded number of times so a single QA run survives a brief
# limit instead of failing outright — but never block the CLI for too long.
_MAX_RETRIES = 2
_RETRY_CAP_SECONDS = 30.0
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)

# A provider's /models listing often mixes in models that can't do chat
# completions at all (speech-to-text, TTS, embeddings, safety classifiers,
# agentic systems). Auto-selecting one of these makes every chat call 400, so
# we drop anything whose id hints at a non-chat modality.
_NON_CHAT_HINTS = (
    "whisper", "distil-whisper", "-asr", "-stt", "speech-to-text",
    "-tts", "text-to-speech", "playai", "orpheus", "sonic",
    "embed", "embedding", "rerank", "reranker", "moderation",
    "guard", "safeguard", "compound",
)

# Models whose id signals image input. Only these are advertised as vision so
# model_selector ranks a *real* multimodal model first instead of any text model
# that a provider blanket-flags as vision-capable.
_VISION_HINTS = (
    "vision", "-vl", "vl-", "llava", "scout", "maverick",
    "llama-4", "llama4", "gpt-4o", "gpt-4.1", "gpt-5", "o4-",
    "pixtral", "gemini", "internvl", "minicpm-v", "multimodal",
)

# Reasoning models (qwen3, gpt-oss, deepseek-r1, …) sometimes emit their chain of
# thought inline as <think>…</think>; strip it so users see only the answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _looks_non_chat(model_id: str) -> bool:
    mid = model_id.lower()
    return any(h in mid for h in _NON_CHAT_HINTS)


def _looks_vision(model_id: str) -> bool:
    mid = model_id.lower()
    return any(h in mid for h in _VISION_HINTS)


_THINK_OPEN = "<think>"


def _remove_think(text: str) -> str:
    """Strip <think>…</think> blocks (incl. a dangling, unclosed one). No trim,
    so callers that diff incrementally (streaming) keep stable offsets."""
    if not text or "<think" not in text.lower():
        return text
    text = _THINK_RE.sub("", text)
    low = text.lower()
    if "<think>" in low and "</think>" not in low:
        text = text[: low.find("<think>")]
    return text


def _strip_reasoning(text: str) -> str:
    """Reasoning-free, trimmed text — for the final stored response."""
    return _remove_think(text).strip()


class _ReasoningStreamFilter:
    """Feed raw streamed deltas; get back only the user-visible text, with
    reasoning removed even when ``<think>`` tags straddle chunk boundaries.

    It diffs the cleaned full text each step and holds back any trailing partial
    ``<think>`` opener so a half-formed tag is never shown, then flushed at end.
    """

    def __init__(self) -> None:
        self._full = ""
        self._emitted = 0

    def feed(self, delta: str) -> str:
        self._full += delta
        visible = _remove_think(self._full)
        safe = len(visible)
        for k in range(len(_THINK_OPEN), 0, -1):   # longest partial opener first
            if visible.endswith(_THINK_OPEN[:k]):
                safe = len(visible) - k
                break
        out = visible[self._emitted:safe]
        if safe > self._emitted:
            self._emitted = safe
        return out

    def flush(self) -> str:
        visible = _remove_think(self._full)
        out = visible[self._emitted:]
        self._emitted = len(visible)
        return out

    @property
    def text(self) -> str:
        return _strip_reasoning(self._full)


def _error_detail(resp: "httpx.Response") -> str:
    """Pull the human-readable reason out of an error response body.

    OpenAI-compatible hosts return ``{"error": {"message": "..."}}`` — far more
    useful than httpx's generic 'Client error 400' — so surface that verbatim.
    """
    try:
        body = resp.json()
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if isinstance(err, str):
            return err
        return json.dumps(body)[:500]
    except Exception:  # noqa: BLE001 — non-JSON body
        return (resp.text or "")[:500] or f"HTTP {resp.status_code}"


def _stop_reason(finish: str) -> str:
    return "tool_use" if finish == "tool_calls" else (
        "end_turn" if finish == "stop" else finish
    )


def _retry_after_seconds(resp: "httpx.Response") -> Optional[float]:
    """How long a 429 says to wait — from the ``Retry-After`` header or the
    ``"try again in 12.3s"`` hint in the body. ``None`` if it doesn't say."""
    header = getattr(resp, "headers", {}) or {}
    ra = header.get("retry-after") or header.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    match = _RETRY_AFTER_RE.search(_error_detail(resp))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


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
        # Whether the *host* can do vision at all (registry default). The actual
        # capability also depends on the pinned model — see ``supports_vision``.
        self._vision_host = supports_vision
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }

    @property
    def supports_vision(self) -> bool:
        """True only if the host allows vision *and* the pinned model is one.

        Before a model is pinned we assume the host default; once a concrete
        model is set we trust its id, so we never ship a screenshot to a
        text-only model (which most OpenAI-compatible hosts reject with a 400).
        """
        if not self._vision_host:
            return False
        if not self._model:
            return True
        return _looks_vision(self._model)

    # ── model discovery ───────────────────────────────────────────────

    def list_models(self) -> List[ModelInfo]:
        url = f"{self._base_url}/models"
        resp = httpx.get(url, headers=self._headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        out: List[ModelInfo] = []
        dropped: List[ModelInfo] = []
        for m in data:
            mid = m.get("id", "")
            if not mid:
                continue
            info = ModelInfo(
                id=mid,
                display_name=m.get("name", mid),
                created_at=str(m.get("created", "")),
                # The OpenAI /models payload rarely advertises modality, so infer
                # vision from the id and only when the host allows it at all.
                supports_vision=self._vision_host and _looks_vision(mid),
                supports_tools=True,
            )
            (dropped if _looks_non_chat(mid) else out).append(info)
        # If every model looked non-chat (unexpected), don't strand the caller
        # with nothing — fall back to the unfiltered list.
        return out or dropped

    # ── chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Run one assistant turn.

        If ``on_delta`` is given, the response is streamed (Server-Sent Events)
        and each visible text chunk is handed to ``on_delta`` as it arrives —
        used by the CLI to type the answer out instead of dumping it whole.
        """
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
        streaming = on_delta is not None
        if streaming:
            payload["stream"] = True

        for attempt in range(_MAX_RETRIES + 1):
            try:
                if streaming:
                    result, rate = self._attempt_stream(payload, on_delta)  # type: ignore[arg-type]
                else:
                    result, rate = self._attempt(payload)
            except httpx.RequestError as exc:
                # Surfaced to the user via the returned response text, so log at
                # debug to avoid duplicating it as a console warning.
                log.debug("OpenAI-compat chat network error: %s", exc)
                return LLMResponse(text=f"[network error: {exc}]", stop_reason="error")

            if result is not None:
                return result

            # Rate limited. Honour the server's suggested wait and retry once or
            # twice if it's short enough; kept at debug so it stays invisible in
            # normal output (the spinner just keeps counting) — see it under -v.
            wait, detail = rate
            if attempt < _MAX_RETRIES and wait is not None and wait <= _RETRY_CAP_SECONDS:
                log.debug("Rate limited; retrying in %.1fs (attempt %d/%d)",
                          wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait + 0.5)
                continue
            log.debug("OpenAI-compat chat 429: %s", detail)
            return LLMResponse(text=f"[provider error 429: {detail}]", stop_reason="error")

        return LLMResponse(text="[provider error: retries exhausted]", stop_reason="error")

    # ── one request attempt (non-stream / stream) ──────────────────────

    def _url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _attempt(self, payload: Dict[str, Any]):
        """Blocking request → ``(LLMResponse, None)`` or ``(None, (wait, detail))``
        when rate limited (retryable)."""
        resp = httpx.post(self._url(), headers=self._headers, json=payload,
                          timeout=_TIMEOUT)
        if resp.status_code == 429:
            return None, (_retry_after_seconds(resp), _error_detail(resp))
        if resp.status_code >= 400:
            detail = _error_detail(resp)
            log.debug("OpenAI-compat chat %s: %s", resp.status_code, detail)
            return LLMResponse(text=f"[provider error {resp.status_code}: {detail}]",
                               stop_reason="error"), None
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(text=f"[provider error: invalid JSON ({exc})]",
                               stop_reason="error"), None
        return self._parse_full(data), None

    def _attempt_stream(self, payload: Dict[str, Any],
                        on_delta: Callable[[str], None]):
        """Streaming request, same return contract as :meth:`_attempt`."""
        with httpx.stream("POST", self._url(), headers=self._headers,
                          json=payload, timeout=_TIMEOUT) as resp:
            if resp.status_code == 429:
                resp.read()
                return None, (_retry_after_seconds(resp), _error_detail(resp))
            if resp.status_code >= 400:
                resp.read()
                detail = _error_detail(resp)
                log.debug("OpenAI-compat chat %s: %s", resp.status_code, detail)
                return LLMResponse(text=f"[provider error {resp.status_code}: {detail}]",
                                   stop_reason="error"), None
            return self._parse_stream(resp, on_delta), None

    @staticmethod
    def _parse_full(data: Dict[str, Any]) -> LLMResponse:
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

        u = data.get("usage") or {}
        return LLMResponse(
            text=_strip_reasoning((message.get("content") or "").strip()),
            tool_calls=tool_calls,
            stop_reason=_stop_reason(finish),
            usage={
                "input_tokens": u.get("prompt_tokens", 0) or 0,
                "output_tokens": u.get("completion_tokens", 0) or 0,
            },
            raw=data,
        )

    @staticmethod
    def _parse_stream(resp: "httpx.Response",
                      on_delta: Callable[[str], None]) -> LLMResponse:
        rfilter = _ReasoningStreamFilter()
        tool_acc: Dict[int, Dict[str, str]] = {}
        finish = ""
        usage: Dict[str, Any] = {}

        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw[5:].strip() if raw.startswith("data:") else raw.strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    visible = rfilter.feed(content)
                    if visible:
                        on_delta(visible)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"] or finish

        tail = rfilter.flush()
        if tail:
            on_delta(tail)

        tool_calls: List[ToolCall] = []
        for idx in sorted(tool_acc):
            slot = tool_acc[idx]
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], input=args))

        u = usage or {}
        return LLMResponse(
            text=rfilter.text,
            tool_calls=tool_calls,
            stop_reason=_stop_reason(finish),
            usage={
                "input_tokens": u.get("prompt_tokens", 0) or 0,
                "output_tokens": u.get("completion_tokens", 0) or 0,
            },
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
