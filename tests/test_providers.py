"""Neutral <-> wire translation for the provider adapters (pure logic, no network)."""

import json

import httpx

from agent.llm import openai_compat_provider as oc
from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.openai_compat_provider import (
    OpenAICompatProvider,
    _error_detail,
    _looks_non_chat,
    _looks_vision,
    _strip_reasoning,
)
from agent.llm.registry import PROVIDERS, make_provider


class _FakeResp:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeStream:
    """Stand-in for the context manager returned by httpx.stream(...)."""

    def __init__(self, status_code=200, lines=None, payload=None, text="", headers=None):
        self.status_code = status_code
        self._lines = lines or []
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        for ln in self._lines:
            yield ln

    def read(self):
        return None

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_registry_has_free_tier_providers():
    for name in ("gemini", "groq", "openrouter", "cerebras", "mistral", "together"):
        assert name in PROVIDERS
    p = make_provider("groq", api_key="x", model="m")
    assert type(p).__name__ == "OpenAICompatProvider"
    assert p.model == "m"


def test_make_provider_unknown_requires_base_url():
    try:
        make_provider("totally-unknown", api_key="x", model="m")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without base_url")
    # but with a base_url it builds a generic openai-compat client
    p = make_provider("totally-unknown", api_key="x", model="m", base_url="https://h/v1")
    assert type(p).__name__ == "OpenAICompatProvider"


def test_anthropic_neutral_to_blocks():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image", "data": "B64"},
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "ok"}]},
        ],
    }
    out = AnthropicProvider._to_anthropic_message(msg)
    types = [b["type"] for b in out["content"]]
    assert types == ["text", "image", "tool_result"]
    assert out["content"][1]["source"]["type"] == "base64"
    assert out["content"][2]["tool_use_id"] == "t1"


def test_openai_assistant_tool_use_and_result():
    p = OpenAICompatProvider("k", "https://h/v1", model="m", supports_vision=True)

    asst = {"role": "assistant", "content": [
        {"type": "text", "text": "go"},
        {"type": "tool_use", "id": "c1", "name": "press_key", "input": {"key": "w"}},
    ]}
    out = p._to_openai_messages(asst)
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["function"]["name"] == "press_key"
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"key": "w"}

    usr = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1",
         "content": [{"type": "text", "text": "done"}]},
    ]}
    out = p._to_openai_messages(usr)
    assert out[0]["role"] == "tool" and out[0]["tool_call_id"] == "c1"


def test_openai_vision_toggle():
    img = {"role": "user", "content": [
        {"type": "text", "text": "see"}, {"type": "image", "data": "B64"}]}

    # Host allows vision AND the pinned model is a vision model → image passes.
    on = OpenAICompatProvider(
        "k", "https://h/v1", model="meta-llama/llama-4-scout", supports_vision=True)
    parts = on._to_openai_messages(img)[0]["content"]
    assert any(x.get("type") == "image_url" for x in parts)

    # Host allows vision but the pinned model is text-only → image is stripped,
    # so we never 400 by shipping a screenshot to a text model.
    text_model = OpenAICompatProvider(
        "k", "https://h/v1", model="qwen/qwen3-32b", supports_vision=True)
    parts = text_model._to_openai_messages(img)[0]["content"]
    assert all(x["type"] == "text" for x in parts)

    # Host can't do vision at all → image stripped regardless of model.
    off = OpenAICompatProvider(
        "k", "https://h/v1", model="meta-llama/llama-4-scout", supports_vision=False)
    parts = off._to_openai_messages(img)[0]["content"]
    assert all(x["type"] == "text" for x in parts)


def test_strip_reasoning():
    assert _strip_reasoning("<think>plan</think>Hello") == "Hello"
    assert _strip_reasoning("no tags here") == "no tags here"
    # a dangling, unclosed think block (truncated output) is dropped entirely
    assert _strip_reasoning("answer first <think>still musing") == "answer first"
    assert _strip_reasoning("") == ""


def test_non_chat_and_vision_hints():
    for bad in ("whisper-large-v3", "meta-llama/llama-prompt-guard-2-86m",
                "playai-tts", "canopylabs/orpheus-v1-english", "groq/compound"):
        assert _looks_non_chat(bad), bad
    for good in ("llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b"):
        assert not _looks_non_chat(good), good

    assert _looks_vision("meta-llama/llama-4-scout-17b-16e-instruct")
    assert _looks_vision("pixtral-12b")
    assert not _looks_vision("qwen/qwen3-32b")
    assert not _looks_vision("llama-3.3-70b-versatile")


def test_list_models_filters_non_chat_and_flags_vision(monkeypatch):
    payload = {"data": [
        {"id": "llama-3.3-70b-versatile", "created": 100},
        {"id": "meta-llama/llama-4-scout-17b", "created": 200},
        {"id": "whisper-large-v3", "created": 300},
        {"id": "meta-llama/llama-prompt-guard-2-86m", "created": 400},
        {"id": "playai-tts", "created": 500},
    ]}
    monkeypatch.setattr(oc.httpx, "get",
                        lambda *a, **k: _FakeResp(200, payload))
    p = OpenAICompatProvider("k", "https://h/v1", supports_vision=True)
    models = p.list_models()
    ids = {m.id for m in models}
    assert ids == {"llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b"}
    vision = {m.id: m.supports_vision for m in models}
    assert vision["meta-llama/llama-4-scout-17b"] is True
    assert vision["llama-3.3-70b-versatile"] is False


def test_chat_surfaces_error_body(monkeypatch):
    body = {"error": {"message": "The model `x` has been decommissioned"}}
    monkeypatch.setattr(oc.httpx, "post",
                        lambda *a, **k: _FakeResp(400, body))
    p = OpenAICompatProvider("k", "https://h/v1", model="x")
    r = p.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [])
    assert r.stop_reason == "error"
    assert "decommissioned" in r.text
    assert "400" in r.text


def test_chat_strips_reasoning_from_content(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "<think>hmm</think>pong"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    monkeypatch.setattr(oc.httpx, "post",
                        lambda *a, **k: _FakeResp(200, payload))
    p = OpenAICompatProvider("k", "https://h/v1", model="qwen/qwen3-32b")
    r = p.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "ping"}]}], [])
    assert r.text == "pong"
    assert r.stop_reason == "end_turn"


def test_error_detail_falls_back_to_text():
    assert "boom" in _error_detail(_FakeResp(500, None, text="boom"))


def test_reasoning_stream_filter_handles_split_tags():
    f = oc._ReasoningStreamFilter()
    out = "".join(f.feed(d) for d in
                  ["<th", "ink>secret reasoning", "</thi", "nk>vis", "ible answer"])
    out += f.flush()
    assert out == "visible answer"     # <think>…</think> never leaked
    assert f.text == "visible answer"


def test_chat_streams_content_and_strips_reasoning(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"<think>plan"}}]}',
        'data: {"choices":[{"delta":{"content":"</think>Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo!"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(oc.httpx, "stream", lambda *a, **k: _FakeStream(200, lines))
    chunks = []
    p = OpenAICompatProvider("k", "https://h/v1", model="qwen/qwen3-32b")
    r = p.chat("s", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
               [], on_delta=chunks.append)
    assert "".join(chunks) == "Hello!"   # streamed progressively, no <think>
    assert r.text == "Hello!"
    assert r.stop_reason == "end_turn"


def test_chat_streams_tool_calls(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        '"function":{"name":"list_test_suites","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"{}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(oc.httpx, "stream", lambda *a, **k: _FakeStream(200, lines))
    p = OpenAICompatProvider("k", "https://h/v1", model="m")
    r = p.chat("s", [{"role": "user", "content": [{"type": "text", "text": "go"}]}],
               [], on_delta=lambda c: None)
    assert r.stop_reason == "tool_use"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "list_test_suites"
    assert r.tool_calls[0].input == {}


def test_chat_stream_surfaces_error_body(monkeypatch):
    monkeypatch.setattr(oc.httpx, "stream",
                        lambda *a, **k: _FakeStream(400, payload={"error": {"message": "bad model"}}))
    p = OpenAICompatProvider("k", "https://h/v1", model="x")
    r = p.chat("s", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
               [], on_delta=lambda c: None)
    assert r.stop_reason == "error"
    assert "bad model" in r.text


def test_chat_retries_on_429_then_succeeds(monkeypatch):
    body_429 = {"error": {"message": "Rate limit reached. Please try again in 0.2s"}}
    ok = {"choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
          "usage": {}}
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _FakeResp(429, body_429) if calls["n"] == 1 else _FakeResp(200, ok)

    slept = {"secs": 0.0}
    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda s: slept.__setitem__("secs", s))

    p = OpenAICompatProvider("k", "https://h/v1", model="qwen/qwen3-32b")
    r = p.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "ping"}]}], [])

    assert r.text == "pong" and r.stop_reason == "end_turn"
    assert calls["n"] == 2          # retried exactly once
    assert slept["secs"] > 0        # honoured the suggested wait


def test_chat_gives_up_on_429_when_wait_too_long(monkeypatch):
    body = {"error": {"message": "Rate limit reached. Please try again in 999s"}}
    monkeypatch.setattr(oc.httpx, "post", lambda *a, **k: _FakeResp(429, body))
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    p = OpenAICompatProvider("k", "https://h/v1", model="x")
    r = p.chat("sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [])
    assert r.stop_reason == "error" and "429" in r.text
