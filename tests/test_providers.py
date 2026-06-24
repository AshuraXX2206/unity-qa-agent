"""Neutral <-> wire translation for the provider adapters (pure logic, no network)."""

import json

from agent.llm.anthropic_provider import AnthropicProvider
from agent.llm.openai_compat_provider import OpenAICompatProvider
from agent.llm.registry import PROVIDERS, make_provider


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

    on = OpenAICompatProvider("k", "https://h/v1", model="m", supports_vision=True)
    parts = on._to_openai_messages(img)[0]["content"]
    assert any(x.get("type") == "image_url" for x in parts)

    off = OpenAICompatProvider("k", "https://h/v1", model="m", supports_vision=False)
    parts = off._to_openai_messages(img)[0]["content"]
    assert all(x["type"] == "text" for x in parts)
