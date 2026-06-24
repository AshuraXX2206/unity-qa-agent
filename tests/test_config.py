"""Agentic-mode config properties."""

from agent.config import Config


def test_defaults():
    c = Config()
    # provider registry defaults
    assert c.provider_name in c.data["provider"] or c.provider_name == c.data["provider"]
    assert c.agent_model == "auto"


def test_set_and_read_agent_key():
    c = Config()
    c["provider"] = "groq"
    c.set_agent_api_key("groq", "KEY123")
    assert c.provider_name == "groq"
    assert c.agent_api_key == "KEY123"


def test_pinning_model():
    c = Config()
    c.agent_model = "some-model-id"
    assert c.agent_model == "some-model-id"


def test_base_url_default_empty():
    c = Config()
    assert c.base_url == ""
