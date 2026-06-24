"""model_selector picks the newest suitable model and degrades gracefully."""

from agent.llm.base import LLMProvider, ModelInfo
from agent.model_selector import select_newest_model


class _ListProvider(LLMProvider):
    def __init__(self, models):
        super().__init__("k")
        self._models = models

    def list_models(self):
        return self._models

    def chat(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


def test_prefers_newest_vision_and_tools():
    p = _ListProvider([
        ModelInfo(id="old", created_at="2024-01-01", supports_vision=False, supports_tools=True),
        ModelInfo(id="mid", created_at="2025-06-01", supports_vision=True, supports_tools=True),
        ModelInfo(id="new", created_at="2026-06-01", supports_vision=True, supports_tools=True),
    ])
    assert select_newest_model(p, prefer_vision=True) == "new"


def test_falls_back_to_tools_only_when_no_vision():
    p = _ListProvider([
        ModelInfo(id="a", created_at="2025-01-01", supports_vision=False, supports_tools=True),
        ModelInfo(id="b", created_at="2026-01-01", supports_vision=False, supports_tools=True),
    ])
    assert select_newest_model(p, prefer_vision=True) == "b"


def test_error_uses_fallback():
    class Broken(LLMProvider):
        def list_models(self):
            raise RuntimeError("boom")

        def chat(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

    assert select_newest_model(Broken("k"), fallback="fb") == "fb"


def test_empty_uses_fallback():
    assert select_newest_model(_ListProvider([]), fallback="fb") == "fb"
