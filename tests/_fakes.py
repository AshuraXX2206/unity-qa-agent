"""Shared lightweight fakes so tests run without Unity, a display, or real LLMs."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.llm.base import LLMProvider, LLMResponse, ToolCall


class FakeImage:
    def save(self, *a: Any, **k: Any) -> None:
        pass


class FakeObserver:
    def capture_screen(self) -> FakeImage:
        return FakeImage()

    def extract_text_from_screen(self) -> str:
        return "HP 100"


class FakeExecutor:
    """Records every input call for assertions."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def press_key(self, key: str) -> None:
        self.calls.append(("press", key))

    def hold_key(self, key: str, duration: float) -> None:
        self.calls.append(("hold", key, duration))

    def release_key(self, key: str) -> None:
        self.calls.append(("release", key))

    def press_combo(self, *keys: str) -> None:
        self.calls.append(("combo", *keys))

    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("click", x, y, button))


class FakeBridge:
    def __init__(self, state: Dict[str, Any] | None = None) -> None:
        self._state = state or {"player": {"position": {"z": 1.5}}, "scene": "L1"}

    def get_game_state(self) -> Dict[str, Any]:
        return self._state


class ScriptedProvider(LLMProvider):
    """Replays a fixed list of LLMResponse turns; ignores actual messages."""

    supports_vision = False

    def __init__(self, turns: List[LLMResponse], model: str = "scripted-1") -> None:
        super().__init__("key", model)
        self._turns = turns
        self._i = 0

    def list_models(self):  # pragma: no cover - not used here
        return []

    def chat(self, system: str, messages: List[Dict[str, Any]],
             tools: List[Dict[str, Any]]) -> LLMResponse:
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return turn


def tool_use(turn_id: str, name: str, args: Dict[str, Any], text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[ToolCall(turn_id, name, args)],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
