"""Full agentic loop with a scripted provider and fakes — no Unity, no real LLM."""

from pathlib import Path

from agent.agent_tools import ToolContext
from agent.llm.base import LLMResponse, ToolCall
from agent.qa_agent import QAAgent
from agent.persona import get_persona
from tests._fakes import FakeBridge, FakeExecutor, FakeObserver, ScriptedProvider, tool_use


def _run(turns, tmp_path, persona=None, on_event=None, max_steps=10):
    exe = FakeExecutor()
    ctx = ToolContext(
        observer=FakeObserver(), executor=exe, bridge=FakeBridge(),
        persona=persona, screenshot_dir=Path(tmp_path),
    )
    agent = QAAgent(ScriptedProvider(turns), ctx, provider_name="scripted",
                    max_steps=max_steps, on_event=on_event)
    return agent.run("check W moves the player forward"), exe


def test_loop_acts_then_reports(tmp_path):
    turns = [
        tool_use("a", "press_key", {"key": "w", "duration": 1.0}, text="press W"),
        tool_use("b", "read_game_state", {}),
        tool_use("c", "report_finding", {"verdict": "pass", "summary": "moves forward"}),
    ]
    result, exe = _run(turns, tmp_path)
    assert result.verdict == "PASS"
    assert result.stopped_reason == "report_finding"
    assert ("hold", "w", 1.0) in exe.calls
    assert result.total_tokens > 0  # usage accumulated


def test_no_tool_call_ends(tmp_path):
    turns = [LLMResponse(text="I think it's fine.", stop_reason="end_turn")]
    result, _ = _run(turns, tmp_path)
    assert result.stopped_reason == "no_tool_call"


def test_max_steps_guard(tmp_path):
    # never reports — should stop at max_steps
    turns = [tool_use("x", "wait", {"duration": 0})]
    result, _ = _run(turns, tmp_path, max_steps=3)
    assert result.stopped_reason == "max_steps"


def test_events_emitted(tmp_path):
    events = []
    turns = [
        tool_use("a", "press_key", {"key": "w"}),
        tool_use("c", "report_finding", {"verdict": "fail", "summary": "stuck"}),
    ]
    result, _ = _run(turns, tmp_path, on_event=lambda k, d: events.append(k))
    kinds = set(events)
    for need in ("step", "action", "result", "finding", "usage"):
        assert need in kinds, (need, kinds)
    assert result.verdict == "FAIL"


def test_persona_in_system_prompt(tmp_path):
    captured = {}

    class Spy(ScriptedProvider):
        def chat(self, system, messages, tools):
            captured["system"] = system
            return super().chat(system, messages, tools)

    exe = FakeExecutor()
    ctx = ToolContext(observer=FakeObserver(), executor=exe, bridge=FakeBridge(),
                      persona=get_persona("griefer"), screenshot_dir=Path(tmp_path))
    turns = [tool_use("c", "report_finding", {"verdict": "bug", "summary": "x"})]
    QAAgent(Spy(turns), ctx, provider_name="scripted").run("break it")
    assert "griefer" in captured["system"].lower()
