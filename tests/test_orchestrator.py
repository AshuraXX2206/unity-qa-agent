"""The conversational orchestrator loop, driven by a scripted provider.

These exercise the REPL "brain" without a real LLM, Unity, or display: a plain
Q&A turn (no tool), a tool dispatch that feeds a result back and produces a final
answer, multi-turn memory, and graceful handling of erroring/unknown tools.
"""

from typing import Any, Dict, List

from agent.llm.base import LLMResponse, ToolCall
from agent.orchestrator import ORCH_TOOL_SCHEMAS, Orchestrator
from tests._fakes import ScriptedProvider, tool_use


def _text_turn(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end_turn",
                       usage={"input_tokens": 7, "output_tokens": 3})


def _recording_handlers() -> tuple[Dict[str, Any], List[tuple]]:
    """A handler set that records calls and returns canned strings."""
    calls: List[tuple] = []

    def make(name: str, ret: str):
        def h(args: Dict[str, Any]) -> str:
            calls.append((name, dict(args)))
            return ret
        return h

    handlers = {
        "list_test_suites": make("list_test_suites", "Available test suites:\n  - a.yaml"),
        "run_test_suite": make("run_test_suite", "Ran suite 'a.yaml': 2 passed, 0 failed."),
        "get_status": make("get_status", "Provider: gemini, suites: 1."),
        "boom": lambda args: (_ for _ in ()).throw(RuntimeError("kaboom")),
    }
    return handlers, calls


def test_plain_question_answers_without_tools():
    handlers, calls = _recording_handlers()
    provider = ScriptedProvider([_text_turn("I test Unity games by playing them.")])
    orch = Orchestrator(provider, handlers, provider_name="scripted")

    answer = orch.ask("what can you do?")

    assert answer == "I test Unity games by playing them."
    assert calls == []  # answered from knowledge, no action taken


def test_tool_call_then_final_answer():
    handlers, calls = _recording_handlers()
    turns = [
        tool_use("t1", "list_test_suites", {}, text="Let me check."),
        _text_turn("You have one suite: a.yaml."),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers, provider_name="scripted")

    answer = orch.ask("what suites do I have?")

    assert [c[0] for c in calls] == ["list_test_suites"]
    assert answer == "You have one suite: a.yaml."


def test_tool_args_are_passed_through():
    handlers, calls = _recording_handlers()
    turns = [
        tool_use("t1", "run_test_suite", {"suite": "a.yaml", "persona": "griefer"}),
        _text_turn("Done — 2 passed."),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers, provider_name="scripted")

    orch.ask("run a.yaml as a griefer")

    assert calls == [("run_test_suite", {"suite": "a.yaml", "persona": "griefer"})]


def test_unknown_tool_is_reported_not_crashed():
    handlers, _ = _recording_handlers()
    turns = [
        tool_use("t1", "nonexistent_tool", {}),
        _text_turn("Sorry, I can't do that."),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers, provider_name="scripted")
    # Should not raise; the loop feeds the error back and continues to a reply.
    assert orch.ask("do something weird") == "Sorry, I can't do that."


def test_erroring_tool_keeps_loop_alive():
    handlers, _ = _recording_handlers()
    turns = [
        tool_use("t1", "boom", {}),
        _text_turn("That tool failed, here's what I know instead."),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers, provider_name="scripted")
    answer = orch.ask("trigger the broken tool")
    assert "what I know" in answer


def test_conversation_memory_persists_across_turns():
    captured: List[List[Dict[str, Any]]] = []

    class Spy(ScriptedProvider):
        def chat(self, system, messages, tools):
            # snapshot the running history the provider sees each call
            captured.append([m for m in messages])
            return super().chat(system, messages, tools)

    handlers, _ = _recording_handlers()
    turns = [_text_turn("First answer."), _text_turn("Second answer, recalling the first.")]
    orch = Orchestrator(Spy(turns), handlers, provider_name="scripted")

    orch.ask("first question")
    orch.ask("second question")

    # The second turn must include the first exchange in its history.
    last_history = captured[-1]
    texts = [
        b.get("text", "")
        for m in last_history
        for b in m.get("content", [])
        if isinstance(b, dict)
    ]
    assert any("first question" in t for t in texts)
    assert any("First answer." in t for t in texts)
    assert any("second question" in t for t in texts)


def test_max_steps_guard_returns_something():
    handlers, _ = _recording_handlers()
    # Always asks for a tool, never gives a final text → must stop at max_steps.
    turns = [tool_use("t", "get_status", {})]
    orch = Orchestrator(ScriptedProvider(turns), handlers,
                        provider_name="scripted", max_steps=3)
    answer = orch.ask("loop forever")
    assert isinstance(answer, str) and answer  # non-empty, no crash


def test_reset_clears_memory():
    captured: List[int] = []

    class Spy(ScriptedProvider):
        def chat(self, system, messages, tools):
            captured.append(len(messages))
            return super().chat(system, messages, tools)

    handlers, _ = _recording_handlers()
    orch = Orchestrator(Spy([_text_turn("a"), _text_turn("b")]), handlers)

    orch.ask("one")
    orch.reset()
    orch.ask("two")

    # After reset the second turn starts a fresh history (just the new user msg).
    assert captured[-1] == 1


def test_history_trim_stays_on_user_text_boundary():
    handlers, _ = _recording_handlers()
    # Tiny limit forces trimming; a tool round-trip must not be split apart.
    turns = [
        tool_use("t1", "get_status", {}),
        _text_turn("ok 1"),
        _text_turn("ok 2"),
        _text_turn("ok 3"),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers, history_limit=2)
    orch.ask("a")
    orch.ask("b")
    orch.ask("c")

    msgs = orch._messages
    # Whatever we kept, the first retained message is a real user-text turn,
    # never a dangling tool_result (which would break provider pairing rules).
    assert msgs, "history should not be empty"
    first = msgs[0]
    assert first["role"] == "user"
    assert first["content"][0]["type"] == "text"


def test_empty_response_does_not_poison_history():
    # A turn with no text and no tool calls must NOT leave an empty assistant
    # message behind (Anthropic rejects empty content on the following turn).
    handlers, _ = _recording_handlers()
    empty = LLMResponse(text="", tool_calls=[], stop_reason="end_turn")
    orch = Orchestrator(ScriptedProvider([empty]), handlers)

    answer = orch.ask("say nothing")

    assert answer  # a non-empty fallback, never ""
    roles_with_empty = [
        m for m in orch._messages
        if m["role"] == "assistant" and not m["content"]
    ]
    assert roles_with_empty == []


def test_turn_lifecycle_and_message_events():
    events = []
    handlers, _ = _recording_handlers()

    # Turn: narration + a tool call, then a final answer.
    turns = [
        tool_use("t1", "list_test_suites", {}, text="Let me look that up."),
        _text_turn("You have one suite."),
    ]
    orch = Orchestrator(ScriptedProvider(turns), handlers,
                        on_event=lambda k, p: events.append((k, p)))
    orch.ask("what suites?")

    kinds = [k for k, _ in events]
    assert kinds[0] == "turn_start"
    assert kinds[-1] == "turn_end"
    assert "action" in kinds

    messages = [p for k, p in events if k == "message"]
    # First message is narration before the tool (final=False); last is the reply.
    assert messages[0] == {"final": False, "text": "Let me look that up."}
    assert messages[-1] == {"final": True, "text": "You have one suite."}


def test_plain_answer_is_one_final_message():
    events = []
    handlers, _ = _recording_handlers()
    orch = Orchestrator(ScriptedProvider([_text_turn("Hi there!")]), handlers,
                        on_event=lambda k, p: events.append((k, p)))
    answer = orch.ask("hi")
    assert answer == "Hi there!"
    messages = [p for k, p in events if k == "message"]
    assert messages == [{"final": True, "text": "Hi there!"}]
    assert not any(k == "action" for k, _ in events)


def test_schemas_have_required_shape():
    for t in ORCH_TOOL_SCHEMAS:
        assert {"name", "description", "input_schema"} <= set(t)
        assert t["input_schema"]["type"] == "object"
