"""The conversational brain of the CLI — a chat agent that *understands* the
user and decides what to do.

Where :class:`agent.qa_agent.QAAgent` is the agent that *plays the game*, the
:class:`Orchestrator` is the agent that *talks to you*. It turns the top-level
REPL from a rigid keyword dispatcher into a real assistant: you can ask it a
question in plain language and it answers, or tell it to do something and it
calls the right capability (run a suite, drive the game, analyse code, …) as a
tool — chaining several when a request needs it.

It reuses the same neutral message format and :class:`~agent.llm.base.LLMProvider`
interface as the game agent, so any configured provider works. The actual heavy
lifting (running suites, launching the game agent) is injected as *handlers* so
this module stays pure and unit-testable without a real LLM, Unity, or display.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from agent.llm.base import LLMProvider

log = logging.getLogger(__name__)

# A handler runs one tool and returns a short text result for the model to read.
Handler = Callable[[Dict[str, Any]], str]


def _accepts_on_delta(chat_fn: Any) -> bool:
    """Whether a provider's ``chat`` can stream (takes an ``on_delta`` arg)."""
    try:
        return "on_delta" in inspect.signature(chat_fn).parameters
    except (TypeError, ValueError):
        return False

_DEFAULT_MAX_STEPS = 8

_SYSTEM_PROMPT = """\
You are the Unity QA Agent assistant — a conversational AI that helps a developer \
test their Unity game. You can do two things:

1. ANSWER questions about the project, QA testing, how to use this tool, your \
capabilities, test suites, personas, configuration, and prior results — in plain \
language, using the conversation so far.
2. ACT on requests by calling tools. When the user wants something *done* — run a \
test suite, play and test the game against a goal, analyse the C# codebase, \
generate tests, validate a suite, or check status — call the matching tool. \
Chain tools when a request needs several steps (e.g. analyse the code, then \
generate tests from it).

Guidelines:
- Prefer doing over describing. If the user clearly asks you to test/run/analyse \
something, call the tool rather than explaining how they could.
- If you don't know a valid suite path, call list_test_suites first.
- For free-form, exploratory testing of behaviour ("can the player jump?", \
"does the menu open?"), use play_and_test_game with a clear goal. For executing \
a predefined scripted suite, use run_test_suite.
- If a request is ambiguous or missing something you truly need, ask a short \
clarifying question instead of guessing.
- After tools run, summarise the outcome for the user in a sentence or two — \
don't dump raw JSON.
- Be concise. This is a terminal; short, direct answers read best.
"""


# ── tool schemas (neutral; each provider translates) ───────────────────

ORCH_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "list_test_suites",
        "description": (
            "List the available YAML test suites in the project's test_cases/ "
            "directory, with each suite's name and test-case count. Call this "
            "before running or validating a suite when you don't know the path."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_test_suite",
        "description": (
            "Execute a scripted YAML test suite against the running Unity game "
            "and report pass/fail results. Provide the suite file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "suite": {
                    "type": "string",
                    "description": "Path to the YAML suite, e.g. test_cases/basic_movement.yaml",
                },
                "persona": {
                    "type": "string",
                    "description": "casual | speedrunner | explorer | griefer",
                },
                "case_id": {
                    "type": "string",
                    "description": "Optional: run only this test-case id (e.g. TC001)",
                },
                "safe_mode": {
                    "type": "boolean",
                    "description": "Log input actions without executing them",
                },
                "no_bridge": {
                    "type": "boolean",
                    "description": "Skip the Unity bridge (screen-capture only)",
                },
            },
            "required": ["suite"],
        },
    },
    {
        "name": "play_and_test_game",
        "description": (
            "Launch the autonomous QA agent to PLAY the Unity game and investigate "
            "a natural-language goal (e.g. 'check the player can jump'), returning a "
            "PASS / FAIL / BUG / INCONCLUSIVE verdict. Use this for exploratory, "
            "free-form behaviour testing rather than a predefined scripted suite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What to test, in plain language",
                },
                "persona": {
                    "type": "string",
                    "description": "casual | speedrunner | explorer | griefer",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Cap on the game agent's tool-calling steps",
                },
                "safe_mode": {
                    "type": "boolean",
                    "description": "Log input actions without executing them",
                },
                "no_bridge": {
                    "type": "boolean",
                    "description": "Skip the Unity bridge (vision/OCR only)",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "analyze_codebase",
        "description": (
            "Scan a Unity C# scripts directory and summarise classes, "
            "MonoBehaviours, detected input bindings and gameplay patterns "
            "(movement / health / combat / ui)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Assets/Scripts (defaults to the configured Unity project path)",
                },
            },
        },
    },
    {
        "name": "generate_tests",
        "description": (
            "Scan a Unity C# scripts directory and auto-generate a YAML test "
            "suite from detected patterns, writing it to disk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Assets/Scripts (defaults to the configured Unity project path)",
                },
                "output": {
                    "type": "string",
                    "description": "Output YAML path (default test_cases/auto_generated.yaml)",
                },
            },
        },
    },
    {
        "name": "validate_suite",
        "description": "Validate the YAML syntax and schema of a test suite file.",
        "input_schema": {
            "type": "object",
            "properties": {"suite": {"type": "string"}},
            "required": ["suite"],
        },
    },
    {
        "name": "get_status",
        "description": (
            "Report current agent status: AI provider, model, API-key presence, "
            "Unity bridge connection, configured paths, and number of test suites."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


class Orchestrator:
    """A conversational agent that answers questions and executes tasks via tools.

    Memory persists across :meth:`ask` calls, so the REPL gets a real
    multi-turn conversation. Tool execution is delegated to the ``handlers``
    mapping (``name -> callable(args) -> str``) injected at construction.
    """

    def __init__(
        self,
        provider: LLMProvider,
        handlers: Dict[str, Handler],
        provider_name: str = "",
        max_steps: int = _DEFAULT_MAX_STEPS,
        on_event: Optional[Callable[[str, Any], None]] = None,
        history_limit: int = 40,
    ) -> None:
        self._provider = provider
        self._handlers = handlers
        self._provider_name = provider_name
        self._max_steps = max_steps
        self._on_event = on_event
        self._history_limit = history_limit
        self._messages: List[Dict[str, Any]] = []
        # Stream token-by-token only when we have somewhere to show it AND the
        # provider supports it; otherwise fall back to a single blocking call.
        self._stream = on_event is not None and _accepts_on_delta(provider.chat)

    # ── public API ──────────────────────────────────────────────────────

    def ask(self, user_text: str) -> str:
        """Run one user turn to completion and return the assistant's reply.

        The loop lets the model call any number of tools before it produces a
        plain-text answer (or a clarifying question); that text is returned and
        the turn ends, ready for the next user input.
        """
        self._messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        )
        self._emit("turn_start", None)
        try:
            return self._run_turn()
        finally:
            self._emit("turn_end", None)

    def _run_turn(self) -> str:
        last_text = ""
        for _ in range(self._max_steps):
            self._emit("thinking", None)
            try:
                resp = self._chat()
            except Exception as exc:  # noqa: BLE001 — surface, don't crash the REPL
                log.exception("Orchestrator chat failed")
                return f"(LLM call failed: {exc})"

            if resp.usage:
                self._emit("usage", resp.usage)
            if resp.text:
                last_text = resp.text
            # Tell the view this step's message is complete and whether it was a
            # final answer (rendered as the reply) or narration before a tool
            # call. Streaming already delivered the text via "delta" events; this
            # just lets the view settle it into the right surface.
            final = (not resp.tool_calls) or resp.stop_reason in ("error", "refusal")
            self._emit("message", {"final": final, "text": resp.text})

            # Record the assistant turn in neutral format. Skip an empty turn:
            # some providers (e.g. Anthropic) reject an assistant message with
            # empty content, which would break the *next* ask() on this history.
            assistant_content: List[Dict[str, Any]] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
            if assistant_content:
                self._messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

            if resp.stop_reason in ("error", "refusal"):
                self._trim()
                return last_text or "(the model could not complete the request)"

            if not resp.tool_calls:
                # No action requested — this text is the final answer / question.
                self._trim()
                return resp.text or "(the model returned an empty response)"

            # Execute every requested tool and feed results back.
            tool_result_blocks: List[Dict[str, Any]] = []
            for tc in resp.tool_calls:
                self._emit("action", {"name": tc.name, "input": tc.input})
                result = self._dispatch(tc.name, tc.input)
                self._emit("result", {"name": tc.name, "text": result})
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": [{"type": "text", "text": result}],
                    }
                )
            self._messages.append({"role": "user", "content": tool_result_blocks})

        self._trim()
        return last_text or "(I couldn't finish within the step limit.)"

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._messages = []

    # ── internals ───────────────────────────────────────────────────────

    def _chat(self):
        """One provider call, streaming visible text as 'delta' events if we can."""
        if self._stream:
            return self._provider.chat(
                _SYSTEM_PROMPT, self._messages, ORCH_TOOL_SCHEMAS,
                on_delta=lambda chunk: self._emit("delta", chunk),
            )
        return self._provider.chat(_SYSTEM_PROMPT, self._messages, ORCH_TOOL_SCHEMAS)

    def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        try:
            out = handler(args or {})
            return out if isinstance(out, str) else json.dumps(out)
        except Exception as exc:  # noqa: BLE001 — report to the model, keep going
            log.exception("Orchestrator tool %s failed", name)
            return f"Tool '{name}' error: {exc}"

    def _emit(self, kind: str, payload: Any) -> None:
        if self._on_event:
            try:
                self._on_event(kind, payload)
            except Exception:  # noqa: BLE001
                pass

    def _trim(self) -> None:
        """Bound conversation growth without splitting a tool_use/tool_result pair.

        We only ever cut at the start of a *real* user turn (a message whose
        first block is plain text), never between an assistant tool call and the
        user message carrying its tool results — so providers that require that
        pairing (e.g. Anthropic) stay happy.
        """
        if len(self._messages) <= self._history_limit:
            return
        drop = len(self._messages) - self._history_limit
        i = drop
        while i < len(self._messages) and not self._is_user_text_turn(self._messages[i]):
            i += 1
        if i < len(self._messages):
            self._messages = self._messages[i:]

    @staticmethod
    def _is_user_text_turn(msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content") or []
        return bool(content) and content[0].get("type") == "text"
