"""The agentic core — an LLM that drives the game to satisfy a QA goal.

Unlike the scripted :class:`TestRunner`, the LLM here is the *brain* of the
loop: it observes (screenshot + game state), reasons, calls a tool to act, then
observes the result, repeating until it reports a finding. This is the
observe → reason → act → observe loop that turns the project from a test runner
into an actual agent.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.agent_tools import (
    TERMINAL_TOOLS,
    TOOL_SCHEMAS,
    Finding,
    ToolContext,
    dispatch,
)
from agent.llm.base import LLMProvider

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert QA tester for Unity games. You control the game like a real \
player: you can see the screen, read structured game state, and send keyboard \
and mouse input through tools.

Your job: investigate the goal below and determine whether the game behaves \
correctly. Work in small steps — act, then observe the result before acting \
again. Don't assume; verify with capture_screenshot and read_game_state.

When you have enough evidence, call report_finding exactly once with your \
verdict (PASS, FAIL, BUG, or INCONCLUSIVE) and stop. Be efficient — a handful \
of well-chosen actions beats dozens of random ones.

GOAL: {goal}
"""

_DEFAULT_MAX_STEPS = 15


@dataclass
class AgentRunResult:
    goal: str
    provider: str
    model: str
    findings: List[Finding] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    transcript: List[Dict[str, str]] = field(default_factory=list)
    stopped_reason: str = ""

    @property
    def verdict(self) -> str:
        return self.findings[-1].verdict if self.findings else "INCONCLUSIVE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "provider": self.provider,
            "model": self.model,
            "verdict": self.verdict,
            "stopped_reason": self.stopped_reason,
            "findings": [
                {"verdict": f.verdict, "summary": f.summary, "details": f.details}
                for f in self.findings
            ],
            "steps": self.steps,
        }


class QAAgent:
    """Drives a single QA goal to completion via the LLM tool loop."""

    def __init__(
        self,
        provider: LLMProvider,
        ctx: ToolContext,
        provider_name: str = "",
        max_steps: int = _DEFAULT_MAX_STEPS,
        on_event: Optional[Any] = None,   # callable(kind, payload) for live TUI
    ) -> None:
        self._provider = provider
        self._ctx = ctx
        self._provider_name = provider_name
        self._max_steps = max_steps
        self._on_event = on_event

    def run(self, goal: str) -> AgentRunResult:
        system = _SYSTEM_PROMPT.format(goal=goal)
        result = AgentRunResult(
            goal=goal, provider=self._provider_name, model=self._provider.model
        )

        # First observation grounds the agent in the current game state.
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": self._observation("Initial game state:")}
        ]

        for step in range(self._max_steps):
            self._emit("step", {"step": step + 1, "max": self._max_steps})
            resp = self._provider.chat(system, messages, TOOL_SCHEMAS)

            if resp.text:
                self._emit("thought", resp.text)
                result.transcript.append({"role": "assistant", "text": resp.text})

            if resp.stop_reason in ("error", "refusal"):
                result.stopped_reason = resp.stop_reason
                break

            # Reconstruct the assistant turn in neutral format.
            assistant_content: List[Dict[str, Any]] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            if not resp.tool_calls:
                # Model answered without acting — treat as done.
                result.stopped_reason = "no_tool_call"
                break

            # Execute every requested tool, gather results.
            tool_result_blocks: List[Dict[str, Any]] = []
            terminal = False
            acted = False
            for tc in resp.tool_calls:
                self._emit("action", {"name": tc.name, "input": tc.input})
                blocks = dispatch(tc.name, tc.input, self._ctx)
                self._emit("result", {"name": tc.name, "text": _result_summary(blocks)})
                tool_result_blocks.append({
                    "type": "tool_result", "tool_use_id": tc.id, "content": blocks,
                })
                if tc.name in TERMINAL_TOOLS:
                    terminal = True
                if tc.name in ("press_key", "hold_key", "mouse_click", "wait"):
                    acted = True

            messages.append({"role": "user", "content": tool_result_blocks})

            if terminal:
                result.stopped_reason = "report_finding"
                if self._ctx.findings:
                    f = self._ctx.findings[-1]
                    self._emit("finding", {
                        "verdict": f.verdict, "summary": f.summary, "details": f.details,
                    })
                break

            # Close the visual loop: after acting, hand the agent a fresh
            # observation so it sees the consequences without having to ask.
            if acted:
                self._emit("observation", {"note": "after actions"})
                messages.append({
                    "role": "user",
                    "content": self._observation("Observation after your actions:"),
                })
        else:
            result.stopped_reason = "max_steps"

        result.findings = list(self._ctx.findings)
        result.steps = list(self._ctx.steps)
        return result

    # ── helpers ────────────────────────────────────────────────────────

    def _observation(self, label: str) -> List[Dict[str, Any]]:
        """Build a user observation: a note, the game state JSON, and (if the
        provider supports vision) a screenshot."""
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": label}]

        state = self._ctx.bridge.get_game_state() if self._ctx.bridge else None
        if state:
            blocks.append({"type": "text", "text": "game_state:\n" + json.dumps(state)})
        else:
            blocks.append({"type": "text", "text": "game_state: unavailable (no bridge)"})

        if self._provider.supports_vision:
            try:
                img = self._ctx.observer.capture_screen()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                blocks.append({"type": "image", "data": b64})
            except Exception as exc:  # noqa: BLE001
                log.warning("Screenshot for observation failed: %s", exc)

        return blocks

    def _emit(self, kind: str, payload: str) -> None:
        if self._on_event:
            try:
                self._on_event(kind, payload)
            except Exception:  # noqa: BLE001
                pass


def _short(d: Dict[str, Any], n: int = 60) -> str:
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 2] + "..."


def _result_summary(blocks: List[Dict[str, Any]], n: int = 100) -> str:
    """One-line summary of a tool result for the live dashboard."""
    parts: List[str] = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[screenshot]")
    s = " ".join(parts).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 2] + "..."
