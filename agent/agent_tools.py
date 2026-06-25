"""Tools the QA agent can call, wired to the existing observation/input layer.

Each tool maps to infrastructure that already exists in the project:

- ``capture_screenshot`` / ``ocr_screen`` → :class:`ScreenObserver`
- ``read_game_state``                     → :class:`BridgeClient`
- ``press_key`` / ``hold_key`` / ``mouse_click`` / ``wait`` → :class:`InputExecutor`
- ``report_finding``                      → records the agent's QA verdict (terminal)

``dispatch`` returns a list of *neutral* content blocks (see
:mod:`agent.llm.base`) that the agentic loop feeds back as a ``tool_result``.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Finding:
    """The agent's conclusion about the goal under test."""

    verdict: str        # "PASS" | "FAIL" | "BUG" | "INCONCLUSIVE"
    summary: str
    details: str = ""


@dataclass
class ToolContext:
    """Everything the tools need to act on and observe the game."""

    observer: Any                       # ScreenObserver
    executor: Any                       # InputExecutor
    bridge: Optional[Any] = None        # BridgeClient | None
    persona: Optional[Any] = None       # Persona | None
    screenshot_dir: Path = field(default_factory=lambda: Path("reports/screenshots"))
    findings: List[Finding] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.screenshot_dir = Path(self.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


# ── tool schemas (neutral; each provider translates) ───────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "capture_screenshot",
        "description": "Capture the current game screen and return it as an image so you can see the game state. Use this to verify visual outcomes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_game_state",
        "description": "Read the latest structured game state JSON from the Unity bridge (player position, hp, animation, active UI, scene). Returns 'unavailable' if the bridge is not connected.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "press_key",
        "description": "Press (and optionally hold) a keyboard key, e.g. 'w', 'space', 'escape'. Use duration (seconds) to hold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name, e.g. w, a, s, d, space, escape"},
                "duration": {"type": "number", "description": "Hold time in seconds (0 = tap)"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "key_release",
        "description": "Release a key that is currently held down (pair with a held key_press).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "key_combo",
        "description": "Press several keys at once as a combo, e.g. ['ctrl','s'] or ['shift','w'].",
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keys"],
        },
    },
    {
        "name": "mouse_click",
        "description": "Click the mouse at absolute screen coordinates (x, y).",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "description": "left | right | middle"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "wait",
        "description": "Pause for a number of seconds to let the game react before observing again.",
        "input_schema": {
            "type": "object",
            "properties": {"duration": {"type": "number"}},
            "required": ["duration"],
        },
    },
    {
        "name": "ocr_screen",
        "description": "Run OCR on the current screen and return any on-screen text. Useful for menus and HUD text.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "plan_action",
        "description": "Log a short-term or long-term plan before taking action. Use this to think step-by-step, state your hypothesis, and decide what to do next to avoid looping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your analysis of the current state and hypothesis"},
                "plan": {"type": "string", "description": "The steps you intend to take"}
            },
            "required": ["thought", "plan"],
        },
    },
    {
        "name": "report_finding",
        "description": "Report your final QA conclusion for the goal and END the test. Call this exactly once when you have enough evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "description": "PASS | FAIL | BUG | INCONCLUSIVE"},
                "summary": {"type": "string", "description": "One-line conclusion"},
                "details": {"type": "string", "description": "Evidence and reasoning"},
            },
            "required": ["verdict", "summary"],
        },
    },
]

# Tools that conclude the run.
TERMINAL_TOOLS = {"report_finding"}


# ── dispatch ───────────────────────────────────────────────────────────

def dispatch(name: str, args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    """Execute a tool by name and return neutral content blocks for tool_result."""
    ctx.steps.append({"tool": name, "args": args})
    try:
        handler = _HANDLERS.get(name)
        if handler is None:
            return [_text(f"Unknown tool: {name}")]
        return handler(args, ctx)
    except Exception as exc:  # noqa: BLE001 — report to the model, keep the loop alive
        log.exception("Tool %s failed", name)
        return [_text(f"Tool '{name}' error: {exc}")]


def _h_capture_screenshot(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    img = ctx.observer.capture_screen()
    path = ctx.screenshot_dir / f"agent_{int(time.time() * 1000)}.png"
    img.save(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [_text("Current screen:"), {"type": "image", "data": b64}]


def _h_read_game_state(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    import json
    state = ctx.bridge.get_game_state() if ctx.bridge else None
    if not state:
        return [_text("game_state: unavailable (bridge not connected)")]
    return [_text("game_state:\n" + json.dumps(state, indent=2))]


def _h_press_key(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    key = str(args.get("key", ""))
    duration = float(args.get("duration", 0) or 0)
    if duration > 0:
        ctx.executor.hold_key(key, duration)
        msg = f"Held '{key}' for {duration}s."
    else:
        ctx.executor.press_key(key)
        msg = f"Pressed '{key}'."
    return [_text(msg)]


def _h_key_release(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    key = str(args.get("key", ""))
    ctx.executor.release_key(key)
    return [_text(f"Released '{key}'.")]


def _h_key_combo(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    keys = [str(k) for k in args.get("keys", [])]
    ctx.executor.press_combo(*keys)
    return [_text(f"Pressed combo: {'+'.join(keys)}.")]


def _h_mouse_click(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    x, y = int(args.get("x", 0)), int(args.get("y", 0))
    button = str(args.get("button", "left"))
    ctx.executor.mouse_click(x, y, button=button)
    return [_text(f"Clicked {button} at ({x}, {y}).")]


def _h_wait(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    duration = float(args.get("duration", 1.0) or 1.0)
    time.sleep(duration)
    return [_text(f"Waited {duration}s.")]


def _h_ocr_screen(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    text = ctx.observer.extract_text_from_screen()
    return [_text(f"On-screen text:\n{text or '(none detected)'}")]


def _h_report_finding(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    finding = Finding(
        verdict=str(args.get("verdict", "INCONCLUSIVE")).upper(),
        summary=str(args.get("summary", "")),
        details=str(args.get("details", "")),
    )
    ctx.findings.append(finding)
    return [_text(f"Recorded finding: {finding.verdict} - {finding.summary}")]


def _h_plan_action(args: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, Any]]:
    thought = args.get("thought", "")
    plan = args.get("plan", "")
    return [_text(f"Plan recorded. Proceed with your planned actions.")]


_HANDLERS = {
    "plan_action": _h_plan_action,
    "capture_screenshot": _h_capture_screenshot,
    "read_game_state": _h_read_game_state,
    "press_key": _h_press_key,
    "key_release": _h_key_release,
    "key_combo": _h_key_combo,
    "mouse_click": _h_mouse_click,
    "wait": _h_wait,
    "ocr_screen": _h_ocr_screen,
    "report_finding": _h_report_finding,
}


def _text(s: str) -> Dict[str, Any]:
    return {"type": "text", "text": s}
