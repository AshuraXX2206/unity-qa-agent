"""Tool dispatch maps to the executor/observer and report_finding is terminal."""

from pathlib import Path

from agent.agent_tools import (
    TERMINAL_TOOLS,
    TOOL_SCHEMAS,
    ToolContext,
    dispatch,
)
from tests._fakes import FakeBridge, FakeExecutor, FakeObserver


def _ctx(tmp):
    return ToolContext(
        observer=FakeObserver(), executor=FakeExecutor(),
        bridge=FakeBridge(), screenshot_dir=Path(tmp),
    )


def test_schema_and_terminal():
    names = {t["name"] for t in TOOL_SCHEMAS}
    for n in ("capture_screenshot", "read_game_state", "press_key", "key_release",
              "key_combo", "mouse_click", "wait", "ocr_screen", "report_finding"):
        assert n in names
    assert "report_finding" in TERMINAL_TOOLS


def test_press_and_combo(tmp_path):
    ctx = _ctx(tmp_path)
    dispatch("press_key", {"key": "w", "duration": 1.0}, ctx)
    dispatch("key_combo", {"keys": ["shift", "w"]}, ctx)
    dispatch("key_release", {"key": "w"}, ctx)
    assert ("hold", "w", 1.0) in ctx.executor.calls
    assert ("combo", "shift", "w") in ctx.executor.calls
    assert ("release", "w") in ctx.executor.calls


def test_read_game_state(tmp_path):
    ctx = _ctx(tmp_path)
    blocks = dispatch("read_game_state", {}, ctx)
    assert any("position" in b.get("text", "") for b in blocks)


def test_capture_returns_image(tmp_path):
    ctx = _ctx(tmp_path)
    blocks = dispatch("capture_screenshot", {}, ctx)
    assert any(b["type"] == "image" for b in blocks)


def test_report_finding_records(tmp_path):
    ctx = _ctx(tmp_path)
    dispatch("report_finding", {"verdict": "pass", "summary": "ok"}, ctx)
    assert len(ctx.findings) == 1
    assert ctx.findings[0].verdict == "PASS"


def test_unknown_tool_is_safe(tmp_path):
    ctx = _ctx(tmp_path)
    blocks = dispatch("nope", {}, ctx)
    assert "Unknown tool" in blocks[0]["text"]
