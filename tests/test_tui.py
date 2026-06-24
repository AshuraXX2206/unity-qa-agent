"""AgentDashboard renders header/feed/stats/footer and stays ASCII-safe."""

import io

from rich.console import Console

from agent.tui import AgentDashboard, print_banner


def _render(dash) -> str:
    cap = Console(file=io.StringIO(), width=100, record=True)
    dash._console = cap
    cap.print(dash)
    return cap.export_text()


def test_dashboard_renders_live_data():
    dash = AgentDashboard("check W moves forward", "scripted", "scripted-1", 8)
    for k, d in [
        ("step", {"step": 1}),
        ("thought", "I'll press W."),
        ("action", {"name": "press_key", "input": {"key": "w"}}),
        ("result", {"name": "press_key", "text": "Held 'w'."}),
        ("usage", {"input": 100, "output": 40}),
        ("finding", {"verdict": "pass", "summary": "moves forward"}),
    ]:
        dash.handle(k, d)

    out = _render(dash)
    assert "check W moves forward" in out
    assert "scripted-1" in out
    assert "press_key" in out
    assert "PASS" in out
    assert "Run Stats" in out and "Activity" in out
    assert "140" in out  # token total shown


def test_dashboard_is_ascii_safe():
    # The content WE author must encode on cp1258 (Rich downgrades its own box
    # borders to ASCII on legacy consoles; that's not our concern here).
    from agent.tui import _EVENT_STYLE

    for icon, _style in _EVENT_STYLE.values():
        icon.encode("cp1258")

    dash = AgentDashboard("goal", "prov", "model", 5)
    for k, d in [
        ("thought", "pressing W and waiting"),
        ("action", {"name": "press_key", "input": {"key": "w"}}),
        ("result", {"name": "press_key", "text": "Held 'w' for 1.0s."}),
        ("finding", {"verdict": "pass", "summary": "moves forward"}),
    ]:
        dash.handle(k, d)
    for line in dash._feed:
        line.plain.encode("cp1258")  # raises if any marker/text is unencodable


def test_banner_ascii_safe():
    cap = Console(file=io.StringIO(), width=100, record=True)
    # print_banner uses the module console; capture by swapping is overkill —
    # just ensure the banner lines themselves encode on cp1258.
    from agent.tui import _BANNER_LINES, _TAGLINE
    for line in _BANNER_LINES:
        line.encode("cp1258")
    _TAGLINE.encode("cp1258")
