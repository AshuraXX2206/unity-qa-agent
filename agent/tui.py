"""Rich Terminal UI — Claude Code / Codex-style interactive interface.

Provides:
- Streaming "thinking" spinner with elapsed time
- Syntax-highlighted code blocks
- Collapsible panels for logs, diffs, and analysis
- Step-by-step execution display with live status
- Interactive prompt for commands
- Token/cost tracking display
"""

from __future__ import annotations

import itertools
import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from rich.align import Align
from rich.box import HEAVY, ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()

# ── Branding ──────────────────────────────────────────────────────────

# Pure-ASCII wordmark (renders cleanly on any console codepage, incl. cp1258).
_BANNER_LINES = [
    r"  ___      _    ____ _____ _   _ _____",
    r" / _ \    / \  / ___| ____| \ | |_   _|",
    r"| | | |  / _ \| |  _|  _| |  \| | | |",
    r"| |_| | / ___ \ |_| | |___| |\  | | |",
    r" \__\_\/_/   \_\____|_____|_| \_| |_|",
]
_BANNER_SHADES = ["#5eead4", "#2dd4bf", "#22d3ee", "#38bdf8", "#3b82f6"]

_VERSION = "0.1.0"
_TAGLINE = "AI agent that plays and tests your Unity game"


def print_banner() -> None:
    console.print()
    for line, shade in zip(_BANNER_LINES, _BANNER_SHADES):
        console.print(Text(line, style=f"bold {shade}"))
    console.print()
    console.print(Text(f"  Unity QA Agent  v{_VERSION}", style="bold white"))
    console.print(Text(f"  {_TAGLINE}", style="dim"))
    console.print()


# ── Thinking / Streaming spinner ──────────────────────────────────────


class ThinkingIndicator:
    """Animated 'thinking…' spinner like Claude Code's streaming indicator."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking") -> None:
        self._label = label
        self._live: Optional[Live] = None
        self._start: float = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._start = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_label: Optional[str] = None) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        elapsed = time.monotonic() - self._start
        label = final_label or self._label
        console.print(
            Text(f"  {label} ", style="bold cyan"),
            Text(f"({elapsed:.1f}s)", style="dim"),
        )

    def _animate(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while self._running:
            elapsed = time.monotonic() - self._start
            frame = next(frames)
            status = f"\r  {frame} [cyan]{self._label}[/] [dim]({elapsed:.1f}s)[/]"
            console.print(status, end="")
            time.sleep(0.08)
        # Clear line
        console.print("\r" + " " * 60, end="\r")


@contextmanager
def thinking(label: str = "Thinking") -> Generator[None, None, None]:
    """Context manager for a thinking spinner."""
    t = ThinkingIndicator(label)
    t.start()
    try:
        yield
    finally:
        t.stop()


# ── Step-by-step execution display ────────────────────────────────────


class StepTracker:
    """Display test execution steps like a CI/CD pipeline."""

    def __init__(self) -> None:
        self._steps: List[Dict[str, Any]] = []

    def add_step(self, label: str, status: str = "pending") -> int:
        idx = len(self._steps)
        self._steps.append({"label": label, "status": status, "start": None, "end": None})
        return idx

    def start_step(self, idx: int) -> None:
        self._steps[idx]["status"] = "running"
        self._steps[idx]["start"] = time.monotonic()
        self._render_step(idx)

    def pass_step(self, idx: int, detail: str = "") -> None:
        self._steps[idx]["status"] = "pass"
        self._steps[idx]["end"] = time.monotonic()
        self._steps[idx]["detail"] = detail
        self._render_step(idx)

    def fail_step(self, idx: int, detail: str = "") -> None:
        self._steps[idx]["status"] = "fail"
        self._steps[idx]["end"] = time.monotonic()
        self._steps[idx]["detail"] = detail
        self._render_step(idx)

    def skip_step(self, idx: int) -> None:
        self._steps[idx]["status"] = "skip"
        self._render_step(idx)

    def _render_step(self, idx: int) -> None:
        step = self._steps[idx]
        icon = {
            "pending": "  ",
            "running": " [bold yellow]>[/]",
            "pass":    " [bold green]PASS[/]",
            "fail":    " [bold red]FAIL[/]",
            "skip":    " [dim]SKIP[/]",
        }.get(step["status"], "  ")

        elapsed = ""
        if step["start"] and step["end"]:
            elapsed = f" [dim]({step['end'] - step['start']:.1f}s)[/]"
        elif step["start"]:
            elapsed = f" [dim](running…)[/]"

        detail = ""
        if step.get("detail"):
            detail = f"\n     [dim]{step['detail']}[/]"

        console.print(f"  {icon} {step['label']}{elapsed}{detail}")


# ── Code display ──────────────────────────────────────────────────────


def show_code(code: str, language: str = "python", title: str = "") -> None:
    """Display syntax-highlighted code in a panel."""
    syntax = Syntax(
        code,
        language,
        theme="monokai",
        line_numbers=True,
        word_wrap=True,
    )
    panel = Panel(
        syntax,
        title=f"[bold]{title}[/]" if title else None,
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)


def show_yaml(data: Dict[str, Any], title: str = "YAML") -> None:
    """Display YAML data with syntax highlighting."""
    import yaml
    text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    show_code(text, language="yaml", title=title)


def show_diff(before: str, after: str, title: str = "Diff") -> None:
    """Show a simple before/after diff."""
    table = Table(title=title, show_header=True, border_style="dim")
    table.add_column("Before", style="red")
    table.add_column("After", style="green")
    table.add_row(before, after)
    console.print(table)


# ── Analysis display ──────────────────────────────────────────────────


def show_codebase_analysis(summary: Dict[str, Any]) -> None:
    """Display codebase analysis results in a rich layout."""
    console.print()
    console.print(Rule("[bold cyan]Codebase Analysis[/]", style="cyan"))
    console.print()

    # Overview stats
    stats_table = Table(box=ROUNDED, border_style="blue", show_header=False, pad_edge=True)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", justify="right", style="cyan")
    stats_table.add_row("Total C# Classes", str(summary["total_classes"]))
    stats_table.add_row("MonoBehaviour Components", str(summary["monobehaviour_count"]))
    stats_table.add_row("Input Bindings Detected", str(summary["input_bindings"]))
    stats_table.add_row("Test Suggestions Generated", str(summary["test_suggestions"]))
    console.print(Padding(stats_table, (0, 2)))

    # Detected patterns
    patterns = summary.get("detected_patterns", {})
    if patterns:
        console.print()
        console.print("  [bold]Detected Patterns:[/]")
        for pat, count in patterns.items():
            icon = {"movement": "🏃", "health": "❤️ ", "combat": "⚔️ ", "ui": "🖥️ "}.get(pat, "  ")
            bar = "█" * min(count * 3, 30)
            console.print(f"    {icon} {pat:<12} {bar} ({count})")

    # Class tree
    if summary.get("classes"):
        console.print()
        tree = Tree("[bold]Project Classes[/]", guide_style="dim")
        for cls in summary["classes"]:
            patterns_str = f" [{', '.join(cls['patterns'])}]" if cls["patterns"] else ""
            style = "green" if cls["patterns"] else "dim"
            node = tree.add(
                f"[{style}]{cls['name']}[/] : {cls['base']}"
                f"  [dim]({cls['methods']}m, {cls['fields']}f)[/]"
                f"[yellow]{patterns_str}[/]"
            )

        console.print(Padding(tree, (0, 2)))

    # Input bindings table
    if summary.get("inputs"):
        console.print()
        inp_table = Table(
            title="[bold]Input Bindings[/]",
            box=SIMPLE,
            border_style="dim",
        )
        inp_table.add_column("Key/Axis", style="cyan")
        inp_table.add_column("Type", style="yellow")
        inp_table.add_column("Source", style="dim")
        for inp in summary["inputs"]:
            inp_table.add_row(inp["key"], inp["type"], f"{inp['file']}:{inp['line']}")
        console.print(Padding(inp_table, (0, 2)))


def show_test_suggestions(suggestions: list) -> None:
    """Display auto-generated test suggestions."""
    console.print()
    console.print(Rule("[bold cyan]Auto-Generated Test Suggestions[/]", style="cyan"))
    console.print()

    table = Table(box=ROUNDED, border_style="blue")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Category", style="yellow")
    table.add_column("Confidence", justify="right")
    table.add_column("Source", style="dim")

    for s in suggestions:
        conf_color = "green" if s.confidence >= 0.8 else ("yellow" if s.confidence >= 0.6 else "red")
        table.add_row(
            s.test_id,
            s.name,
            s.category,
            f"[{conf_color}]{s.confidence:.0%}[/]",
            s.source_class if len(s.source_class) < 30 else Path(s.source_class).name,
        )

    console.print(Padding(table, (0, 2)))


# ── Test execution display ────────────────────────────────────────────


def show_test_execution_header(suite: str, persona: str, total: int) -> None:
    """Print header before test execution begins."""
    console.print()
    console.print(Rule("[bold]Test Execution[/]", style="bright_blue"))
    console.print()
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Suite:", suite)
    info.add_row("Persona:", persona)
    info.add_row("Test Cases:", str(total))
    info.add_row("Started:", datetime.now().strftime("%H:%M:%S"))
    console.print(Padding(info, (0, 2)))
    console.print()


def show_test_progress(case_id: str, name: str, status: str, duration: float,
                       error: Optional[str] = None, step_num: int = 0, total_steps: int = 0) -> None:
    """Print a single test case result line."""
    if status == "PASS":
        icon = "[bold green]PASS[/]"
    elif status == "FAIL":
        icon = "[bold red]FAIL[/]"
    else:
        icon = "[dim]SKIP[/]"

    steps_info = f" ({step_num}/{total_steps} steps)" if total_steps else ""
    console.print(
        f"  {icon}  [bold]{case_id}[/] {name}{steps_info} [dim]({duration:.1f}s)[/]"
    )
    if error:
        console.print(f"        [red]→ {error}[/]")


# ── Report display ────────────────────────────────────────────────────


def show_report_summary(result: Dict[str, Any]) -> None:
    """Display the final test report summary panel."""
    summary = result.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("pass", 0)
    failed = summary.get("fail", 0)
    skipped = summary.get("skip", 0)
    rate = summary.get("pass_rate", 0)

    rate_color = "green" if rate >= 0.8 else ("yellow" if rate >= 0.5 else "red")

    console.print()
    console.print(Rule(f"[bold {rate_color}]Results[/]", style=rate_color))
    console.print()

    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")

    grid.add_row(
        f"[bold]{total}[/]\n[dim]Total[/]",
        f"[bold green]{passed}[/]\n[dim]Passed[/]",
        f"[bold red]{failed}[/]\n[dim]Failed[/]",
        f"[dim]{skipped}[/]\n[dim]Skipped[/]",
        f"[bold {rate_color}]{rate:.0%}[/]\n[dim]Pass Rate[/]",
    )
    console.print(Align.center(grid))

    duration = result.get("duration_seconds", 0)
    console.print()
    console.print(Align.center(Text(f"Duration: {duration:.1f}s", style="dim")))
    console.print()


# ── Interactive prompt ────────────────────────────────────────────────


def prompt_choice(message: str, choices: List[str]) -> str:
    """Present an interactive numbered choice list."""
    console.print(f"\n  [bold]{message}[/]")
    for i, c in enumerate(choices, 1):
        console.print(f"    [cyan]{i}[/]) {c}")
    console.print()
    while True:
        try:
            raw = console.input("  [dim]>[/] ").strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            elif raw in choices:
                return raw
            console.print(f"  [red]Invalid choice. Enter 1-{len(choices)}.[/]")
        except (EOFError, KeyboardInterrupt):
            return choices[0]


def prompt_confirm(message: str, default: bool = True) -> bool:
    """Yes/No prompt."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = console.input(f"  [bold]{message}[/] {suffix} ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return default


def prompt_input(message: str, default: str = "") -> str:
    """Free text prompt."""
    default_hint = f" [dim]({default})[/]" if default else ""
    try:
        raw = console.input(f"  [bold]{message}[/]{default_hint} ").strip()
        return raw or default
    except (EOFError, KeyboardInterrupt):
        return default


# ── Streaming message display ─────────────────────────────────────────


def stream_message(text: str, style: str = "", delay: float = 0.01) -> None:
    """Print text character by character like a streaming LLM response."""
    for char in text:
        console.print(char, end="", style=style)
        time.sleep(delay)
    console.print()


def agent_message(text: str) -> None:
    """Display a message from the agent in a styled box."""
    console.print()
    console.print(Panel(
        Markdown(text),
        title="[bold cyan]Agent[/]",
        border_style="cyan",
        padding=(1, 2),
    ))


def system_message(text: str) -> None:
    """Display a system/info message."""
    console.print(f"  [dim italic]{text}[/]")


def error_message(text: str) -> None:
    """Display an error message."""
    console.print(Panel(
        Text(text, style="red"),
        title="[bold red]Error[/]",
        border_style="red",
        padding=(0, 1),
    ))


def success_message(text: str) -> None:
    """Display a success message."""
    console.print(f"  [bold green]>[/] {text}")


def warning_message(text: str) -> None:
    """Display a warning."""
    console.print(f"  [bold yellow]![/] {text}")


# ── File tree display ─────────────────────────────────────────────────


def show_file_tree(root: str, files: List[str], title: str = "Project") -> None:
    """Display a file tree."""
    tree = Tree(f"[bold]{title}[/] [dim]({root})[/]", guide_style="dim")
    dirs: Dict[str, Any] = {}

    for f in sorted(files):
        parts = Path(f).parts
        current = tree
        for i, part in enumerate(parts):
            key = "/".join(parts[:i + 1])
            if key not in dirs:
                style = "dim" if i < len(parts) - 1 else "green"
                dirs[key] = current.add(f"[{style}]{part}[/]")
            current = dirs[key]

    console.print(Padding(tree, (1, 2)))


# ── Progress bar for suite execution ──────────────────────────────────


def create_suite_progress() -> Progress:
    """Create a rich progress bar for test suite execution."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


# ── Agentic live dashboard ─────────────────────────────────────────────


# ASCII-safe markers — Windows consoles on non-UTF8 codepages (e.g. cp1258)
# cannot encode emoji or braille, so the dashboard avoids both.
_EVENT_STYLE = {
    "thought": ("[~]", "cyan"),
    "action": ("[>]", "bold yellow"),
    "result": ("   ", "dim"),
    "observation": ("[o]", "magenta"),
    "finding": ("[*]", "bold green"),
    "error": ("[x]", "bold red"),
}

_VERDICT_COLOR = {
    "PASS": "green",
    "FAIL": "red",
    "BUG": "red",
    "INCONCLUSIVE": "yellow",
}


class AgentDashboard:
    """A live, multi-panel dashboard for the autonomous QA agent.

    Renders the observe → reason → act loop in real time: a scrolling activity
    feed on the left, run stats on the right, a goal header, and a status
    footer. Use as a context manager around ``QAAgent.run`` and pass
    :meth:`handle` as the agent's ``on_event`` callback. On exit it drops the
    full activity log back into the scrollback so nothing is lost when the
    alternate screen closes.
    """

    def __init__(
        self,
        goal: str,
        provider: str,
        model: str,
        max_steps: int,
        console: Console = console,
    ) -> None:
        self.goal = goal
        self.provider = provider
        self.model = model
        self.max_steps = max_steps
        self._console = console
        self._start = time.monotonic()
        self._step = 0
        self._feed: List[Text] = []
        self._tool_counts: Dict[str, int] = {}
        self._status = "running"
        self._verdict = ""
        self._summary = ""
        self._live: Optional[Live] = None

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self) -> "AgentDashboard":
        self._live = Live(
            self, console=self._console, screen=True,
            refresh_per_second=8, redirect_stdout=False, redirect_stderr=False,
        )
        self._live.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._dump_log()
        return False

    # ── event handling ─────────────────────────────────────────────────

    def handle(self, kind: str, data: Any) -> None:
        """``on_event`` callback for :class:`agent.qa_agent.QAAgent`."""
        if kind == "step":
            self._step = data.get("step", self._step)
            return
        if kind == "thought":
            self._add("thought", _truncate(str(data), 240))
        elif kind == "action":
            name = data.get("name", "?")
            self._tool_counts[name] = self._tool_counts.get(name, 0) + 1
            self._add("action", f"{name}  [dim]{_compact(data.get('input', {}))}[/]")
        elif kind == "result":
            self._add("result", _truncate(str(data.get("text", "")), 100))
        elif kind == "observation":
            self._add("observation", "fresh observation captured")
        elif kind == "finding":
            self._verdict = str(data.get("verdict", "")).upper()
            self._summary = str(data.get("summary", ""))
            self._status = "done"
            color = _VERDICT_COLOR.get(self._verdict, "yellow")
            self._add("finding", f"[{color}]{self._verdict}[/] - {self._summary}")
        elif kind == "error":
            self._status = "error"
            self._add("error", str(data))

    # ── rendering ──────────────────────────────────────────────────────

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(self._footer(), name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(self._feed_panel(), name="feed", ratio=2),
            Layout(self._side_panel(), name="side", size=36),
        )
        return layout

    def _header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            Text(f"GOAL  {self.goal}", style="bold white"),
            Text(f"step {self._step}/{self.max_steps}", style="dim"),
        )
        grid.add_row(
            Text(f"{self.provider} / {self.model}", style="cyan"),
            Text(f"{self._elapsed():.0f}s", style="dim"),
        )
        return Panel(grid, title="[bold]Unity QA Agent[/]", border_style="cyan",
                     padding=(0, 1))

    def _feed_panel(self) -> Panel:
        height = max(6, self._console.size.height - 9)
        tail = self._feed[-height:]
        body = Group(*tail) if tail else Text("waiting for the agent...", style="dim italic")
        return Panel(body, title="[bold]Activity[/]", border_style="blue",
                     padding=(0, 1))

    def _side_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold", justify="right")
        table.add_column()
        table.add_row("Step", f"{self._step}/{self.max_steps}")
        table.add_row("Elapsed", f"{self._elapsed():.0f}s")
        table.add_row("Actions", str(sum(self._tool_counts.values())))
        if self._verdict:
            color = _VERDICT_COLOR.get(self._verdict, "yellow")
            table.add_row("Verdict", f"[{color}]{self._verdict}[/]")

        tools = Table.grid(padding=(0, 1))
        tools.add_column(style="yellow")
        tools.add_column(justify="right", style="dim")
        if self._tool_counts:
            for name, count in sorted(self._tool_counts.items(), key=lambda kv: -kv[1]):
                tools.add_row(name, str(count))
        else:
            tools.add_row("[dim]no tools yet[/]", "")

        body = Group(
            table,
            Rule(characters="-", style="dim"),
            Text("Tool calls", style="bold"),
            tools,
        )
        return Panel(body, title="[bold]Run Stats[/]", border_style="magenta",
                     padding=(0, 1))

    def _footer(self) -> Panel:
        if self._status == "running":
            content: Any = Columns(
                [Spinner("line", style="cyan"), Text("agent is working...", style="cyan")],
                padding=(0, 1),
            )
            border = "cyan"
        elif self._status == "done":
            color = _VERDICT_COLOR.get(self._verdict, "yellow")
            content = Text(f"{self._verdict} - {self._summary}", style=f"bold {color}")
            border = color
        else:
            content = Text("run ended with an error", style="bold red")
            border = "red"
        return Panel(content, border_style=border, padding=(0, 1))

    # ── helpers ────────────────────────────────────────────────────────

    def _add(self, kind: str, markup: str) -> None:
        icon, style = _EVENT_STYLE.get(kind, ("•", "white"))
        line = Text.from_markup(f"[{style}]{icon}[/] {markup}")
        self._feed.append(line)

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def _dump_log(self) -> None:
        """Reprint the full activity feed into normal scrollback."""
        if not self._feed:
            return
        self._console.print(Panel(
            Group(*self._feed),
            title="[bold]Agent Activity Log[/]",
            border_style="blue",
            padding=(0, 1),
        ))


def agent_dashboard(goal: str, provider: str, model: str, max_steps: int) -> AgentDashboard:
    """Convenience factory mirroring the other tui helpers."""
    return AgentDashboard(goal, provider, model, max_steps)


def _truncate(s: str, n: int) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 2] + "..."


def _compact(d: Dict[str, Any], n: int = 50) -> str:
    import json
    s = json.dumps(d, ensure_ascii=False)
    if s in ("{}", ""):
        return ""
    return s if len(s) <= n else s[: n - 2] + "..."
