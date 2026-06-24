"""Generate test reports in JSON and rich console output."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

log = logging.getLogger(__name__)

console = Console()


class CaseResult:
    """Outcome of a single test case."""

    __slots__ = (
        "id", "name", "status", "duration_seconds",
        "steps_executed", "verify_details", "error",
    )

    def __init__(
        self,
        id: str,
        name: str,
        status: str = "PENDING",
        duration_seconds: float = 0.0,
        steps_executed: int = 0,
        verify_details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.status = status
        self.duration_seconds = duration_seconds
        self.steps_executed = steps_executed
        self.verify_details = verify_details or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
            "steps_executed": self.steps_executed,
            "verify_details": self.verify_details,
        }
        if self.error:
            d["error"] = self.error
        return d


class TestResult:
    """Aggregated result for a full test suite run."""

    def __init__(
        self,
        suite: str,
        persona: str,
    ) -> None:
        self.suite = suite
        self.persona = persona
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.cases: List[CaseResult] = []

    def add(self, case: CaseResult) -> None:
        self.cases.append(case)

    def finish(self) -> None:
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if c.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.cases if c.status == "SKIP")

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "persona": self.persona,
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "summary": {
                "total": self.total,
                "pass": self.passed,
                "fail": self.failed,
                "skip": self.skipped,
                "pass_rate": round(self.pass_rate, 3),
            },
            "results": [c.to_dict() for c in self.cases],
        }


class ReportGenerator:
    """Render test results to console and JSON file."""

    def __init__(self, reports_dir: str = "reports") -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ── console report ────────────────────────────────────────────────

    def print_report(self, result: TestResult) -> None:
        """Print a rich table to the terminal."""
        table = Table(
            title="Unity QA Agent - Test Report",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_blue",
            expand=False,
        )
        table.add_column("ID", style="dim", width=8)
        table.add_column("Name", min_width=28)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Details", max_width=36)

        for c in result.cases:
            status_str = self._status_text(c.status)
            detail = c.error or ""
            table.add_row(
                c.id,
                c.name,
                status_str,
                f"{c.duration_seconds:.1f}s",
                detail,
            )

        console.print()
        console.print(Panel.fit(
            f"[bold]Suite:[/] {result.suite}\n"
            f"[bold]Persona:[/] {result.persona}\n"
            f"[bold]Duration:[/] {result.duration_seconds:.1f}s",
            title="Run Info",
            border_style="bright_blue",
        ))
        console.print(table)

        summary_color = "green" if result.pass_rate >= 0.8 else (
            "yellow" if result.pass_rate >= 0.5 else "red"
        )
        console.print(Panel.fit(
            f"[bold]Total:[/] {result.total}  "
            f"[green]Pass:[/] {result.passed}  "
            f"[red]Fail:[/] {result.failed}  "
            f"[dim]Skip:[/] {result.skipped}  "
            f"[{summary_color}]({result.pass_rate:.0%})[/]",
            title="Summary",
            border_style=summary_color,
        ))
        console.print()

    # ── JSON report ───────────────────────────────────────────────────

    def save_json(self, result: TestResult) -> str:
        """Write a JSON report file and return its path."""
        ts = result.start_time.strftime("%Y%m%d_%H%M%S")
        path = self._reports_dir / f"report_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("JSON report saved → %s", path)
        return str(path)

    # ── agentic report ────────────────────────────────────────────────

    def print_agent_report(self, result: Any) -> None:
        """Render an :class:`agent.qa_agent.AgentRunResult` to the console."""
        verdict = result.verdict
        color = {"PASS": "green", "FAIL": "red", "BUG": "red"}.get(verdict, "yellow")

        finding = result.findings[-1] if result.findings else None
        body = (
            f"[bold]Goal:[/] {result.goal}\n"
            f"[bold]Provider:[/] {result.provider}    [bold]Model:[/] {result.model}\n"
            f"[bold]Verdict:[/] [{color}]{verdict}[/]    "
            f"[dim](stopped: {result.stopped_reason}, {len(result.steps)} tool calls)[/]"
        )
        if finding:
            body += f"\n\n[bold]Summary:[/] {finding.summary}"
            if finding.details:
                body += f"\n[dim]{finding.details}[/]"

        console.print()
        console.print(Panel(body, title="Agentic QA Result", border_style=color))
        console.print()

    def save_agent_json(self, result: Any) -> str:
        """Write an agentic run report and return its path."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._reports_dir / f"agent_report_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("Agent JSON report saved → %s", path)
        return str(path)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _status_text(status: str) -> str:
        if status == "PASS":
            return "[bold green]PASS[/]"
        if status == "FAIL":
            return "[bold red]FAIL[/]"
        if status == "SKIP":
            return "[dim]SKIP[/]"
        return status
