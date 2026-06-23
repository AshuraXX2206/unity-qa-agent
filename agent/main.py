#!/usr/bin/env python3
"""Unity QA Agent — CLI entry point.

Usage examples
--------------
  # Run a full suite with a persona
  python -m agent.main run --suite test_cases/basic_movement.yaml --persona casual

  # Run a single test case
  python -m agent.main run --suite test_cases/basic_movement.yaml --id TC001

  # Screen-capture-only mode (no Unity bridge)
  python -m agent.main run --suite test_cases/ui_flow.yaml --no-bridge

  # List test cases in a suite
  python -m agent.main list --suite test_cases/basic_movement.yaml

  # Validate YAML syntax
  python -m agent.main validate --suite test_cases/basic_movement.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from agent.ai_verifier import AIVerifier
from agent.bridge_client import BridgeClient
from agent.input_executor import InputExecutor
from agent.persona import get_persona
from agent.report_generator import ReportGenerator
from agent.screen_observer import ScreenObserver
from agent.test_runner import TestRunner

console = Console()
_bridge: Optional[BridgeClient] = None


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def _shutdown(signum: int, _frame: object) -> None:
    console.print("\n[yellow]Shutting down…[/]")
    if _bridge is not None:
        _bridge.stop()
    sys.exit(0)


# ── commands ──────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Execute test suite(s)."""
    global _bridge

    load_dotenv()

    ws_url = os.getenv("UNITY_WS_URL", "ws://localhost:8765")
    safe_mode = getattr(args, "safe_mode", False)

    # Bridge
    bridge = BridgeClient(url=ws_url)
    _bridge = bridge
    if not args.no_bridge:
        bridge.start()
        console.print(f"[cyan]Connecting to Unity bridge at {ws_url}…[/]")
        import time
        deadline = time.monotonic() + 5
        while not bridge.connected and time.monotonic() < deadline:
            time.sleep(0.3)
        if bridge.connected:
            console.print("[green]Connected to Unity bridge[/]")
        else:
            console.print("[yellow]Unity bridge not available — screen-capture-only mode[/]")
    else:
        console.print("[yellow]Bridge disabled — screen-capture-only mode[/]")

    # Components
    persona = get_persona(args.persona)
    observer = ScreenObserver()
    executor = InputExecutor(safe_mode=safe_mode, action_delay=persona.input_delay)
    ai_verifier = AIVerifier() if os.getenv("ANTHROPIC_API_KEY") else None
    reporter = ReportGenerator()

    runner = TestRunner(
        bridge=bridge,
        observer=observer,
        executor=executor,
        persona=persona,
        ai_verifier=ai_verifier,
    )

    console.print(f"[bold]Suite:[/]   {args.suite}")
    console.print(f"[bold]Persona:[/] {persona.name} — {persona.description}")
    console.print()

    result = runner.run_suite(args.suite, case_id=args.id)

    reporter.print_report(result)
    json_path = reporter.save_json(result)
    console.print(f"[dim]JSON report → {json_path}[/]")

    bridge.stop()

    # Exit code reflects test outcome
    sys.exit(0 if result.failed == 0 else 1)


def cmd_list(args: argparse.Namespace) -> None:
    """List test cases in a suite."""
    from rich.table import Table

    cases = TestRunner.list_cases(args.suite)
    table = Table(title=f"Test Cases — {args.suite}", border_style="blue")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Description", style="dim")
    for c in cases:
        table.add_row(c["id"], c["name"], c["description"])
    console.print(table)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a YAML test suite."""
    errors = TestRunner.validate_suite(args.suite)
    if errors:
        console.print(f"[red]Validation failed for {args.suite}:[/]")
        for e in errors:
            console.print(f"  [red]•[/] {e}")
        sys.exit(1)
    else:
        console.print(f"[green]OK[/] — {args.suite} is valid")


# ── CLI parser ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unity-qa-agent",
        description="AI-powered QA agent for Unity games",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Execute a test suite")
    run_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    run_p.add_argument("--persona", default="casual", help="Persona profile (default: casual)")
    run_p.add_argument("--id", default=None, help="Run only this test case ID")
    run_p.add_argument("--no-bridge", action="store_true", help="Skip Unity bridge (screen-capture only)")
    run_p.add_argument("--safe-mode", action="store_true", help="Log actions without executing them")
    run_p.set_defaults(func=cmd_run)

    # list
    list_p = sub.add_parser("list", help="List test cases in a suite")
    list_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    list_p.set_defaults(func=cmd_list)

    # validate
    val_p = sub.add_parser("validate", help="Validate YAML test suite syntax")
    val_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    val_p.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
