#!/usr/bin/env python3
"""Unity QA Agent — Claude Code-style interactive CLI.

Install globally:
  pip install .

Then just run:
  qagent              # interactive mode (first run triggers setup wizard)
  qagent run ...      # run test suites
  qagent analyze ...  # analyze Unity codebase
  qagent generate ... # auto-generate tests from code
  qagent setup        # re-run setup wizard
  qagent config       # view/edit configuration
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from agent import tui
from agent.agent_tools import ToolContext
from agent.ai_verifier import AIVerifier
from agent.bridge_client import BridgeClient
from agent.code_reader import CodeReader
from agent.config import Config
from agent.gemini_verifier import GeminiVerifier
from agent.input_executor import InputExecutor
from agent.llm.registry import PROVIDERS, make_provider
from agent.model_selector import select_newest_model
from agent.persona import get_persona, PERSONAS
from agent.qa_agent import QAAgent
from agent.report_generator import ReportGenerator
from agent.screen_observer import ScreenObserver
from agent.setup_wizard import run_setup_wizard
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


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive mode — Claude Code-style REPL."""
    load_dotenv()
    cfg = Config()

    # First-run setup wizard
    if not cfg.is_setup_complete:
        cfg = run_setup_wizard(cfg)
    else:
        tui.print_banner()

    tui.system_message("Type a command or ask a question. Type 'help' for available commands.")
    tui.system_message("Press Ctrl+C to exit.\n")

    while True:
        try:
            raw = console.input("[bold cyan]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            tui.system_message("Goodbye!")
            break

        if not raw:
            continue

        cmd = raw.split()
        command = cmd[0].lower()

        if command in ("exit", "quit", "q"):
            tui.system_message("Goodbye!")
            break

        elif command == "help":
            _show_interactive_help()

        elif command == "agent":
            _interactive_agent(raw[len("agent"):].strip(), args, cfg)

        elif command == "run":
            _interactive_run(cmd[1:], args, cfg)

        elif command == "list":
            _interactive_list(cmd[1:])

        elif command == "validate":
            _interactive_validate(cmd[1:])

        elif command == "analyze":
            _interactive_analyze(cmd[1:], cfg)

        elif command == "generate":
            _interactive_generate(cmd[1:], cfg)

        elif command == "personas":
            _show_personas()

        elif command == "suites":
            _show_available_suites()

        elif command == "status":
            _show_status(cfg)

        elif command == "setup":
            cfg = run_setup_wizard(cfg)

        elif command == "config":
            _show_config(cfg)

        else:
            tui.agent_message(
                f"Unknown command: **{command}**\n\n"
                "Available commands: `agent`, `run`, `list`, `validate`, `analyze`, "
                "`generate`, `personas`, `suites`, `status`, `help`, `exit`"
            )


def _show_interactive_help() -> None:
    tui.agent_message(
        "## Available Commands\n\n"
        "| Command | Description |\n"
        "|---------|-------------|\n"
        "| `agent <goal>` | Autonomous AI agent tests a goal (Claude-Code style) |\n"
        "| `run <suite.yaml>` | Execute a scripted YAML test suite |\n"
        "| `list <suite.yaml>` | List test cases in a suite |\n"
        "| `validate <suite.yaml>` | Validate YAML syntax |\n"
        "| `analyze <path>` | Analyze Unity C# codebase |\n"
        "| `generate <path>` | Auto-generate test cases from code |\n"
        "| `personas` | Show available persona profiles |\n"
        "| `suites` | List available test suites |\n"
        "| `status` | Show agent & connection status |\n"
        "| `config` | View current configuration |\n"
        "| `setup` | Re-run the setup wizard |\n"
        "| `help` | Show this help |\n"
        "| `exit` | Quit the agent |\n\n"
        "### Options\n"
        "- Add `--persona <name>` to `run` to choose behaviour\n"
        "- Add `--safe-mode` to `run` to log without executing\n"
        "- Add `--no-bridge` to `run` for screen-capture-only mode"
    )


def _interactive_agent(goal: str, ns: argparse.Namespace,
                       cfg: Optional[Config] = None) -> None:
    if not goal:
        goal = tui.prompt_input("What should the agent test?")
    if not goal:
        tui.error_message("No goal provided.")
        return
    no_bridge = getattr(ns, "no_bridge", False)
    safe_mode = getattr(ns, "safe_mode", False)
    _execute_agent(goal, cfg, no_bridge=no_bridge, safe_mode=safe_mode)


def _execute_agent(goal: str, cfg: Optional[Config], no_bridge: bool = False,
                   safe_mode: bool = False, max_steps: int = 15) -> None:
    """Run the autonomous QA agent against a natural-language goal."""
    global _bridge

    cfg = cfg or Config()
    provider_name = cfg.provider_name
    api_key = cfg.agent_api_key
    if not api_key:
        tui.error_message(
            f"No API key for provider '{provider_name}'. Run `setup` or set the "
            f"provider's env var."
        )
        return

    # Build provider + auto-select newest model.
    try:
        provider = make_provider(
            provider_name, api_key=api_key, model="", base_url=cfg.base_url
        )
    except Exception as exc:  # noqa: BLE001
        tui.error_message(f"Could not init provider '{provider_name}': {exc}")
        return

    configured = cfg.agent_model
    if configured and configured != "auto":
        provider.model = configured
    else:
        with tui.thinking("Discovering newest model"):
            model = select_newest_model(provider, prefer_vision=True)
        if not model:
            tui.error_message("Could not discover a model and none is pinned in config.")
            return
        provider.model = model

    tui.success_message(f"Using {provider_name} / {provider.model}")

    # Bridge (optional).
    bridge = None
    if not no_bridge:
        ws_url = cfg.get("unity_ws_url", "ws://localhost:8765")
        bridge = BridgeClient(url=ws_url)
        _bridge = bridge
        with tui.thinking("Connecting to Unity bridge"):
            bridge.start()
            deadline = time.monotonic() + 5
            while not bridge.connected and time.monotonic() < deadline:
                time.sleep(0.3)
        if bridge.connected:
            tui.success_message("Connected to Unity bridge")
        else:
            tui.warning_message("Unity bridge not available — vision/OCR only")

    persona = get_persona(cfg["default_persona"] if cfg else "casual")
    observer = ScreenObserver()
    executor = InputExecutor(safe_mode=safe_mode, action_delay=persona.input_delay)

    ctx = ToolContext(
        observer=observer, executor=executor, bridge=bridge, persona=persona,
    )
    # Live dashboard owns the screen while the agent runs.
    with tui.agent_dashboard(goal, provider_name, provider.model, max_steps) as dash:
        agent = QAAgent(
            provider=provider, ctx=ctx, provider_name=provider_name,
            max_steps=max_steps, on_event=dash.handle,
        )
        result = agent.run(goal)

    reporter = ReportGenerator()
    reporter.print_agent_report(result)
    json_path = reporter.save_agent_json(result)
    tui.system_message(f"JSON report saved → {json_path}")

    if bridge is not None:
        bridge.stop()


def _interactive_run(args_list: list, ns: argparse.Namespace,
                     cfg: Optional[Config] = None) -> None:
    if not args_list:
        suites = _find_suites()
        if not suites:
            tui.error_message("No test suites found in test_cases/")
            return
        suite = tui.prompt_choice("Select a test suite:", suites)
    else:
        suite = args_list[0]

    persona_name = (cfg["default_persona"] if cfg else "casual")
    safe_mode = False
    no_bridge = getattr(ns, "no_bridge", False)
    case_id = None

    for i, a in enumerate(args_list):
        if a == "--persona" and i + 1 < len(args_list):
            persona_name = args_list[i + 1]
        elif a == "--safe-mode":
            safe_mode = True
        elif a == "--no-bridge":
            no_bridge = True
        elif a == "--id" and i + 1 < len(args_list):
            case_id = args_list[i + 1]

    _execute_run(suite, persona_name, safe_mode, no_bridge, case_id, cfg)


def _execute_run(suite: str, persona_name: str, safe_mode: bool,
                 no_bridge: bool, case_id: Optional[str],
                 cfg: Optional[Config] = None) -> None:
    global _bridge

    ws_url = (cfg["unity_ws_url"] if cfg else None) or os.getenv("UNITY_WS_URL", "ws://localhost:8765")

    # Bridge connection with thinking spinner
    bridge = BridgeClient(url=ws_url)
    _bridge = bridge

    if not no_bridge:
        with tui.thinking("Connecting to Unity bridge"):
            bridge.start()
            deadline = time.monotonic() + 5
            while not bridge.connected and time.monotonic() < deadline:
                time.sleep(0.3)

        if bridge.connected:
            tui.success_message("Connected to Unity bridge")
        else:
            tui.warning_message("Unity bridge not available — screen-capture-only mode")
    else:
        tui.system_message("Bridge disabled — screen-capture-only mode")

    # Load persona
    try:
        persona = get_persona(persona_name)
    except KeyError as e:
        tui.error_message(str(e))
        return

    observer = ScreenObserver()
    executor = InputExecutor(safe_mode=safe_mode, action_delay=persona.input_delay)

    # AI verifier — use config provider or fall back to env vars
    ai_verifier = None
    if cfg and cfg.api_key:
        if cfg.ai_provider == "gemini":
            ai_verifier = GeminiVerifier(api_key=cfg.api_key, model=cfg.model)
        else:
            ai_verifier = AIVerifier(api_key=cfg.api_key, model=cfg.model)
    elif os.getenv("GEMINI_API_KEY"):
        ai_verifier = GeminiVerifier()
    elif os.getenv("ANTHROPIC_API_KEY"):
        ai_verifier = AIVerifier()

    reporter = ReportGenerator()

    runner = TestRunner(
        bridge=bridge,
        observer=observer,
        executor=executor,
        persona=persona,
        ai_verifier=ai_verifier,
    )

    # Load and validate suite
    with tui.thinking("Loading test suite"):
        errors = TestRunner.validate_suite(suite)
    if errors:
        tui.error_message(f"Suite validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        return

    cases = TestRunner.list_cases(suite)
    tui.show_test_execution_header(suite, persona.name, len(cases))

    # Execute with progress
    progress = tui.create_suite_progress()
    with progress:
        task = progress.add_task("Running tests…", total=len(cases))

        result = runner.run_suite(suite, case_id=case_id)

        for cr in result.cases:
            tui.show_test_progress(
                cr.id, cr.name, cr.status, cr.duration_seconds,
                error=cr.error,
                step_num=cr.steps_executed,
                total_steps=cr.steps_executed,
            )
            progress.advance(task)

    # Report
    reporter.print_report(result)
    json_path = reporter.save_json(result)
    tui.show_report_summary(result.to_dict())
    tui.system_message(f"JSON report saved → {json_path}")

    bridge.stop()


def _interactive_list(args_list: list) -> None:
    if not args_list:
        suites = _find_suites()
        if not suites:
            tui.error_message("No test suites found in test_cases/")
            return
        suite = tui.prompt_choice("Select a test suite:", suites)
    else:
        suite = args_list[0]

    with tui.thinking("Loading suite"):
        cases = TestRunner.list_cases(suite)

    from rich.table import Table
    table = Table(title=f"Test Cases — {suite}", border_style="blue")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Description", style="dim")
    for c in cases:
        table.add_row(c["id"], c["name"], c["description"])
    console.print()
    console.print(table)
    console.print()


def _interactive_validate(args_list: list) -> None:
    if not args_list:
        suites = _find_suites()
        if not suites:
            tui.error_message("No test suites found in test_cases/")
            return
        suite = tui.prompt_choice("Select a test suite:", suites)
    else:
        suite = args_list[0]

    with tui.thinking("Validating YAML"):
        errors = TestRunner.validate_suite(suite)

    if errors:
        tui.error_message(f"Validation failed for {suite}:")
        for e in errors:
            console.print(f"    [red]-[/] {e}")
    else:
        tui.success_message(f"{suite} is valid")


def _interactive_analyze(args_list: list, cfg: Optional[Config] = None) -> None:
    default_path = (cfg.unity_project_path if cfg else "") or "."
    if not args_list:
        path = tui.prompt_input("Path to Unity Assets/Scripts:", default_path)
    else:
        path = args_list[0]

    if not Path(path).exists():
        tui.error_message(f"Path not found: {path}")
        return

    with tui.thinking("Analyzing codebase"):
        reader = CodeReader(path)
        reader.scan()
        summary = reader.get_summary()

    tui.show_codebase_analysis(summary)

    if reader.suggestions:
        tui.show_test_suggestions(reader.suggestions)
        if tui.prompt_confirm("Export suggestions as YAML test suite?"):
            output = tui.prompt_input("Output path:", "test_cases/auto_generated.yaml")
            reader.export_suggestions_yaml(output)
            tui.success_message(f"Test suite exported → {output}")


def _interactive_generate(args_list: list, cfg: Optional[Config] = None) -> None:
    default_path = (cfg.unity_project_path if cfg else "") or "."
    if not args_list:
        path = tui.prompt_input("Path to Unity Assets/Scripts:", default_path)
    else:
        path = args_list[0]

    output = "test_cases/auto_generated.yaml"
    for i, a in enumerate(args_list):
        if a == "--output" and i + 1 < len(args_list):
            output = args_list[i + 1]

    if not Path(path).exists():
        tui.error_message(f"Path not found: {path}")
        return

    with tui.thinking("Scanning codebase and generating tests"):
        reader = CodeReader(path)
        reader.scan()

    if not reader.suggestions:
        tui.warning_message("No test suggestions could be generated from the codebase")
        return

    tui.show_test_suggestions(reader.suggestions)

    with tui.thinking("Writing YAML"):
        outpath = reader.export_suggestions_yaml(output)

    tui.success_message(f"Generated {len(reader.suggestions)} test cases → {outpath}")

    # Show the generated YAML
    import yaml
    with open(outpath) as f:
        data = yaml.safe_load(f)
    tui.show_yaml(data, title=f"Generated: {outpath}")


def _show_personas() -> None:
    from rich.table import Table
    table = Table(title="Persona Profiles", border_style="cyan")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Delay", justify="right")
    table.add_column("Randomness", justify="right")
    table.add_column("Exploration", justify="right")
    table.add_column("Skip Tutorial", justify="center")

    for name, p in PERSONAS.items():
        table.add_row(
            name,
            p.description,
            f"{p.input_delay}s",
            f"{p.action_randomness:.0%}",
            f"{p.exploration_rate:.0%}",
            "[green]Yes[/]" if p.skip_tutorial else "[red]No[/]",
        )
    console.print()
    console.print(table)
    console.print()


def _show_available_suites() -> None:
    suites = _find_suites()
    if not suites:
        tui.warning_message("No test suites found in test_cases/")
        return

    from rich.table import Table
    table = Table(title="Available Test Suites", border_style="blue")
    table.add_column("File", style="bold")
    table.add_column("Suite Name")
    table.add_column("Cases", justify="right")

    import yaml
    for s in suites:
        try:
            with open(s) as f:
                data = yaml.safe_load(f)
            name = data.get("test_suite", "—")
            count = len(data.get("test_cases", []))
        except Exception:
            name = "—"
            count = 0
        table.add_row(s, name, str(count))

    console.print()
    console.print(table)
    console.print()


def _show_status(cfg: Optional[Config] = None) -> None:
    bridge_status = "[green]Connected[/]" if (_bridge and _bridge.connected) else "[red]Disconnected[/]"
    provider = cfg.ai_provider if cfg else "unknown"
    has_key = bool(cfg.api_key) if cfg else bool(os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    key_status = "[green]Set[/]" if has_key else "[yellow]Not set[/]"
    model = cfg.model if cfg else "—"

    from rich.table import Table
    table = Table(title="Agent Status", border_style="cyan", show_header=False)
    table.add_column("Property", style="bold")
    table.add_column("Status")
    table.add_row("Unity Bridge", bridge_status)
    table.add_row("AI Provider", provider.title())
    table.add_row("AI Model", model)
    table.add_row("API Key", key_status)
    table.add_row("Default Persona", cfg["default_persona"] if cfg else "casual")
    table.add_row("Unity Project", cfg.unity_project_path or "[dim]not set[/]" if cfg else "—")
    table.add_row("Working Directory", str(Path.cwd()))
    table.add_row("Config File", Config.config_path() if cfg else "—")
    table.add_row("Test Suites", str(len(_find_suites())))

    console.print()
    console.print(table)
    console.print()


def _show_config(cfg: Config) -> None:
    """Display current configuration."""
    from rich.table import Table
    from rich.panel import Panel

    table = Table(title="Configuration", border_style="cyan", show_header=True)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for key, value in cfg.data.items():
        display = str(value)
        if "api_key" in key and value:
            display = f"****{str(value)[-4:]}" if len(str(value)) > 4 else "****"
        table.add_row(key, display)

    console.print()
    console.print(table)
    console.print(f"  [dim]Config file: {Config.config_path()}[/]")
    console.print()


def _find_suites() -> list:
    p = Path("test_cases")
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.yaml"))


# ── Non-interactive command wrappers ──────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Execute test suite (non-interactive)."""
    load_dotenv()
    cfg = Config()
    tui.print_banner()
    _execute_run(
        args.suite,
        args.persona,
        getattr(args, "safe_mode", False),
        args.no_bridge,
        args.id,
        cfg,
    )
    sys.exit(0)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run the autonomous QA agent (non-interactive)."""
    load_dotenv()
    cfg = Config()
    tui.print_banner()
    _execute_agent(
        args.goal,
        cfg,
        no_bridge=getattr(args, "no_bridge", False),
        safe_mode=getattr(args, "safe_mode", False),
        max_steps=getattr(args, "max_steps", 15),
    )
    sys.exit(0)


def cmd_setup(args: argparse.Namespace) -> None:
    """Run setup wizard."""
    cfg = Config()
    run_setup_wizard(cfg)


def cmd_config(args: argparse.Namespace) -> None:
    """Show configuration."""
    cfg = Config()
    _show_config(cfg)


def cmd_list(args: argparse.Namespace) -> None:
    """List test cases in a suite."""
    _interactive_list([args.suite])


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a YAML test suite."""
    _interactive_validate([args.suite])


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze Unity codebase."""
    tui.print_banner()
    _interactive_analyze([args.path])


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate test cases from codebase."""
    tui.print_banner()
    parts = [args.path]
    if args.output:
        parts += ["--output", args.output]
    _interactive_generate(parts)


# ── CLI parser ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qagent",
        description="AI-powered QA agent for Unity games",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-bridge", action="store_true",
                        help="Skip Unity bridge (screen-capture only)")
    sub = parser.add_subparsers(dest="command")

    # agent
    agent_p = sub.add_parser("agent", help="Autonomous AI agent tests a natural-language goal")
    agent_p.add_argument("goal", help="What to test, e.g. \"check the player can jump\"")
    agent_p.add_argument("--no-bridge", action="store_true",
                         help="Skip Unity bridge (vision/OCR only)")
    agent_p.add_argument("--safe-mode", action="store_true",
                         help="Log actions without executing them")
    agent_p.add_argument("--max-steps", type=int, default=15,
                         help="Max agent tool-calling steps (default: 15)")
    agent_p.set_defaults(func=cmd_agent)

    # run
    run_p = sub.add_parser("run", help="Execute a test suite")
    run_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    run_p.add_argument("--persona", default="casual",
                       help="Persona profile (default: casual)")
    run_p.add_argument("--id", default=None, help="Run only this test case ID")
    run_p.add_argument("--no-bridge", action="store_true",
                       help="Skip Unity bridge (screen-capture only)")
    run_p.add_argument("--safe-mode", action="store_true",
                       help="Log actions without executing them")
    run_p.set_defaults(func=cmd_run)

    # list
    list_p = sub.add_parser("list", help="List test cases in a suite")
    list_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    list_p.set_defaults(func=cmd_list)

    # validate
    val_p = sub.add_parser("validate", help="Validate YAML test suite syntax")
    val_p.add_argument("--suite", required=True, help="Path to YAML test suite")
    val_p.set_defaults(func=cmd_validate)

    # analyze
    ana_p = sub.add_parser("analyze", help="Analyze Unity C# codebase")
    ana_p.add_argument("--path", required=True,
                       help="Path to Unity Assets/Scripts directory")
    ana_p.set_defaults(func=cmd_analyze)

    # generate
    gen_p = sub.add_parser("generate", help="Auto-generate test cases from code")
    gen_p.add_argument("--path", required=True,
                       help="Path to Unity Assets/Scripts directory")
    gen_p.add_argument("--output", default="test_cases/auto_generated.yaml",
                       help="Output YAML path")
    gen_p.set_defaults(func=cmd_generate)

    # setup
    setup_p = sub.add_parser("setup", help="Run first-time setup wizard")
    setup_p.set_defaults(func=cmd_setup)

    # config
    config_p = sub.add_parser("config", help="View current configuration")
    config_p.set_defaults(func=cmd_config)

    return parser


def _harden_console_encoding() -> None:
    """Force UTF-8 (replace on error) so non-ASCII output never crashes the CLI
    on legacy Windows consoles (e.g. the cp1258 Vietnamese codepage)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — older/odd streams without reconfigure
            pass


def main() -> None:
    _harden_console_encoding()
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    if args.command is None:
        # Launch interactive mode
        cmd_interactive(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
