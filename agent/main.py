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
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from agent import tui
from agent.agent_tools import ToolContext
from agent.ai_verifier import AIVerifier
from agent.bridge_client import BridgeClient
from agent.code_reader import CodeReader
from agent.config import Config
from agent.mcp_client import McpClientManager
from agent.gemini_verifier import GeminiVerifier
from agent.input_executor import InputExecutor
from agent.llm.registry import PROVIDERS, make_provider
from agent.model_selector import select_newest_model
from agent.orchestrator import Orchestrator, ORCH_TOOL_SCHEMAS
from agent.persona import get_persona, PERSONAS
from agent.qa_agent import QAAgent
from agent.report_generator import ReportGenerator
from agent.screen_observer import ScreenObserver
from agent.setup_wizard import run_setup_wizard
from agent.test_runner import TestRunner

console = Console()
_bridge: Optional[BridgeClient] = None


def _setup_logging(verbose: bool) -> None:
    # Default to WARNING so library chatter (httpx 'HTTP Request: GET …', etc.)
    # never leaks into the clean CLI output; -v opens it up for debugging.
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )
    if not verbose:
        # These are INFO-spammy and break the live spinner when interleaved.
        for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3",
                      "websockets", "google", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


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

    tui.print_welcome(cfg.provider_name, cfg.agent_model, cfg["default_persona"])

    # The conversational agent is built lazily on first natural-language input
    # and reused, so it remembers the conversation across turns.
    orch: Optional[Orchestrator] = None

    while True:
        try:
            raw = console.input(tui.PROMPT).strip()
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

        elif command == "reset":
            if orch is not None:
                orch.reset()
            tui.system_message("Conversation memory cleared.")

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

        elif command == "models":
            _show_models(cfg)

        elif command == "setup":
            cfg = run_setup_wizard(cfg)

        elif command == "config":
            _show_config(cfg)

        else:
            # Not a known keyword — treat it as natural language and let the
            # conversational agent understand and act on it.
            if orch is None:
                orch = _make_orchestrator(cfg, args)
            if orch is None:
                continue  # no provider/key; _make_orchestrator already explained
            try:
                # The view renders the reply live (streamed panel); ask() also
                # returns the text but we don't re-print it here.
                orch.ask(raw)
            except KeyboardInterrupt:
                tui.warning_message("Interrupted.")
                continue


def _show_interactive_help() -> None:
    tui.agent_message(
        "## Just talk to me\n\n"
        "Type anything in plain language and I'll figure out what to do — answer a "
        "question, run a suite, play and test the game, analyse your code, generate "
        "tests, and so on. Examples:\n\n"
        "- *\"can the player jump?\"* — I'll play the game and check\n"
        "- *\"analyse Assets/Scripts then generate tests for the combat system\"*\n"
        "- *\"which test suites do I have and what do they cover?\"*\n"
        "- *\"what's my current setup?\"*\n\n"
        "## Explicit Commands (shortcuts)\n\n"
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
        "| `models` | List models from the configured provider |\n"
        "| `status` | Show agent & connection status |\n"
        "| `config` | View current configuration |\n"
        "| `setup` | Re-run the setup wizard |\n"
        "| `reset` | Clear the conversation memory |\n"
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
    input_mode = getattr(ns, "input", "auto")
    _execute_agent(goal, cfg, no_bridge=no_bridge, safe_mode=safe_mode, input_mode=input_mode)


def _execute_agent(goal: str, cfg: Optional[Config], no_bridge: bool = False,
                   safe_mode: bool = False, max_steps: int = 15,
                   input_mode: str = "auto",
                   persona_name: Optional[str] = None) -> Optional["Any"]:
    """Run the autonomous QA agent against a natural-language goal.

    Returns the :class:`~agent.qa_agent.AgentRunResult` (or ``None`` if the run
    could not start) so callers like the orchestrator can summarise the outcome.
    """
    global _bridge

    cfg = cfg or Config()
    provider_name = cfg.provider_name
    api_key = cfg.agent_api_key
    if not api_key:
        tui.error_message(
            f"No API key for provider '{provider_name}'. Run `setup` or set the "
            f"provider's env var."
        )
        return None

    # Build provider + auto-select newest model.
    try:
        provider = make_provider(
            provider_name, api_key=api_key, model="", base_url=cfg.base_url
        )
    except Exception as exc:  # noqa: BLE001
        tui.error_message(f"Could not init provider '{provider_name}': {exc}")
        return None

    configured = cfg.agent_model
    if configured and configured != "auto":
        provider.model = configured
    else:
        with tui.thinking("Discovering newest model"):
            model = select_newest_model(provider, prefer_vision=True)
        if not model:
            tui.error_message("Could not discover a model and none is pinned in config.")
            return None
        provider.model = model

    tui.success_message(f"Using {provider_name} / {provider.model}")

    with tui.thinking("Testing API quota"):
        test_resp = provider.chat(
            system="You are a ping bot. Reply 'pong' and nothing else.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            tools=[]
        )
        if test_resp.stop_reason == "error":
            if "429" in (test_resp.text or ""):
                tui.error_message(
                    f"API Quota Exceeded (429) for model '{provider.model}'.\n"
                    f"If you are using the free tier, this model might have exhausted its quota.\n"
                    f"Tip: Run `qagent config` and change the Agent Model to 'gemini-2.5-flash'."
                )
            else:
                tui.error_message(f"API connection test failed: {test_resp.text}")
            return None

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

    pname = persona_name or (cfg["default_persona"] if cfg else "casual")
    try:
        persona = get_persona(pname)
    except KeyError as exc:
        tui.error_message(str(exc))
        return None
    observer = ScreenObserver()

    # Pick the input backend: bridge (drive Unity over WebSocket) vs OS (pyautogui).
    use_bridge_input = (
        input_mode in ("bridge", "auto")
        and bridge is not None and bridge.connected
    )
    if input_mode == "bridge" and not use_bridge_input:
        tui.warning_message("Bridge input requested but bridge not connected — using OS input")
    if use_bridge_input:
        from agent.bridge_input import BridgeInputExecutor
        executor: Any = BridgeInputExecutor(
            bridge, safe_mode=safe_mode, action_delay=persona.input_delay
        )
        tui.system_message("Input backend: Unity bridge")
    else:
        executor = InputExecutor(safe_mode=safe_mode, action_delay=persona.input_delay)
        tui.system_message("Input backend: OS (pyautogui)")

    mcp_manager = None
    if cfg and getattr(cfg, "mcp_servers", None):
        mcp_manager = McpClientManager(cfg.mcp_servers)
        with tui.thinking("Connecting to MCP servers"):
            mcp_manager.start()

    ctx = ToolContext(
        observer=observer, executor=executor, bridge=bridge, persona=persona,
    )
    # Live dashboard owns the screen while the agent runs.
    with tui.agent_dashboard(goal, provider_name, provider.model, max_steps) as dash:
        agent = QAAgent(
            provider=provider, ctx=ctx, provider_name=provider_name,
            max_steps=max_steps, on_event=dash.handle, mcp_manager=mcp_manager
        )
        result = agent.run(goal)

    reporter = ReportGenerator()
    reporter.print_agent_report(result)
    json_path = reporter.save_agent_json(result)
    tui.system_message(f"JSON report saved → {json_path}")

    if mcp_manager:
        mcp_manager.stop()

    if bridge is not None:
        bridge.stop()
        _bridge = None

    return result


# ── Orchestrator: the conversational agent behind the REPL ─────────────


def _summarize_suite_result(result: Any, suite: str) -> str:
    """Compact, model-readable summary of a scripted suite run."""
    if result is None:
        return f"The suite '{suite}' did not run (setup/validation error — see console)."
    lines = [
        f"Ran suite '{suite}': {result.passed} passed, {result.failed} failed, "
        f"{result.skipped} skipped ({result.pass_rate:.0%} pass rate)."
    ]
    fails = [c for c in result.cases if c.status == "FAIL"]
    for c in fails[:8]:
        lines.append(f"  FAIL {c.id} {c.name}: {c.error or ''}")
    return "\n".join(lines)


def _summarize_agent_result(result: Any, goal: str) -> str:
    """Compact, model-readable summary of an autonomous game-agent run."""
    if result is None:
        return (
            f"The game agent could not start for goal '{goal}' "
            f"(missing API key or model — see console)."
        )
    parts = [
        f"Game agent finished goal '{goal}': verdict {result.verdict} "
        f"(stopped: {result.stopped_reason}, {len(result.steps)} steps)."
    ]
    if result.findings:
        f = result.findings[-1]
        parts.append(f"Summary: {f.summary}")
        if f.details:
            parts.append(f"Details: {f.details}")
    return " ".join(parts)


def _build_orchestrator_handlers(cfg: Config, ns: argparse.Namespace) -> dict:
    """Wrap the existing capabilities as callable tools for the orchestrator.

    Each handler performs the action (with its normal rich console output) and
    returns a short text result the orchestrator feeds back to the LLM.
    """
    no_bridge_default = getattr(ns, "no_bridge", False)
    input_mode = getattr(ns, "input", "auto")

    def _default_persona() -> str:
        return cfg["default_persona"] if cfg else "casual"

    def _default_path(args: dict) -> str:
        return str(args.get("path") or (cfg.unity_project_path if cfg else "") or ".")

    def h_list_suites(args: dict) -> str:
        suites = _find_suites()
        if not suites:
            return "No test suites found in test_cases/."
        import yaml
        lines = ["Available test suites:"]
        for s in suites:
            try:
                with open(s) as f:
                    data = yaml.safe_load(f)
                name = data.get("test_suite", "?")
                count = len(data.get("test_cases", []))
            except Exception:  # noqa: BLE001
                name, count = "?", 0
            lines.append(f'  - {s} — "{name}" ({count} cases)')
        return "\n".join(lines)

    def h_run_suite(args: dict) -> str:
        suite = str(args.get("suite", "")).strip()
        if not suite:
            return "Error: 'suite' is required. Call list_test_suites to find one."
        if not Path(suite).exists():
            return f"Error: suite not found: {suite}. Call list_test_suites for valid paths."
        result = _execute_run(
            suite,
            str(args.get("persona") or _default_persona()),
            bool(args.get("safe_mode", False)),
            bool(args.get("no_bridge", no_bridge_default)),
            args.get("case_id"),
            cfg,
        )
        return _summarize_suite_result(result, suite)

    def h_play_game(args: dict) -> str:
        goal = str(args.get("goal", "")).strip()
        if not goal:
            return "Error: 'goal' is required."
        result = _execute_agent(
            goal,
            cfg,
            no_bridge=bool(args.get("no_bridge", no_bridge_default)),
            safe_mode=bool(args.get("safe_mode", False)),
            max_steps=int(args.get("max_steps", 15) or 15),
            input_mode=input_mode,
            persona_name=args.get("persona"),
        )
        return _summarize_agent_result(result, goal)

    def h_analyze(args: dict) -> str:
        path = _default_path(args)
        if not Path(path).exists():
            return f"Error: path not found: {path}"
        with tui.thinking("Analyzing codebase"):
            reader = CodeReader(path)
            reader.scan()
            summary = reader.get_summary()
        tui.show_codebase_analysis(summary)
        if reader.suggestions:
            tui.show_test_suggestions(reader.suggestions)
        patterns = ", ".join(f"{k}:{v}" for k, v in summary.get("detected_patterns", {}).items())
        return (
            f"Analyzed {path}: {summary['total_classes']} classes, "
            f"{summary['monobehaviour_count']} MonoBehaviours, "
            f"{summary['input_bindings']} input bindings, "
            f"{summary['test_suggestions']} test suggestions. "
            f"Patterns: {patterns or 'none'}."
        )

    def h_generate(args: dict) -> str:
        path = _default_path(args)
        output = str(args.get("output") or "test_cases/auto_generated.yaml")
        if not Path(path).exists():
            return f"Error: path not found: {path}"
        with tui.thinking("Scanning codebase and generating tests"):
            reader = CodeReader(path)
            reader.scan()
        if not reader.suggestions:
            return f"No test suggestions could be generated from {path}."
        tui.show_test_suggestions(reader.suggestions)
        outpath = reader.export_suggestions_yaml(output)
        return f"Generated {len(reader.suggestions)} test cases → {outpath}."

    def h_validate(args: dict) -> str:
        suite = str(args.get("suite", "")).strip()
        if not suite:
            return "Error: 'suite' is required."
        errors = TestRunner.validate_suite(suite)
        if errors:
            return f"{suite} is INVALID:\n" + "\n".join(f"  - {e}" for e in errors)
        return f"{suite} is valid."

    def h_status(args: dict) -> str:
        _show_status(cfg)
        bridge_ok = bool(_bridge and _bridge.connected)
        return (
            f"Provider: {cfg.provider_name}, model: {cfg.agent_model}, "
            f"API key: {'set' if cfg.agent_api_key else 'missing'}, "
            f"bridge: {'connected' if bridge_ok else 'disconnected'}, "
            f"persona: {cfg['default_persona']}, "
            f"unity project: {cfg.unity_project_path or 'not set'}, "
            f"suites: {len(_find_suites())}."
        )

    return {
        "list_test_suites": h_list_suites,
        "run_test_suite": h_run_suite,
        "play_and_test_game": h_play_game,
        "analyze_codebase": h_analyze,
        "generate_tests": h_generate,
        "validate_suite": h_validate,
        "get_status": h_status,
    }


class _OrchestratorView:
    """Turns the orchestrator's event stream into a high-end live display.

    One instance is reused across REPL turns. While the model thinks, an
    animated 'working' line (:class:`tui.ThinkingStream`) shimmers; once tokens
    arrive they stream into a :class:`tui.ResponseStream` that types the answer
    out and settles into a panel. Tool calls and reasoning appear as a feed in
    between. Both live regions are stopped before a tool runs, so a tool that
    opens its own ``Live`` (progress bar, dashboard) never nests.
    """

    def __init__(self) -> None:
        self._think: Optional[Any] = None   # tui.ThinkingStream (shimmer)
        self._resp: Optional[Any] = None    # tui.ResponseStream (typed answer)

    def __call__(self, kind: str, payload: Any) -> None:
        if kind == "turn_start":
            self._think = tui.ThinkingStream()
            self._resp = None
        elif kind == "turn_end":
            self._settle_answer()
            if self._think is not None:
                self._think.stop()
                self._think = None
        elif kind == "thinking":
            if self._think is not None:
                self._think.think("Thinking")
        elif kind == "delta":
            # First token of a message: drop the shimmer and start typing it out.
            if self._think is not None:
                self._think.stop()
            if self._resp is None:
                self._resp = tui.ResponseStream()
            self._resp.feed(str(payload))
        elif kind == "message":
            self._finish_message(payload)
        elif kind == "action":
            if self._think is not None:
                self._think.stop()
            tui.tool_call_line(payload.get("name", "?"), payload.get("input", {}))

    def _finish_message(self, payload: Any) -> None:
        final = bool(payload.get("final"))
        text = str(payload.get("text") or "")
        if self._think is not None:
            self._think.stop()
        if final:
            if self._resp is not None:
                self._resp.finalize_answer()       # streamed → settle the panel
            elif text.strip():
                tui.agent_message(text)            # non-streaming fallback
        else:
            if self._resp is not None:
                self._resp.finalize_narration()    # was narration before a tool
            elif text.strip():
                tui.narration(text)
        self._resp = None

    def _settle_answer(self) -> None:
        """Safety net: if a response was mid-stream at turn end, settle it."""
        if self._resp is not None:
            self._resp.finalize_answer()
            self._resp = None


def _make_orchestrator(cfg: Config, ns: argparse.Namespace) -> Optional[Orchestrator]:
    """Build the conversational agent: provider (auto model) + tool handlers."""
    if not cfg.agent_api_key:
        tui.error_message(
            f"No API key for provider '{cfg.provider_name}'. Run `setup` to chat "
            f"with the agent, or use the explicit commands (type `help`)."
        )
        return None
    try:
        provider = make_provider(
            cfg.provider_name, api_key=cfg.agent_api_key, model="", base_url=cfg.base_url
        )
    except Exception as exc:  # noqa: BLE001
        tui.error_message(f"Could not init provider '{cfg.provider_name}': {exc}")
        return None

    configured = cfg.agent_model
    if configured and configured != "auto":
        provider.model = configured
    else:
        with tui.thinking("Discovering newest model"):
            model = select_newest_model(provider, prefer_vision=False)
        if not model:
            tui.error_message("Could not discover a model and none is pinned in config.")
            return None
        provider.model = model

    tui.system_message(f"Chatting with {cfg.provider_name} / {provider.model}")
    handlers = _build_orchestrator_handlers(cfg, ns)
    return Orchestrator(
        provider=provider, handlers=handlers, provider_name=cfg.provider_name,
        on_event=_OrchestratorView(),
    )


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
                 cfg: Optional[Config] = None) -> Optional["Any"]:
    """Execute a scripted YAML suite. Returns the suite result (or ``None`` on a
    setup error) so callers like the orchestrator can summarise it."""
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
        return None

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
        return None

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
    _bridge = None

    return result


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
    provider = cfg.provider_name if cfg else "unknown"
    has_key = bool(cfg.agent_api_key) if cfg else False
    key_status = "[green]Set[/]" if has_key else "[yellow]Not set[/]"
    model = cfg.agent_model if cfg else "—"

    from rich.table import Table
    table = Table(title="Agent Status", border_style="cyan", show_header=False)
    table.add_column("Property", style="bold")
    table.add_column("Status")
    table.add_row("Unity Bridge", bridge_status)
    table.add_row("AI Provider", provider.title())
    table.add_row("AI Model", model + (" [dim](auto)[/]" if model == "auto" else ""))
    table.add_row("API Key", key_status)
    table.add_row("Default Persona", cfg["default_persona"] if cfg else "casual")
    table.add_row("Unity Project", cfg.unity_project_path or "[dim]not set[/]" if cfg else "—")
    table.add_row("Working Directory", str(Path.cwd()))
    table.add_row("Config File", Config.config_path() if cfg else "—")
    table.add_row("Test Suites", str(len(_find_suites())))

    console.print()
    console.print(table)
    console.print()


def _show_models(cfg: Optional[Config] = None) -> None:
    """Query the active provider's /models endpoint and list what's available."""
    cfg = cfg or Config()
    if not cfg.agent_api_key:
        tui.error_message(
            f"No API key for provider '{cfg.provider_name}'. Run `setup` first."
        )
        return
    try:
        provider = make_provider(
            cfg.provider_name, api_key=cfg.agent_api_key, base_url=cfg.base_url
        )
    except Exception as exc:  # noqa: BLE001
        tui.error_message(f"Could not init provider: {exc}")
        return

    with tui.thinking(f"Querying {cfg.provider_name} /models"):
        try:
            models = provider.list_models()
        except Exception as exc:  # noqa: BLE001
            models = []
            err = str(exc)
        else:
            err = ""
    if err:
        tui.error_message(f"Failed to list models: {err}")
        return
    if not models:
        tui.warning_message("Provider returned no models.")
        return

    models.sort(key=lambda m: m.sort_key(), reverse=True)
    newest = select_newest_model(provider, prefer_vision=True)

    from rich.table import Table
    table = Table(title=f"Models — {cfg.provider_name}", border_style="cyan")
    table.add_column("Model ID", style="bold")
    table.add_column("Vision", justify="center")
    table.add_column("Tools", justify="center")
    table.add_column("", style="green")
    for m in models[:40]:
        table.add_row(
            m.id,
            "[green]yes[/]" if m.supports_vision else "[dim]no[/]",
            "[green]yes[/]" if m.supports_tools else "[dim]no[/]",
            "<- auto-selected" if m.id == newest else "",
        )
    console.print()
    console.print(table)
    console.print(f"  [dim]'auto' would pick:[/] [bold]{newest}[/]")
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
        input_mode=getattr(args, "input", "auto"),
        persona_name=getattr(args, "persona", None),
    )
    sys.exit(0)


def cmd_ask(args: argparse.Namespace) -> None:
    """One-shot conversational agent turn (non-interactive)."""
    load_dotenv()
    cfg = Config()
    tui.print_banner()
    orch = _make_orchestrator(cfg, args)
    if orch is None:
        sys.exit(1)
    orch.ask(args.message)   # the view renders the reply (streamed)
    sys.exit(0)


def cmd_setup(args: argparse.Namespace) -> None:
    """Run setup wizard."""
    cfg = Config()
    run_setup_wizard(cfg)


def cmd_config(args: argparse.Namespace) -> None:
    """Show configuration."""
    cfg = Config()
    _show_config(cfg)


def cmd_models(args: argparse.Namespace) -> None:
    """List models from the configured provider's /models endpoint."""
    load_dotenv()
    cfg = Config()
    tui.print_banner()
    _show_models(cfg)
    sys.exit(0)


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


def cmd_init(args: argparse.Namespace) -> None:
    """Install QABridge.cs into a Unity project."""
    target_path = Path(args.path).resolve()
    assets_dir = target_path / "Assets"
    
    if not assets_dir.is_dir():
        console.print(f"[red]Error: Could not find 'Assets' directory in {target_path}. Is this a Unity project?[/red]")
        sys.exit(1)
        
    qa_scripts_dir = assets_dir / "Scripts" / "QAAgent"
    qa_scripts_dir.mkdir(parents=True, exist_ok=True)
    
    source_bridge_path = Path(__file__).resolve().parent.parent / "unity-bridge" / "QABridge.cs"
    if not source_bridge_path.exists():
        console.print(f"[red]Error: Source QABridge.cs not found at {source_bridge_path}[/red]")
        sys.exit(1)
        
    dest_bridge_path = qa_scripts_dir / "QABridge.cs"
    try:
        shutil.copy2(source_bridge_path, dest_bridge_path)
        console.print(f"[green]Successfully installed QABridge.cs to {dest_bridge_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error copying file: {e}[/red]")
        sys.exit(1)


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
    agent_p.add_argument("--persona", default=None,
                         help="Persona profile: casual / speedrunner / explorer / griefer")
    agent_p.add_argument("--no-bridge", action="store_true",
                         help="Skip Unity bridge (vision/OCR only)")
    agent_p.add_argument("--safe-mode", action="store_true",
                         help="Log actions without executing them")
    agent_p.add_argument("--max-steps", type=int, default=15,
                         help="Max agent tool-calling steps (default: 15)")
    agent_p.add_argument("--input", choices=["auto", "bridge", "os"], default="auto",
                         help="Input backend: bridge (Unity WebSocket), os (pyautogui), "
                              "or auto (bridge if connected; default)")
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

    # ask — one-shot conversational agent
    ask_p = sub.add_parser(
        "ask", help="Ask the agent a question or give it a task in plain language"
    )
    ask_p.add_argument("message", help='e.g. "analyse Assets/Scripts and generate tests"')
    ask_p.add_argument("--no-bridge", action="store_true",
                       help="Skip Unity bridge for any actions it takes")
    ask_p.add_argument("--input", choices=["auto", "bridge", "os"], default="auto",
                       help="Input backend for game-driving actions (default auto)")
    ask_p.set_defaults(func=cmd_ask)

    # setup
    setup_p = sub.add_parser("setup", help="Run first-time setup wizard")
    setup_p.set_defaults(func=cmd_setup)

    # config
    config_p = sub.add_parser("config", help="View current configuration")
    config_p.set_defaults(func=cmd_config)

    # models
    models_p = sub.add_parser("models", help="List models from the configured provider")
    models_p.set_defaults(func=cmd_models)

    # init
    init_p = sub.add_parser("init", help="Install QABridge.cs into a Unity project")
    init_p.add_argument("--path", default=".", help="Path to Unity project (default: current directory)")
    init_p.set_defaults(func=cmd_init)

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
