"""First-run setup wizard — guides user through initial configuration."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent import tui
from agent.config import Config

log = logging.getLogger(__name__)
console = Console()


def run_setup_wizard(config: Optional[Config] = None) -> Config:
    """Interactive first-run wizard. Returns the updated Config."""
    if config is None:
        config = Config()

    tui.print_banner()

    console.print(Panel(
        "[bold cyan]Welcome to Unity QA Agent![/]\n\n"
        "Let's set up your environment. This only takes a minute.\n"
        "Your settings will be saved to [dim]~/.qagent/config.yaml[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # ── Step 1: AI Provider ───────────────────────────────────────────
    from agent.llm.registry import PROVIDERS

    _step_header(1, "AI Provider")
    console.print("  Choose the AI provider that powers the agent.\n")

    names = list(PROVIDERS.keys())
    labels = [PROVIDERS[n].label for n in names]
    chosen_label = tui.prompt_choice("Select AI provider:", labels)
    provider_key = names[labels.index(chosen_label)]
    spec = PROVIDERS[provider_key]
    config["provider"] = provider_key
    config["base_url"] = spec.base_url
    config["model"] = "auto"  # Reset model when provider changes
    # Keep the legacy scripted-verifier field aligned for gemini/anthropic.
    if provider_key in ("gemini", "anthropic"):
        config["ai_provider"] = provider_key

    # ── Step 2: API Key ───────────────────────────────────────────────
    _step_header(2, "API Key")

    if spec.key_url:
        console.print(
            f"  Get your API key at:\n"
            f"  [link={spec.key_url}]{spec.key_url}[/link]\n"
        )
    key = tui.prompt_input(f"Enter your {spec.label.split(' (')[0]} API key:")
    if key:
        config.set_agent_api_key(provider_key, key)
        # Mirror into legacy fields so scripted mode keeps working.
        if provider_key == "gemini":
            config["gemini_api_key"] = key
        elif provider_key == "anthropic":
            config["anthropic_api_key"] = key
        tui.success_message("API key saved")
    else:
        tui.warning_message("No API key entered — the agent won't be able to run")

    # ── Step 3: AI Model ──────────────────────────────────────────────
    _step_header(3, "AI Model")
    console.print(
        "  The agent auto-discovers the newest model via the provider's /models\n"
        "  endpoint at runtime, so it stays current without code changes.\n"
    )
    pin = tui.prompt_confirm("Auto-select newest model each run? (recommended)", default=True)
    if pin:
        config["model"] = "auto"
        tui.success_message("Model: auto (newest discovered at runtime)")
    else:
        model_name = tui.prompt_input("Pin a specific model id:", "")
        config["model"] = model_name or "auto"
        tui.success_message(f"Model: {config['model']}")

    # ── Step 4: Unity Project Path ────────────────────────────────────
    _step_header(4, "Unity Project")

    console.print(
        "  Point to your Unity project's Scripts folder so the agent\n"
        "  can read your codebase and auto-generate tests.\n"
    )
    unity_path = tui.prompt_input(
        "Path to Unity Assets/Scripts (or Enter to skip):",
        "",
    )
    if unity_path:
        config["unity_project_path"] = unity_path
        tui.success_message(f"Unity project path: {unity_path}")
    else:
        tui.system_message("Skipped — you can set this later with 'qagent config'")

    # ── Step 5: Default Persona ───────────────────────────────────────
    _step_header(5, "Default Persona")

    persona = tui.prompt_choice("Select default player persona:", [
        "casual — skips tutorials, rushes into action",
        "speedrunner — optimises every action",
        "explorer — tries everything, goes everywhere",
        "griefer — tries to break the game",
    ])
    persona_name = persona.split(" ")[0]
    config["default_persona"] = persona_name
    tui.success_message(f"Default persona: {persona_name}")

    # ── Step 6: WebSocket URL ─────────────────────────────────────────
    _step_header(6, "Unity Bridge")

    ws_url = tui.prompt_input(
        "WebSocket URL for Unity bridge:",
        "ws://localhost:8765",
    )
    config["unity_ws_url"] = ws_url

    # ── Done ──────────────────────────────────────────────────────────
    config["setup_complete"] = True
    config.save()

    console.print()
    console.print(Panel(
        f"[bold green]Setup complete![/]\n\n"
        f"Config saved to: [dim]{config.config_path()}[/]\n\n"
        f"[bold]Quick start:[/]\n"
        f"  [cyan]qagent[/]              — interactive mode\n"
        f"  [cyan]qagent agent \"...\"[/]  — autonomous AI agent (Claude-Code style)\n"
        f"  [cyan]qagent run[/]          — run scripted test suites\n"
        f"  [cyan]qagent analyze[/]      — analyze Unity codebase\n"
        f"  [cyan]qagent generate[/]     — auto-generate tests\n"
        f"  [cyan]qagent setup[/]        — re-run this wizard\n"
        f"  [cyan]qagent config[/]       — view/edit config",
        title="[bold]Ready to go![/]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()

    return config


def _step_header(num: int, title: str) -> None:
    console.print()
    console.print(f"  [bold cyan]Step {num}[/] — [bold]{title}[/]")
    console.print()
