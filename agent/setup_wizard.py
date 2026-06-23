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
    _step_header(1, "AI Provider")
    console.print("  Choose your AI provider for test verification.\n")

    provider = tui.prompt_choice(
        "Select AI provider:",
        ["gemini (Google — free tier available)", "anthropic (Claude)"],
    )
    provider_key = "gemini" if "gemini" in provider else "anthropic"
    config["ai_provider"] = provider_key

    # ── Step 2: API Key ───────────────────────────────────────────────
    _step_header(2, "API Key")

    if provider_key == "gemini":
        console.print(
            "  Get your free Gemini API key at:\n"
            "  [link=https://aistudio.google.com/apikey]"
            "https://aistudio.google.com/apikey[/link]\n"
        )
        key = tui.prompt_input("Enter your Gemini API key:")
        if key:
            config["gemini_api_key"] = key
            tui.success_message("Gemini API key saved")
        else:
            tui.warning_message("No API key entered — AI verification will be disabled")
    else:
        console.print(
            "  Get your Anthropic API key at:\n"
            "  [link=https://console.anthropic.com/settings/keys]"
            "https://console.anthropic.com/settings/keys[/link]\n"
        )
        key = tui.prompt_input("Enter your Anthropic API key:")
        if key:
            config["anthropic_api_key"] = key
            tui.success_message("Anthropic API key saved")
        else:
            tui.warning_message("No API key entered — AI verification will be disabled")

    # ── Step 3: AI Model ──────────────────────────────────────────────
    _step_header(3, "AI Model")

    if provider_key == "gemini":
        model = tui.prompt_choice("Select Gemini model:", [
            "gemini-2.5-flash (fast, recommended)",
            "gemini-2.5-pro (most capable)",
            "gemini-2.0-flash (balanced)",
        ])
        model_name = model.split(" ")[0]
        config["gemini_model"] = model_name
    else:
        model = tui.prompt_choice("Select Anthropic model:", [
            "claude-sonnet-4-6 (recommended)",
            "claude-sonnet-4-20250514 (latest)",
        ])
        model_name = model.split(" ")[0]
        config["anthropic_model"] = model_name

    tui.success_message(f"Model set: {model_name}")

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
        f"  [cyan]qagent run[/]          — run test suites\n"
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
