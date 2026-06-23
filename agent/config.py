"""Persistent configuration manager — stores settings in ~/.qagent/config.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".qagent"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"

_DEFAULTS: Dict[str, Any] = {
    "ai_provider": "gemini",           # "gemini" | "anthropic"
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "anthropic_model": "claude-sonnet-4-6",
    "unity_ws_url": "ws://localhost:8765",
    "screenshot_interval": 0.5,
    "default_persona": "casual",
    "unity_project_path": "",
    "safe_mode": False,
    "setup_complete": False,
}


class Config:
    """Read / write ~/.qagent/config.yaml with defaults."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = dict(_DEFAULTS)
        self._load()

    # ── access ────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    @property
    def is_setup_complete(self) -> bool:
        return bool(self._data.get("setup_complete"))

    @property
    def ai_provider(self) -> str:
        return str(self._data.get("ai_provider", "gemini"))

    @property
    def api_key(self) -> str:
        """Return the active API key based on provider."""
        if self.ai_provider == "gemini":
            return self._data.get("gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")
        return self._data.get("anthropic_api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def model(self) -> str:
        if self.ai_provider == "gemini":
            return self._data.get("gemini_model", "gemini-2.5-flash")
        return self._data.get("anthropic_model", "claude-sonnet-4-6")

    @property
    def unity_project_path(self) -> str:
        return self._data.get("unity_project_path", "")

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    # ── persistence ───────────────────────────────────────────────────

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        safe_data = dict(self._data)
        # Mask API keys in file (store only last 4 chars hint)
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(safe_data, f, default_flow_style=False, sort_keys=False)
        log.debug("Config saved → %s", _CONFIG_FILE)

    def _load(self) -> None:
        if not _CONFIG_FILE.exists():
            return
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._data.update(data)
            log.debug("Config loaded from %s", _CONFIG_FILE)
        except Exception as exc:
            log.warning("Failed to load config: %s", exc)

    def reset(self) -> None:
        self._data = dict(_DEFAULTS)
        self.save()

    @staticmethod
    def config_path() -> str:
        return str(_CONFIG_FILE)
