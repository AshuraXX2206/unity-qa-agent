"""Simulate keyboard and mouse input via pyautogui."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import pyautogui

log = logging.getLogger(__name__)

# Disable pyautogui's built-in pause (we manage delays ourselves).
pyautogui.PAUSE = 0.0
# Keep the fail-safe (move to corner to abort).
pyautogui.FAILSAFE = True


class InputExecutor:
    """Thin wrapper around pyautogui with safe-mode and configurable delays.

    Parameters
    ----------
    safe_mode:
        When ``True``, actions are logged but **not** executed.
    action_delay:
        Seconds to sleep after each action to avoid input flooding.
    """

    def __init__(
        self,
        safe_mode: bool = False,
        action_delay: float = 0.05,
    ) -> None:
        self.safe_mode = safe_mode
        self.action_delay = action_delay
        screen = pyautogui.size()
        self._screen_w = screen.width
        self._screen_h = screen.height
        log.info(
            "InputExecutor ready (safe_mode=%s, delay=%.3fs, screen=%dx%d)",
            safe_mode, action_delay, self._screen_w, self._screen_h,
        )

    # ── keyboard ──────────────────────────────────────────────────────

    def press_key(self, key: str) -> None:
        """Press and release a single key."""
        log.debug("press_key(%s)", key)
        if not self.safe_mode:
            pyautogui.press(key)
        self._delay()

    def hold_key(self, key: str, duration: float) -> None:
        """Hold *key* down for *duration* seconds, then release."""
        log.debug("hold_key(%s, %.2fs)", key, duration)
        if not self.safe_mode:
            pyautogui.keyDown(key)
            time.sleep(max(duration, 0.01))
            pyautogui.keyUp(key)
        else:
            time.sleep(duration)
        self._delay()

    def release_key(self, key: str) -> None:
        """Release a key that is being held down."""
        log.debug("release_key(%s)", key)
        if not self.safe_mode:
            pyautogui.keyUp(key)
        self._delay()

    def press_combo(self, *keys: str) -> None:
        """Press a key combination (e.g. ``press_combo('ctrl', 'c')``)."""
        log.debug("press_combo(%s)", "+".join(keys))
        if not self.safe_mode:
            pyautogui.hotkey(*keys)
        self._delay()

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type *text* character by character."""
        log.debug("type_text(%r)", text[:40])
        if not self.safe_mode:
            pyautogui.write(text, interval=interval)
        self._delay()

    # ── mouse ─────────────────────────────────────────────────────────

    def mouse_click(
        self, x: int, y: int, button: str = "left",
    ) -> None:
        """Click at *(x, y)* with the given button."""
        log.debug("mouse_click(%d, %d, %s)", x, y, button)
        if not self.safe_mode:
            pyautogui.click(x, y, button=button)
        self._delay()

    def mouse_move(
        self, x: int, y: int, duration: float = 0.2,
    ) -> None:
        """Smoothly move the cursor to *(x, y)*."""
        log.debug("mouse_move(%d, %d, %.2fs)", x, y, duration)
        if not self.safe_mode:
            pyautogui.moveTo(x, y, duration=duration)
        self._delay()

    def mouse_drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3,
    ) -> None:
        """Drag from *(x1, y1)* to *(x2, y2)*."""
        log.debug("mouse_drag(%d,%d → %d,%d)", x1, y1, x2, y2)
        if not self.safe_mode:
            pyautogui.moveTo(x1, y1, duration=0.05)
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
        self._delay()

    def scroll(self, x: int, y: int, amount: int) -> None:
        """Scroll the mouse wheel at *(x, y)* by *amount* clicks."""
        log.debug("scroll(%d, %d, %d)", x, y, amount)
        if not self.safe_mode:
            pyautogui.moveTo(x, y, duration=0.05)
            pyautogui.scroll(amount)
        self._delay()

    # ── internal ──────────────────────────────────────────────────────

    def _delay(self) -> None:
        if self.action_delay > 0:
            time.sleep(self.action_delay)
