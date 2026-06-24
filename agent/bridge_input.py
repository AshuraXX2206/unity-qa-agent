"""Input executor that drives the game through the Unity bridge (WebSocket).

Mirrors the :class:`~agent.input_executor.InputExecutor` interface but, instead
of synthesising OS-level events with pyautogui, sends action messages to the
running ``QABridge`` over the WebSocket. This closes the agent→game control
loop entirely inside Unity (no focused-window requirement), and exercises the
bridge's write path that ``InputExecutor`` left unused.

The JSON it emits matches ``QABridge.ActionPayload``:
``{"action", "key", "keys", "duration", "x", "y", "button"}``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_BUTTON_TO_INT = {"left": 0, "right": 1, "middle": 2}


class BridgeInputExecutor:
    """Send simulated input to Unity via :class:`~agent.bridge_client.BridgeClient`.

    Parameters
    ----------
    bridge:
        A started ``BridgeClient`` (its ``send_action`` is used).
    safe_mode:
        When ``True``, actions are logged but **not** sent.
    action_delay:
        Seconds to sleep after each action (matches ``InputExecutor``).
    """

    def __init__(self, bridge: Any, safe_mode: bool = False, action_delay: float = 0.05) -> None:
        self._bridge = bridge
        self.safe_mode = safe_mode
        self.action_delay = action_delay
        log.info("BridgeInputExecutor ready (safe_mode=%s, delay=%.3fs)", safe_mode, action_delay)

    # ── keyboard ──────────────────────────────────────────────────────

    def press_key(self, key: str) -> None:
        log.debug("bridge press_key(%s)", key)
        self._send({"action": "key_press", "key": key})
        self._delay()

    def hold_key(self, key: str, duration: float) -> None:
        log.debug("bridge hold_key(%s, %.2fs)", key, duration)
        self._send({"action": "key_hold", "key": key, "duration": float(duration)})
        # Block for the hold so the next observation reflects its effect, mirroring
        # InputExecutor.hold_key (Unity ticks the hold over the same wall-clock window).
        time.sleep(max(duration, 0.01))
        self._delay()

    def release_key(self, key: str) -> None:
        log.debug("bridge release_key(%s)", key)
        self._send({"action": "key_release", "key": key})
        self._delay()

    def press_combo(self, *keys: str) -> None:
        log.debug("bridge press_combo(%s)", "+".join(keys))
        self._send({"action": "key_combo", "keys": list(keys)})
        self._delay()

    def type_text(self, text: str, interval: float = 0.02) -> None:
        log.debug("bridge type_text(%r)", text[:40])
        for ch in text:
            if ch == " ":
                self._send({"action": "key_press", "key": "space"})
            else:
                self._send({"action": "key_press", "key": ch})
            if interval and not self.safe_mode:
                time.sleep(interval)
        self._delay()

    # ── mouse ─────────────────────────────────────────────────────────

    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        log.debug("bridge mouse_click(%d, %d, %s)", x, y, button)
        self._send({
            "action": "mouse_click", "x": int(x), "y": int(y),
            "button": _BUTTON_TO_INT.get(button, 0),
        })
        self._delay()

    def mouse_move(self, x: int, y: int, duration: float = 0.2) -> None:
        log.debug("bridge mouse_move(%d, %d)", x, y)
        self._send({"action": "mouse_move", "x": int(x), "y": int(y)})
        self._delay()

    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> None:
        # The bridge has no native drag; approximate with move→click at the end.
        log.debug("bridge mouse_drag(%d,%d -> %d,%d)", x1, y1, x2, y2)
        self._send({"action": "mouse_move", "x": int(x1), "y": int(y1)})
        self._send({"action": "mouse_move", "x": int(x2), "y": int(y2)})
        self._delay()

    def scroll(self, x: int, y: int, amount: int) -> None:
        # Not modelled by the bridge; log so the agent knows it was a no-op.
        log.debug("bridge scroll(%d, %d, %d) — not supported over bridge", x, y, amount)
        self._delay()

    # ── internal ──────────────────────────────────────────────────────

    def _send(self, action: dict) -> None:
        if self.safe_mode:
            return
        self._bridge.send_action(action)

    def _delay(self) -> None:
        if self.action_delay > 0:
            time.sleep(self.action_delay)
