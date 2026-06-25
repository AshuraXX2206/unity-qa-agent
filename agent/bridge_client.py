"""WebSocket client that connects to the Unity QABridge server."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

import websockets
import websockets.exceptions

log = logging.getLogger(__name__)

_DEFAULT_URL = "ws://localhost:8765"
_MAX_RETRIES = 10
_RETRY_DELAY = 3.0  # seconds


class BridgeClient:
    """Async WebSocket client running in a background thread.

    Usage::

        client = BridgeClient(url="ws://localhost:8765")
        client.start()

        state = client.get_game_state()    # latest snapshot (dict | None)
        client.send_action({"action": "key_press", "key": "w"})

        client.stop()
    """

    def __init__(self, url: str = _DEFAULT_URL) -> None:
        self._url = url
        self._game_state: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._action_queue: asyncio.Queue[str] = asyncio.Queue()

    # ── public API ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """Start the background event-loop thread."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._action_queue = asyncio.Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Bridge client thread started")

    def stop(self) -> None:
        """Gracefully shut down the client."""
        self._running = False
        loop = self._loop
        if loop is not None:
            # Close the socket on the still-running loop before stopping it, so
            # websockets tears down cleanly (avoids "event loop is closed" noise).
            async def _shutdown() -> None:
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                loop.stop()
            try:
                asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            except Exception:  # noqa: BLE001 — loop already gone
                loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._connected = False
        log.info("Bridge client stopped")

    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """Return the most recently received game state, or ``None``."""
        with self._lock:
            return self._game_state

    def send_action(self, action: Dict[str, Any]) -> None:
        """Enqueue an action dict to be sent to Unity."""
        if self._loop is None:
            log.warning("Cannot send action — client not started")
            return
        payload = json.dumps(action)
        self._loop.call_soon_threadsafe(self._action_queue.put_nowait, payload)

    # ── internal ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except RuntimeError as e:
            if "Event loop stopped" not in str(e):
                raise

    async def _connect_loop(self) -> None:
        retries = 0
        while self._running and retries < _MAX_RETRIES:
            try:
                log.info("Connecting to %s (attempt %d/%d)…",
                         self._url, retries + 1, _MAX_RETRIES)
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    self._connected = True
                    retries = 0
                    log.info("Connected to Unity bridge")
                    await self._io_loop(ws)
            except (
                OSError,
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidURI,
                websockets.exceptions.InvalidHandshake,
            ) as exc:
                self._connected = False
                retries += 1
                log.warning("Connection lost (%s). Retry in %.0fs (%d/%d)",
                            exc, _RETRY_DELAY, retries, _MAX_RETRIES)
                await asyncio.sleep(_RETRY_DELAY)

        if retries >= _MAX_RETRIES:
            log.warning("Max retries reached. Running in screen-capture-only mode.")
        self._connected = False

    async def _io_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Concurrently read state and write queued actions."""
        read_task = asyncio.ensure_future(self._read_state(ws))
        write_task = asyncio.ensure_future(self._write_actions(ws))
        try:
            await asyncio.gather(read_task, write_task)
        except websockets.exceptions.ConnectionClosed:
            read_task.cancel()
            write_task.cancel()
            raise

    async def _read_state(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for message in ws:
            try:
                state = json.loads(message)
                with self._lock:
                    self._game_state = state
            except json.JSONDecodeError:
                log.debug("Non-JSON message from bridge: %s", message[:120])

    async def _write_actions(self, ws: websockets.WebSocketClientProtocol) -> None:
        while self._running:
            try:
                payload = await asyncio.wait_for(self._action_queue.get(), timeout=0.5)
                await ws.send(payload)
            except asyncio.TimeoutError:
                continue
