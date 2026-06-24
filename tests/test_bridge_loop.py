"""End-to-end bridge control loop WITHOUT Unity.

Stands up a fake QABridge WebSocket server (mimicking the C# side), connects the
real BridgeClient, and proves both directions of the loop:
  • the agent reads broadcast game state, and
  • BridgeInputExecutor's actions arrive at the server in the exact JSON shape
    QABridge.ActionPayload expects.

This is the proof that the previously-dead send_action write path now works.
"""

import asyncio
import json
import threading
import time

import websockets

from agent.bridge_client import BridgeClient
from agent.bridge_input import BridgeInputExecutor


class FakeQABridgeServer:
    """Minimal stand-in for the Unity QABridge WebSocket server."""

    def __init__(self):
        self.received = []
        self.port = None
        self._loop = None
        self._stop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        assert self._ready.wait(5), "fake bridge server failed to start"

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        self._stop = asyncio.Event()

        async def handler(ws, *_):
            # Broadcast a state snapshot on connect, like QABridge does.
            await ws.send(json.dumps(
                {"player": {"position": {"z": 2.0}}, "scene": "Level_01"}))
            try:
                async for msg in ws:
                    self.received.append(msg)
            except Exception:
                pass

        async with websockets.serve(handler, "localhost", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def stop(self):
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_bridge_read_state_and_send_action():
    server = FakeQABridgeServer()
    server.start()
    client = BridgeClient(url=f"ws://localhost:{server.port}")
    try:
        client.start()
        assert _wait(lambda: client.connected), "client never connected"

        # ── read direction: broadcast state reaches the agent ──
        assert _wait(lambda: client.get_game_state() is not None), "no state received"
        state = client.get_game_state()
        assert state["player"]["position"]["z"] == 2.0
        assert state["scene"] == "Level_01"

        # ── write direction: agent input reaches the server ──
        exe = BridgeInputExecutor(client, action_delay=0)
        exe.press_key("w")
        exe.press_combo("shift", "w")
        exe.mouse_click(540, 360, button="right")

        assert _wait(lambda: len(server.received) >= 3), \
            f"server only got {len(server.received)} actions"

        actions = [json.loads(m) for m in server.received]
        kp = next(a for a in actions if a["action"] == "key_press")
        assert kp["key"] == "w"
        combo = next(a for a in actions if a["action"] == "key_combo")
        assert combo["keys"] == ["shift", "w"]
        click = next(a for a in actions if a["action"] == "mouse_click")
        assert click["x"] == 540 and click["y"] == 360 and click["button"] == 1
    finally:
        client.stop()
        server.stop()


def test_safe_mode_sends_nothing():
    server = FakeQABridgeServer()
    server.start()
    client = BridgeClient(url=f"ws://localhost:{server.port}")
    try:
        client.start()
        assert _wait(lambda: client.connected)
        exe = BridgeInputExecutor(client, safe_mode=True, action_delay=0)
        exe.press_key("w")
        time.sleep(1.0)
        assert server.received == []
    finally:
        client.stop()
        server.stop()
