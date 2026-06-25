"""Manager for MCP (Model Context Protocol) client connections."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


class McpClientManager:
    """Manages connections to multiple MCP servers in a background thread.
    
    Exposes synchronous methods to get tools and call tools.
    """

    def __init__(self, servers_config: Dict[str, Any]) -> None:
        self._servers_config = servers_config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._sessions: Dict[str, ClientSession] = {}
        self._exit_stacks: Dict[str, Any] = {}
        self._tools_cache: List[Dict[str, Any]] = []
        self._tools_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Start the background asyncio loop and connect to servers."""
        if self._running or not self._servers_config:
            return
            
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Block until tools are loaded (or failed) to ensure agent has them
        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        try:
            future.result(timeout=15.0)
            log.info("MCP Client Manager started and tools loaded")
        except Exception as e:
            log.warning("MCP connect timeout or error: %s", e)

    def stop(self) -> None:
        """Close connections and stop the background thread."""
        self._running = False
        loop = self._loop
        if loop is not None:
            async def _shutdown() -> None:
                # We do not strictly need to await exit stack cleanup on process exit,
                # but we stop the loop gracefully.
                loop.stop()
            try:
                asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            except Exception:
                loop.call_soon_threadsafe(loop.stop)
        
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Return the cached list of all available MCP tools."""
        with self._tools_lock:
            return list(self._tools_cache)

    def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> str:
        """Call an MCP tool synchronously and return the result as text."""
        if self._loop is None or server_name not in self._sessions:
            return f"Error: MCP server '{server_name}' not connected."
            
        session = self._sessions[server_name]
        
        async def _call() -> str:
            try:
                result = await session.call_tool(tool_name, arguments=args)
                if not result.content:
                    return "Success (no output)"
                # Concatenate all text parts
                text_parts = []
                for c in result.content:
                    if getattr(c, "type", "") == "text":
                        text_parts.append(c.text)
                return "\n".join(text_parts) if text_parts else "Success (no text output)"
            except Exception as e:
                return f"MCP tool call failed: {e}"
                
        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        try:
            return future.result(timeout=60.0)
        except Exception as e:
            return f"MCP tool call timeout/error: {e}"

    # ── internal ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self) -> None:
        from contextlib import AsyncExitStack
        
        for name, config in self._servers_config.items():
            cmd = config.get("command")
            args = config.get("args", [])
            if not cmd:
                continue
                
            try:
                stack = AsyncExitStack()
                self._exit_stacks[name] = stack
                
                params = StdioServerParameters(command=cmd, args=args)
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                self._sessions[name] = session
                
                # Fetch tools
                result = await session.list_tools()
                
                with self._tools_lock:
                    for tool in result.tools:
                        # Format as neutral schema
                        schema = {
                            "name": f"mcp__{name}__{tool.name}",
                            "description": f"[{name} server] {tool.description or tool.name}",
                            "input_schema": tool.inputSchema
                        }
                        self._tools_cache.append(schema)
                        
                log.info("Connected to MCP server: %s", name)
            except Exception as e:
                log.warning("Failed to connect to MCP server %s: %s", name, e)
