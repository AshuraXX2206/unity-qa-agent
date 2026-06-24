"""Pick the newest suitable model from a provider's live ``/models`` listing.

This is the "auto-update model" mechanism: instead of hardcoding a model id in
source (which goes stale), we query the provider at runtime and choose the most
recent model that can do what the QA agent needs (vision + tool use, ideally).
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.llm.base import LLMProvider

log = logging.getLogger(__name__)


def select_newest_model(
    provider: LLMProvider,
    prefer_vision: bool = True,
    fallback: str = "",
) -> str:
    """Return the id of the newest model exposed by *provider*.

    Selection order, best to worst:
      1. newest model with both vision and tool use
      2. newest model with tool use (vision not required)
      3. newest model of any kind
      4. *fallback* (or "" if none given)

    Any error querying the provider falls back gracefully — the agent should
    still run on the configured/fallback model rather than crash.
    """
    try:
        models = provider.list_models()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not list models (%s); using fallback %r", exc, fallback)
        return fallback

    if not models:
        return fallback

    models.sort(key=lambda m: m.sort_key())  # oldest → newest

    def newest(pred) -> Optional[str]:
        chosen = [m for m in models if pred(m)]
        return chosen[-1].id if chosen else None

    candidate = None
    if prefer_vision:
        candidate = newest(lambda m: m.supports_vision and m.supports_tools)
    if not candidate:
        candidate = newest(lambda m: m.supports_tools)
    if not candidate:
        candidate = models[-1].id

    log.info("Auto-selected model: %s", candidate)
    return candidate or fallback
