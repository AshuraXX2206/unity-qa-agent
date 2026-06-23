"""Player persona profiles that influence how the agent interacts with the game."""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Persona:
    """Immutable persona configuration."""

    name: str
    description: str
    input_delay: float          # seconds between actions
    skip_tutorial: bool
    action_randomness: float    # 0.0 – 1.0  probability of a "wrong" action
    exploration_rate: float     # 0.0 – 1.0  tendency to explore vs. follow path

    def should_random_action(self) -> bool:
        """Return ``True`` if the persona would deviate from the plan this step."""
        return random.random() < self.action_randomness

    def should_explore(self) -> bool:
        """Return ``True`` if the persona prefers exploring over progressing."""
        return random.random() < self.exploration_rate


PERSONAS: Dict[str, Persona] = {
    "casual": Persona(
        name="casual",
        description="Casual player — skips tutorials, rushes into action",
        input_delay=0.5,
        skip_tutorial=True,
        action_randomness=0.3,
        exploration_rate=0.2,
    ),
    "speedrunner": Persona(
        name="speedrunner",
        description="Speedrunner — optimises every action for speed",
        input_delay=0.05,
        skip_tutorial=True,
        action_randomness=0.0,
        exploration_rate=0.0,
    ),
    "explorer": Persona(
        name="explorer",
        description="Explorer — tries everything, goes everywhere",
        input_delay=0.3,
        skip_tutorial=False,
        action_randomness=0.1,
        exploration_rate=0.9,
    ),
    "griefer": Persona(
        name="griefer",
        description="Griefer — tries to break the game and find exploits",
        input_delay=0.05,
        skip_tutorial=True,
        action_randomness=0.8,
        exploration_rate=0.5,
    ),
}


def get_persona(name: str) -> Persona:
    """Look up a persona by name.  Raises ``KeyError`` if not found."""
    key = name.lower()
    if key not in PERSONAS:
        available = ", ".join(sorted(PERSONAS))
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    persona = PERSONAS[key]
    log.info("Persona loaded: %s — %s", persona.name, persona.description)
    return persona


# Random action pool used when a persona deviates.
RANDOM_KEYS: List[str] = [
    "w", "a", "s", "d", "space", "e", "q", "tab", "escape",
    "1", "2", "3", "4", "f", "r", "i",
]


def pick_random_key() -> str:
    """Return a random key from the pool (for deviation actions)."""
    return random.choice(RANDOM_KEYS)
