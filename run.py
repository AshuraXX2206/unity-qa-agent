#!/usr/bin/env python3
"""Launcher script — equivalent to ``python -m agent.main``."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``agent`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.main import main

main()
