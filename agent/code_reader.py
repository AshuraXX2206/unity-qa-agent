"""Codebase analysis module — reads Unity C# source files to understand game logic.

Capabilities:
- Scan a Unity project directory for C# scripts
- Extract class definitions, methods, fields, MonoBehaviour components
- Identify input handling (GetKey, GetAxis, OnClick, etc.)
- Identify player controllers, health systems, UI panels
- Suggest test cases based on detected patterns
- Generate YAML test stubs from code analysis
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)


@dataclass
class CSharpMethod:
    name: str
    return_type: str
    parameters: List[str]
    body_snippet: str
    line_number: int


@dataclass
class CSharpField:
    name: str
    field_type: str
    access: str
    is_serialized: bool
    default_value: Optional[str] = None


@dataclass
class CSharpClass:
    name: str
    base_class: str
    file_path: str
    namespace: str
    methods: List[CSharpMethod] = field(default_factory=list)
    fields: List[CSharpField] = field(default_factory=list)
    is_monobehaviour: bool = False
    detected_patterns: List[str] = field(default_factory=list)


@dataclass
class InputBinding:
    key_or_axis: str
    action_description: str
    source_file: str
    source_line: int
    input_type: str  # "key", "axis", "button", "mouse"


@dataclass
class TestSuggestion:
    test_id: str
    name: str
    description: str
    category: str  # "movement", "combat", "ui", "interaction"
    confidence: float
    source_class: str
    steps: List[Dict[str, Any]]
    verify: List[Dict[str, Any]]


# ── Regex patterns for C# parsing ─────────────────────────────────────

_RE_CLASS = re.compile(
    r'(?:public|private|internal|protected)?\s*(?:abstract|sealed|static|partial)?\s*'
    r'class\s+(\w+)\s*(?::\s*([\w\s,.<>]+))?',
    re.MULTILINE,
)
_RE_METHOD = re.compile(
    r'(?:public|private|protected|internal|override|virtual|static|async|\s)*'
    r'([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)',
    re.MULTILINE,
)
_RE_FIELD = re.compile(
    r'(?:\[SerializeField\]\s*)?'
    r'(public|private|protected)\s+'
    r'([\w<>\[\]]+)\s+(\w+)\s*(?:=\s*([^;]+))?;',
    re.MULTILINE,
)
_RE_SERIALIZE = re.compile(r'\[SerializeField\]')
_RE_NAMESPACE = re.compile(r'namespace\s+([\w.]+)')

# Input detection patterns
_RE_GETKEY = re.compile(r'Input\.GetKey(?:Down|Up)?\s*\(\s*(?:KeyCode\.)?["\']?(\w+)["\']?\s*\)')
_RE_GETAXIS = re.compile(r'Input\.GetAxis(?:Raw)?\s*\(\s*"(\w+)"\s*\)')
_RE_GETBUTTON = re.compile(r'Input\.GetButton(?:Down|Up)?\s*\(\s*"(\w+)"\s*\)')
_RE_GETMOUSE = re.compile(r'Input\.GetMouseButton(?:Down|Up)?\s*\(\s*(\d+)\s*\)')
_RE_ONCLICK = re.compile(r'\.onClick\.AddListener')
_RE_UI_PANEL = re.compile(r'(?:panel|menu|dialog|popup|window|canvas|hud|overlay)', re.IGNORECASE)
_RE_HEALTH = re.compile(r'(?:health|hp|hitPoints|damage|heal|die|death|respawn|alive)', re.IGNORECASE)
_RE_MOVEMENT = re.compile(r'(?:move|walk|run|sprint|jump|velocity|speed|transform\.Translate|CharacterController|Rigidbody)', re.IGNORECASE)
_RE_COMBAT = re.compile(r'(?:attack|weapon|shoot|bullet|projectile|cooldown|combo|block|parry|dodge)', re.IGNORECASE)


class CodeReader:
    """Analyze a Unity C# codebase to extract game logic patterns."""

    def __init__(self, project_path: str) -> None:
        self._root = Path(project_path)
        self._classes: List[CSharpClass] = []
        self._inputs: List[InputBinding] = []
        self._suggestions: List[TestSuggestion] = []

    @property
    def classes(self) -> List[CSharpClass]:
        return list(self._classes)

    @property
    def inputs(self) -> List[InputBinding]:
        return list(self._inputs)

    @property
    def suggestions(self) -> List[TestSuggestion]:
        return list(self._suggestions)

    # ── Scanning ──────────────────────────────────────────────────────

    def scan(self) -> "CodeReader":
        """Recursively scan for .cs files and analyze them."""
        cs_files = self._find_cs_files()
        log.info("Found %d C# files in %s", len(cs_files), self._root)

        for fp in cs_files:
            try:
                source = fp.read_text(encoding="utf-8", errors="replace")
                self._parse_file(str(fp), source)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", fp, exc)

        self._detect_patterns()
        self._generate_suggestions()
        log.info("Analysis complete: %d classes, %d inputs, %d suggestions",
                 len(self._classes), len(self._inputs), len(self._suggestions))
        return self

    def _find_cs_files(self) -> List[Path]:
        """Find all .cs files, excluding Editor and Plugins folders."""
        results = []
        for root, dirs, files in os.walk(self._root):
            # Skip common non-game folders
            dirs[:] = [d for d in dirs if d not in (
                "Editor", "Plugins", "ThirdParty", "Packages",
                ".git", "Library", "Temp", "obj", "bin",
            )]
            for f in files:
                if f.endswith(".cs"):
                    results.append(Path(root) / f)
        return results

    # ── Parsing ───────────────────────────────────────────────────────

    def _parse_file(self, path: str, source: str) -> None:
        lines = source.split("\n")

        # Namespace
        ns_match = _RE_NAMESPACE.search(source)
        namespace = ns_match.group(1) if ns_match else ""

        # Classes
        for m in _RE_CLASS.finditer(source):
            cls_name = m.group(1)
            base = m.group(2) or ""
            is_mono = any(b.strip() in ("MonoBehaviour", "NetworkBehaviour")
                         for b in base.split(","))

            cls = CSharpClass(
                name=cls_name,
                base_class=base.strip(),
                file_path=path,
                namespace=namespace,
                is_monobehaviour=is_mono,
            )

            # Methods
            for mm in _RE_METHOD.finditer(source):
                ret = mm.group(1)
                name = mm.group(2)
                params = mm.group(3).strip()
                line_no = source[:mm.start()].count("\n") + 1
                # Grab a snippet of the body (up to 200 chars after the match)
                body_start = source.find("{", mm.end())
                body_snippet = ""
                if body_start != -1:
                    body_snippet = source[body_start:body_start + 200]

                cls.methods.append(CSharpMethod(
                    name=name,
                    return_type=ret,
                    parameters=[p.strip() for p in params.split(",") if p.strip()],
                    body_snippet=body_snippet,
                    line_number=line_no,
                ))

            # Fields
            for fm in _RE_FIELD.finditer(source):
                is_serialized = bool(_RE_SERIALIZE.search(
                    source[max(0, fm.start() - 30):fm.start()]))
                cls.fields.append(CSharpField(
                    name=fm.group(3),
                    field_type=fm.group(2),
                    access=fm.group(1),
                    is_serialized=is_serialized or fm.group(1) == "public",
                    default_value=fm.group(4),
                ))

            self._classes.append(cls)

        # Input bindings
        self._extract_inputs(path, source)

    def _extract_inputs(self, path: str, source: str) -> None:
        for m in _RE_GETKEY.finditer(source):
            line = source[:m.start()].count("\n") + 1
            self._inputs.append(InputBinding(
                key_or_axis=m.group(1), action_description="Key input",
                source_file=path, source_line=line, input_type="key",
            ))
        for m in _RE_GETAXIS.finditer(source):
            line = source[:m.start()].count("\n") + 1
            self._inputs.append(InputBinding(
                key_or_axis=m.group(1), action_description="Axis input",
                source_file=path, source_line=line, input_type="axis",
            ))
        for m in _RE_GETBUTTON.finditer(source):
            line = source[:m.start()].count("\n") + 1
            self._inputs.append(InputBinding(
                key_or_axis=m.group(1), action_description="Button input",
                source_file=path, source_line=line, input_type="button",
            ))
        for m in _RE_GETMOUSE.finditer(source):
            line = source[:m.start()].count("\n") + 1
            btn = {"0": "left", "1": "right", "2": "middle"}.get(m.group(1), m.group(1))
            self._inputs.append(InputBinding(
                key_or_axis=btn, action_description="Mouse button",
                source_file=path, source_line=line, input_type="mouse",
            ))

    # ── Pattern detection ─────────────────────────────────────────────

    def _detect_patterns(self) -> None:
        for cls in self._classes:
            source = ""
            for method in cls.methods:
                source += method.body_snippet + " "
            all_text = source + " ".join(f.name for f in cls.fields)

            if _RE_MOVEMENT.search(all_text):
                cls.detected_patterns.append("movement")
            if _RE_HEALTH.search(all_text):
                cls.detected_patterns.append("health")
            if _RE_COMBAT.search(all_text):
                cls.detected_patterns.append("combat")
            if _RE_UI_PANEL.search(cls.name) or _RE_ONCLICK.search(source):
                cls.detected_patterns.append("ui")

    # ── Test suggestion generation ────────────────────────────────────

    def _generate_suggestions(self) -> None:
        counter = 1

        # Movement-based suggestions from input bindings
        key_map = {
            "W": ("Move Forward", "z", "greater_than"),
            "A": ("Strafe Left", "x", "less_than"),
            "S": ("Move Backward", "z", "less_than"),
            "D": ("Strafe Right", "x", "greater_than"),
            "Space": ("Jump", "y", "greater_than"),
        }

        seen_keys = set()
        for inp in self._inputs:
            key_upper = inp.key_or_axis.upper() if inp.input_type == "key" else ""
            if key_upper in key_map and key_upper not in seen_keys:
                seen_keys.add(key_upper)
                label, axis, op = key_map[key_upper]
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=label,
                    description=f"Press {inp.key_or_axis} — detected in {Path(inp.source_file).name}:{inp.source_line}",
                    category="movement",
                    confidence=0.85,
                    source_class=inp.source_file,
                    steps=[{"action": "key_press", "key": inp.key_or_axis.lower(), "duration": 1.0}],
                    verify=[{
                        "type": "game_state",
                        "field": f"player.position.{axis}",
                        "operator": op,
                        "expected": 0.5 if op == "greater_than" else -0.5,
                        "tolerance": 0.2,
                    }],
                ))
                counter += 1

        # Axis-based suggestions
        for inp in self._inputs:
            if inp.input_type == "axis" and inp.key_or_axis in ("Horizontal", "Vertical"):
                axis_key = "d" if inp.key_or_axis == "Horizontal" else "w"
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=f"{inp.key_or_axis} Axis Movement",
                    description=f"Input.GetAxis(\"{inp.key_or_axis}\") in {Path(inp.source_file).name}",
                    category="movement",
                    confidence=0.8,
                    source_class=inp.source_file,
                    steps=[{"action": "key_press", "key": axis_key, "duration": 1.0}],
                    verify=[{"type": "game_state", "field": "player.position.z", "operator": "not_equals", "expected": 0}],
                ))
                counter += 1

        # Health/combat suggestions
        for cls in self._classes:
            if "health" in cls.detected_patterns:
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=f"Health System ({cls.name})",
                    description=f"Health-related logic detected in {cls.name}",
                    category="combat",
                    confidence=0.7,
                    source_class=cls.name,
                    steps=[
                        {"action": "key_press", "key": "w", "duration": 3.0},
                    ],
                    verify=[
                        {"type": "game_state", "field": "player.hp", "operator": "less_than", "expected": 100},
                    ],
                ))
                counter += 1

            if "combat" in cls.detected_patterns:
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=f"Combat Mechanic ({cls.name})",
                    description=f"Attack/combat logic detected in {cls.name}",
                    category="combat",
                    confidence=0.7,
                    source_class=cls.name,
                    steps=[
                        {"action": "mouse_click", "x": 540, "y": 360, "button": "left"},
                    ],
                    verify=[
                        {"type": "animation", "field": "player.currentAnimation", "expected": "attack"},
                    ],
                ))
                counter += 1

            if "ui" in cls.detected_patterns:
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=f"UI Panel ({cls.name})",
                    description=f"UI panel detected: {cls.name}",
                    category="ui",
                    confidence=0.65,
                    source_class=cls.name,
                    steps=[
                        {"action": "key_press", "key": "escape"},
                    ],
                    verify=[
                        {"type": "ui_active", "element": cls.name, "timeout": 1.0},
                    ],
                ))
                counter += 1

        # Mouse click suggestions
        for inp in self._inputs:
            if inp.input_type == "mouse":
                self._suggestions.append(TestSuggestion(
                    test_id=f"AUTO_{counter:03d}",
                    name=f"Mouse {inp.key_or_axis.title()} Click Action",
                    description=f"Mouse input detected in {Path(inp.source_file).name}:{inp.source_line}",
                    category="interaction",
                    confidence=0.6,
                    source_class=inp.source_file,
                    steps=[
                        {"action": "mouse_click", "x": 540, "y": 360, "button": inp.key_or_axis},
                    ],
                    verify=[
                        {"type": "animation", "field": "player.currentAnimation", "operator": "not_equals", "expected": "idle"},
                    ],
                ))
                counter += 1

    # ── Export ─────────────────────────────────────────────────────────

    def export_suggestions_yaml(self, output_path: str) -> str:
        """Write auto-generated test cases to a YAML file."""
        suite = {
            "test_suite": "Auto-Generated Tests",
            "description": f"Tests auto-generated from codebase analysis of {self._root}",
            "setup": {"scene": "Level_01", "wait_for_load": 2.0},
            "test_cases": [],
        }
        for s in self._suggestions:
            suite["test_cases"].append({
                "id": s.test_id,
                "name": s.name,
                "description": s.description,
                "steps": s.steps,
                "verify": s.verify,
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(suite, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        log.info("Exported %d auto-generated test cases → %s", len(self._suggestions), output_path)
        return output_path

    # ── Summary for display ───────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict for display."""
        mono_classes = [c for c in self._classes if c.is_monobehaviour]
        patterns: Dict[str, int] = {}
        for c in self._classes:
            for p in c.detected_patterns:
                patterns[p] = patterns.get(p, 0) + 1

        return {
            "total_classes": len(self._classes),
            "monobehaviour_count": len(mono_classes),
            "input_bindings": len(self._inputs),
            "detected_patterns": patterns,
            "test_suggestions": len(self._suggestions),
            "classes": [
                {
                    "name": c.name,
                    "base": c.base_class,
                    "file": c.file_path,
                    "methods": len(c.methods),
                    "fields": len(c.fields),
                    "patterns": c.detected_patterns,
                }
                for c in self._classes
            ],
            "inputs": [
                {
                    "key": i.key_or_axis,
                    "type": i.input_type,
                    "file": Path(i.source_file).name,
                    "line": i.source_line,
                }
                for i in self._inputs
            ],
        }
