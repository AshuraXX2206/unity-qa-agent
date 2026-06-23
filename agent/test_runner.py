"""Load YAML test suites and execute them against a running Unity game."""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from agent.ai_verifier import AIVerifier
from agent.bridge_client import BridgeClient
from agent.input_executor import InputExecutor
from agent.persona import Persona, get_persona, pick_random_key
from agent.report_generator import CaseResult, TestResult
from agent.screen_observer import ScreenObserver

log = logging.getLogger(__name__)

_OPS: Dict[str, Callable[..., bool]] = {
    "equals":       operator.eq,
    "not_equals":   operator.ne,
    "greater_than": operator.gt,
    "less_than":    operator.lt,
    "gte":          operator.ge,
    "lte":          operator.le,
}

DEFAULT_TIMEOUT = 5.0  # seconds


class TestRunner:
    """Orchestrates loading, executing, and verifying test cases."""

    def __init__(
        self,
        bridge: BridgeClient,
        observer: ScreenObserver,
        executor: InputExecutor,
        persona: Persona,
        ai_verifier: Optional[AIVerifier] = None,
        screenshot_dir: str = "reports/screenshots",
    ) -> None:
        self._bridge = bridge
        self._observer = observer
        self._executor = executor
        self._persona = persona
        self._ai = ai_verifier
        self._screenshot_dir = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────

    def run_suite(self, yaml_path: str, case_id: Optional[str] = None) -> TestResult:
        """Load a YAML suite and run all (or one) test case(s).

        Parameters
        ----------
        yaml_path:
            Path to the YAML test-suite file.
        case_id:
            If given, only execute the test case with this ``id``.
        """
        suite = self._load_suite(yaml_path)
        suite_name = suite.get("test_suite", Path(yaml_path).stem)
        result = TestResult(suite=suite_name, persona=self._persona.name)

        setup = suite.get("setup", {})
        self._run_setup(setup)

        cases = suite.get("test_cases", [])
        for tc in cases:
            if case_id and tc.get("id") != case_id:
                continue
            cr = self.run_single(tc)
            result.add(cr)

        result.finish()
        return result

    def run_single(self, test_case: Dict[str, Any]) -> CaseResult:
        """Execute a single test-case dict and return its result."""
        tc_id = test_case.get("id", "?")
        tc_name = test_case.get("name", "Unnamed")
        log.info("▶ Running %s — %s", tc_id, tc_name)

        cr = CaseResult(id=tc_id, name=tc_name)
        start = time.monotonic()

        # Screenshot before
        self._save_screenshot(f"{tc_id}_before")

        try:
            steps = test_case.get("steps", [])
            self.execute_steps(steps)
            cr.steps_executed = len(steps)

            # Screenshot after
            self._save_screenshot(f"{tc_id}_after")

            # Verification
            verifications = test_case.get("verify", [])
            game_state = self._bridge.get_game_state() or {}
            all_passed = True
            details: Dict[str, Any] = {}

            for v in verifications:
                vtype = v.get("type", "")
                timeout = v.get("timeout", DEFAULT_TIMEOUT)
                ok = self._verify_with_timeout(v, game_state, timeout)
                details[f"{vtype}_check"] = ok
                if not ok:
                    all_passed = False

            # AI verification if requested
            use_ai = test_case.get("ai_verify", False)
            if (use_ai or not all_passed) and self._ai is not None:
                screenshot = self._observer.capture_screen()
                ai_result = self._ai.verify_with_llm(test_case, screenshot, game_state)
                details["ai_verify"] = {
                    "result": ai_result.result,
                    "confidence": ai_result.confidence,
                    "reason": ai_result.reason,
                }
                if use_ai:
                    all_passed = ai_result.result == "PASS"

            cr.status = "PASS" if all_passed else "FAIL"
            cr.verify_details = details
            if not all_passed:
                failed_checks = [k for k, v in details.items() if v is False]
                cr.error = f"Failed checks: {', '.join(failed_checks)}" if failed_checks else None

        except Exception as exc:
            cr.status = "FAIL"
            cr.error = str(exc)
            log.exception("Test %s crashed", tc_id)

        cr.duration_seconds = time.monotonic() - start
        log.info("  %s %s (%.1fs)", cr.status, tc_name, cr.duration_seconds)
        return cr

    # ── step execution ────────────────────────────────────────────────

    def execute_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Execute a sequence of action steps."""
        for step in steps:
            # Persona deviation
            if self._persona.should_random_action():
                rk = pick_random_key()
                log.debug("Persona deviation → random key %s", rk)
                self._executor.press_key(rk)

            action = step.get("action", "")
            if action == "key_press":
                duration = step.get("duration", 0)
                if duration:
                    self._executor.hold_key(step["key"], duration)
                else:
                    self._executor.press_key(step["key"])
            elif action == "key_hold":
                self._executor.hold_key(step["key"], step.get("duration", 0.5))
            elif action == "key_combo":
                keys = step.get("keys", [])
                self._executor.press_combo(*keys)
            elif action == "mouse_click":
                self._executor.mouse_click(
                    step.get("x", 0), step.get("y", 0),
                    button=step.get("button", "left"),
                )
            elif action == "mouse_move":
                self._executor.mouse_move(
                    step.get("x", 0), step.get("y", 0),
                    duration=step.get("duration", 0.2),
                )
            elif action == "mouse_drag":
                self._executor.mouse_drag(
                    step["x1"], step["y1"], step["x2"], step["y2"],
                    duration=step.get("duration", 0.3),
                )
            elif action == "type_text":
                self._executor.type_text(step.get("text", ""))
            elif action == "scroll":
                self._executor.scroll(
                    step.get("x", 0), step.get("y", 0), step.get("amount", 3),
                )
            elif action == "wait":
                wait_time = step.get("duration", 1.0)
                log.debug("Waiting %.2fs", wait_time)
                time.sleep(wait_time)
            else:
                log.warning("Unknown action: %s", action)

            # Persona input delay
            time.sleep(self._persona.input_delay)

    # ── verification ──────────────────────────────────────────────────

    def verify_condition(
        self,
        verify: Dict[str, Any],
        game_state: Dict[str, Any],
    ) -> bool:
        """Evaluate a single verify clause against current state."""
        vtype = verify.get("type", "")

        if vtype == "game_state":
            return self._verify_game_state(verify, game_state)
        if vtype == "animation":
            return self._verify_animation(verify, game_state)
        if vtype == "ui_active":
            return self._verify_ui_active(verify, game_state)
        if vtype == "screen_text":
            return self._verify_screen_text(verify)
        if vtype == "screen_image":
            return self._verify_screen_image(verify)

        log.warning("Unknown verify type: %s", vtype)
        return False

    def _verify_with_timeout(
        self,
        verify: Dict[str, Any],
        game_state: Dict[str, Any],
        timeout: float,
    ) -> bool:
        """Retry verification until it passes or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fresh = self._bridge.get_game_state() or game_state
            if self.verify_condition(verify, fresh):
                return True
            time.sleep(0.2)
        return False

    # ── verify implementations ────────────────────────────────────────

    def _verify_game_state(self, v: Dict[str, Any], state: Dict[str, Any]) -> bool:
        field = v.get("field", "")
        expected = v.get("expected")
        op_name = v.get("operator", "equals")
        tolerance = v.get("tolerance", 0)
        actual = self._resolve_field(state, field)
        if actual is None:
            return False
        op_fn = _OPS.get(op_name, operator.eq)
        try:
            if tolerance and isinstance(actual, (int, float)):
                return op_fn(actual, expected) or abs(actual - expected) <= tolerance
            return op_fn(actual, expected)
        except TypeError:
            return False

    def _verify_animation(self, v: Dict[str, Any], state: Dict[str, Any]) -> bool:
        expected = v.get("expected", "")
        actual = self._resolve_field(state, v.get("field", "player.currentAnimation"))
        return str(actual).lower() == str(expected).lower()

    def _verify_ui_active(self, v: Dict[str, Any], state: Dict[str, Any]) -> bool:
        element = v.get("element", "")
        active = state.get("activeUI", [])
        return element in active

    def _verify_screen_text(self, v: Dict[str, Any]) -> bool:
        target = v.get("contains", "")
        text = self._observer.extract_text_from_screen()
        return target.lower() in text.lower()

    def _verify_screen_image(self, v: Dict[str, Any]) -> bool:
        template = v.get("template", "")
        confidence = v.get("confidence", 0.8)
        match = self._observer.find_image_on_screen(template, confidence=confidence)
        return match is not None

    # ── helpers ───────────────────────────────────────────────────────

    def _run_setup(self, setup: Dict[str, Any]) -> None:
        """Handle suite-level setup: wait for scene load, etc."""
        scene = setup.get("scene", "")
        wait = setup.get("wait_for_load", 0)
        if scene:
            log.info("Expecting scene: %s", scene)
            state = self._bridge.get_game_state()
            if state and state.get("scene") != scene:
                log.warning("Current scene '%s' != expected '%s'",
                            state.get("scene"), scene)
        if wait > 0:
            log.info("Waiting %.1fs for scene to load…", wait)
            time.sleep(wait)

    @staticmethod
    def _load_suite(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _resolve_field(data: Dict[str, Any], field: str) -> Any:
        """Resolve a dotted field path like ``player.position.z``."""
        parts = field.split(".")
        current: Any = data
        for p in parts:
            if isinstance(current, dict):
                current = current.get(p)
            else:
                return None
        return current

    def _save_screenshot(self, label: str) -> None:
        path = self._screenshot_dir / f"{label}_{int(time.time())}.png"
        self._observer.save_screenshot(str(path))

    # ── suite listing / validation ────────────────────────────────────

    @staticmethod
    def list_cases(yaml_path: str) -> List[Dict[str, str]]:
        """Return a list of ``{id, name, description}`` for every case in the suite."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            suite = yaml.safe_load(f)
        return [
            {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "Unnamed"),
                "description": tc.get("description", ""),
            }
            for tc in suite.get("test_cases", [])
        ]

    @staticmethod
    def validate_suite(yaml_path: str) -> List[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: List[str] = []
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                suite = yaml.safe_load(f)
        except Exception as exc:
            return [f"YAML parse error: {exc}"]

        if not isinstance(suite, dict):
            return ["Root must be a mapping"]
        if "test_suite" not in suite:
            errors.append("Missing 'test_suite' field")
        cases = suite.get("test_cases", [])
        if not cases:
            errors.append("No test_cases defined")
        ids_seen: set[str] = set()
        for i, tc in enumerate(cases):
            tc_id = tc.get("id", "")
            if not tc_id:
                errors.append(f"test_cases[{i}]: missing 'id'")
            elif tc_id in ids_seen:
                errors.append(f"Duplicate test case id: {tc_id}")
            ids_seen.add(tc_id)
            if "steps" not in tc:
                errors.append(f"{tc_id}: missing 'steps'")
            if "verify" not in tc:
                errors.append(f"{tc_id}: missing 'verify'")
        return errors
