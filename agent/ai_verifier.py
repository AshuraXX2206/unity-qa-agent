"""LLM-powered test verification using Anthropic Claude with vision."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a game QA expert. Given the following information, determine whether the \
test case PASSED or FAILED.

## Test Case
- **ID**: {test_id}
- **Name**: {test_name}
- **Description**: {test_description}
- **Expected behaviour**: {expected}

## Current Game State (JSON)
```json
{game_state_json}
```

## Screenshot
See the attached image.

---

Respond with **only** a JSON object (no markdown fences):
{{"result": "PASS" or "FAIL", "confidence": 0.0-1.0, "reason": "..."}}
"""


@dataclass
class VerifyResult:
    """Result returned by :meth:`AIVerifier.verify_with_llm`."""

    result: str       # "PASS" | "FAIL"
    confidence: float
    reason: str


class AIVerifier:
    """Verify complex test outcomes by calling Claude with vision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self._client: Optional[Any] = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception as exc:
                log.error("Failed to initialise Anthropic client: %s", exc)
                raise
        return self._client

    # ── public API ────────────────────────────────────────────────────

    def verify_with_llm(
        self,
        test_case: Dict[str, Any],
        screenshot: Image.Image,
        game_state: Dict[str, Any],
    ) -> VerifyResult:
        """Send screenshot + game state to Claude and return a verdict.

        Only call this when:
        - The test case has ``ai_verify: true`` in its YAML definition, **or**
        - Rule-based verification returned an ambiguous result.
        """
        client = self._ensure_client()

        # Encode screenshot to base64 PNG
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        prompt_text = _PROMPT_TEMPLATE.format(
            test_id=test_case.get("id", "?"),
            test_name=test_case.get("name", "?"),
            test_description=test_case.get("description", ""),
            expected=json.dumps(test_case.get("verify", []), indent=2),
            game_state_json=json.dumps(game_state, indent=2),
        )

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt_text,
                            },
                        ],
                    }
                ],
            )
            raw_text = response.content[0].text.strip()
            return self._parse_response(raw_text)
        except Exception as exc:
            log.error("LLM verification failed: %s", exc)
            return VerifyResult(result="FAIL", confidence=0.0,
                                reason=f"LLM call failed: {exc}")

    # ── internal ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> VerifyResult:
        """Parse the JSON response from Claude."""
        # Strip markdown fences if present
        cleaned = text
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                l for l in lines if not l.startswith("```")
            )
        try:
            data = json.loads(cleaned)
            return VerifyResult(
                result=data.get("result", "FAIL").upper(),
                confidence=float(data.get("confidence", 0.0)),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Could not parse LLM response: %s — %s", exc, text[:200])
            # Heuristic fallback
            upper = text.upper()
            if "PASS" in upper:
                return VerifyResult(result="PASS", confidence=0.5,
                                    reason=f"Heuristic parse: {text[:120]}")
            return VerifyResult(result="FAIL", confidence=0.5,
                                reason=f"Heuristic parse: {text[:120]}")
