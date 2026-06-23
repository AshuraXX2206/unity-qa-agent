"""LLM-powered test verification using Google Gemini with vision."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Dict, Optional

from PIL import Image

from agent.ai_verifier import VerifyResult

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


class GeminiVerifier:
    """Verify complex test outcomes by calling Gemini with vision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client: Optional[Any] = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self._model)
            except ImportError:
                # Fallback: use REST API directly
                log.info("google-generativeai not installed, using REST API")
                self._client = "rest"
            except Exception as exc:
                log.error("Failed to initialise Gemini client: %s", exc)
                raise
        return self._client

    def verify_with_llm(
        self,
        test_case: Dict[str, Any],
        screenshot: Image.Image,
        game_state: Dict[str, Any],
    ) -> VerifyResult:
        """Send screenshot + game state to Gemini and return a verdict."""
        client = self._ensure_client()

        prompt_text = _PROMPT_TEMPLATE.format(
            test_id=test_case.get("id", "?"),
            test_name=test_case.get("name", "?"),
            test_description=test_case.get("description", ""),
            expected=json.dumps(test_case.get("verify", []), indent=2),
            game_state_json=json.dumps(game_state, indent=2),
        )

        # Encode screenshot
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        try:
            if client == "rest":
                return self._verify_rest(prompt_text, img_bytes)
            else:
                return self._verify_sdk(client, prompt_text, screenshot)
        except Exception as exc:
            log.error("Gemini verification failed: %s", exc)
            return VerifyResult(result="FAIL", confidence=0.0,
                                reason=f"Gemini call failed: {exc}")

    def _verify_sdk(self, client: Any, prompt: str, image: Image.Image) -> VerifyResult:
        """Use google-generativeai SDK."""
        response = client.generate_content([prompt, image])
        raw = response.text.strip()
        return self._parse_response(raw)

    def _verify_rest(self, prompt: str, img_bytes: bytes) -> VerifyResult:
        """Fallback REST API call."""
        import urllib.request

        img_b64 = base64.b64encode(img_bytes).decode()
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": img_b64,
                    }},
                ],
            }],
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:generateContent?key={self._api_key}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(text: str) -> VerifyResult:
        cleaned = text
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(l for l in lines if not l.startswith("```"))
        try:
            data = json.loads(cleaned)
            return VerifyResult(
                result=data.get("result", "FAIL").upper(),
                confidence=float(data.get("confidence", 0.0)),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, ValueError):
            upper = text.upper()
            if "PASS" in upper:
                return VerifyResult(result="PASS", confidence=0.5,
                                    reason=f"Heuristic: {text[:120]}")
            return VerifyResult(result="FAIL", confidence=0.5,
                                reason=f"Heuristic: {text[:120]}")
