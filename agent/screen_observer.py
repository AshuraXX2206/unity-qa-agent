"""Screen capture and visual analysis utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image

log = logging.getLogger(__name__)


class ScreenObserver:
    """Capture the screen and perform lightweight vision analysis."""

    def __init__(self) -> None:
        self._sct = mss.mss()
        monitor = self._sct.monitors[1]  # primary monitor
        self._width = monitor["width"]
        self._height = monitor["height"]
        log.info("ScreenObserver initialised — display %dx%d", self._width, self._height)

    # ── capture ───────────────────────────────────────────────────────

    def capture_screen(self) -> Image.Image:
        """Return a PIL Image of the full primary monitor."""
        raw = self._sct.grab(self._sct.monitors[1])
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def capture_region(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """Capture a rectangular region of the screen."""
        region = {"left": x, "top": y, "width": w, "height": h}
        raw = self._sct.grab(region)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    # ── OCR ───────────────────────────────────────────────────────────

    def extract_text_from_screen(self, image: Optional[Image.Image] = None) -> str:
        """Run Tesseract OCR on the given image (or a fresh screenshot)."""
        if image is None:
            image = self.capture_screen()
        return pytesseract.image_to_string(image).strip()

    # ── template matching ─────────────────────────────────────────────

    def find_image_on_screen(
        self,
        template_path: str,
        screenshot: Optional[Image.Image] = None,
        confidence: float = 0.8,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find *template_path* on screen via OpenCV template matching.

        Returns ``(x, y, w, h)`` of the best match if above *confidence*,
        otherwise ``None``.
        """
        if screenshot is None:
            screenshot = self.capture_screen()

        screen_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        template_cv = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template_cv is None:
            log.warning("Template not found: %s", template_path)
            return None

        result = cv2.matchTemplate(screen_cv, template_cv, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= confidence:
            th, tw = template_cv.shape[:2]
            return (max_loc[0], max_loc[1], tw, th)
        return None

    # ── pixel colour ──────────────────────────────────────────────────

    def get_pixel_color(
        self, x: int, y: int, image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, int]:
        """Return the (R, G, B) colour of pixel at *(x, y)*."""
        if image is None:
            image = self.capture_screen()
        return image.getpixel((x, y))[:3]

    # ── comparison ────────────────────────────────────────────────────

    @staticmethod
    def compare_screenshots(
        img1: Image.Image,
        img2: Image.Image,
        threshold: float = 0.95,
    ) -> float:
        """Return structural-similarity score (0–1) between two images.

        Uses a fast normalised cross-correlation rather than full SSIM for speed.
        """
        a = cv2.cvtColor(np.array(img1.resize((640, 480))), cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(np.array(img2.resize((640, 480))), cv2.COLOR_RGB2GRAY).astype(np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        score = float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))
        return max(0.0, min(score, 1.0))

    # ── helpers ───────────────────────────────────────────────────────

    def save_screenshot(self, path: str, image: Optional[Image.Image] = None) -> str:
        """Save a screenshot to *path* and return the absolute path."""
        if image is None:
            image = self.capture_screen()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        log.debug("Screenshot saved → %s", path)
        return str(Path(path).resolve())
