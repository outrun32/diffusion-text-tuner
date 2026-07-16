"""Combinatoric style generator with per-content-type constraints."""

import random

from .config import COLORS, EFFECTS, FONTS, SIZES, STYLE_CONSTRAINTS


class StyleGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def sample(self, content_type: str | None = None) -> dict[str, str]:
        c = STYLE_CONSTRAINTS.get(content_type, {})
        return {
            "font": self.rng.choice(c.get("fonts", FONTS)),
            "color": self.rng.choice(c.get("colors", COLORS)),
            "effect": self.rng.choice(c.get("effects", EFFECTS)),
            "size": self.rng.choice(c.get("sizes", SIZES)),
        }
