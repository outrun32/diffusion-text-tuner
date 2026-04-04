"""Text generator with character-coverage balancing.

Handles tiers 1-2 algorithmically and produces must-include word lists
for LLM-generated tiers 3-5.
"""

import json
import random
import math
from collections import Counter
from pathlib import Path

from .config import (
    CHAR_RENDER_BOOST,
    CYRILLIC_WORD_RE,
    CYRILLIC_LOWER,
    RARE_CHARS,
    TIER_WEIGHTS,
    TIER_OVERRIDES,
    CASE_WEIGHTS,
)


class TextGenerator:

    def __init__(self, freq_dict_path: str, thematic_path: str | None = None,
                 seed: int = 42):
        self.rng = random.Random(seed)
        self.char_counts: Counter = Counter()

        self.words = self._load_freq_dict(freq_dict_path)
        self.thematic = self._load_thematic(thematic_path) if thematic_path else {}

        self._precompute_rarity()
        self._preindex_candidates()

        # Weight cache: refreshed every N coverage updates
        self._weight_cache: dict[tuple, tuple[list, list]] = {}
        self._updates_since_refresh: int = 0
        self._REFRESH_INTERVAL: int = 200

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_freq_dict(path: str) -> list[tuple[str, int]]:
        words = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                word, freq = parts[0], int(parts[1])
                if CYRILLIC_WORD_RE.match(word) and 2 <= len(word) <= 20:
                    words.append((word, freq))
        return words

    @staticmethod
    def _load_thematic(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Rarity / coverage helpers
    # ------------------------------------------------------------------

    def _precompute_rarity(self):
        """Compute per-character rarity based on how many dict words contain it."""
        char_word_count: Counter = Counter()
        for word, _ in self.words:
            for ch in set(word.lower()):
                if ch in CYRILLIC_LOWER:
                    char_word_count[ch] += 1

        max_count = max(char_word_count.values()) if char_word_count else 1
        self.char_rarity = {
            ch: max_count / char_word_count.get(ch, 1)
            for ch in CYRILLIC_LOWER
        }

    def _preindex_candidates(self):
        """Pre-filter word lists by common length ranges used in sampling."""
        self._candidate_index: dict[tuple, list[tuple[str, int]]] = {}
        for min_l, max_l in [(2, 8), (2, 20), (3, 10), (3, 12)]:
            self._candidate_index[(min_l, max_l)] = [
                (w, f) for w, f in self.words if min_l <= len(w) <= max_l
            ]

    def _get_candidates(self, min_len: int, max_len: int | None) -> list[tuple[str, int]]:
        key = (min_len, max_len or 20)
        if key in self._candidate_index:
            return self._candidate_index[key]
        cands = [(w, f) for w, f in self.words if len(w) >= min_len]
        if max_len:
            cands = [(w, f) for w, f in cands if len(w) <= max_len]
        return cands or self.words[:1000]

    def _compute_weights(self, candidates: list[tuple[str, int]]) -> list[float]:
        """Vectorised weight computation for a candidate list.

        Three multiplicative factors:
          1. Corpus rarity  — inverse of how many dict words contain this char
          2. Coverage boost — dynamic boost for currently under-represented chars
          3. Render boost   — static boost for chars that FLUX renders poorly
                             (unique Cyrillic glyphs with no Latin equivalent)
        """
        cyrillic_set = frozenset(CYRILLIC_LOWER)
        total = sum(self.char_counts.values()) or 1
        expected = total / len(CYRILLIC_LOWER)
        char_rarity = self.char_rarity
        char_counts = self.char_counts
        render_boost = CHAR_RENDER_BOOST

        weights = []
        for w, _ in candidates:
            chars = set(w.lower()) & cyrillic_set
            if not chars:
                weights.append(0.01)
                continue

            # Factor 1: corpus rarity
            rarity = max(char_rarity.get(ch, 1.0) for ch in chars)

            # Factor 2: dynamic coverage boost
            coverage = 1.0
            for ch in chars:
                actual = char_counts.get(ch, 0)
                if actual < expected * 0.5:
                    coverage *= 2.0
                elif actual < expected * 0.7:
                    coverage *= 1.6
                elif actual < expected:
                    coverage *= 1.2

            # Factor 3: render difficulty (unique Cyrillic glyphs)
            render = max((render_boost.get(ch, 1.0) for ch in chars), default=1.0)

            weights.append(rarity * coverage * render)
        return weights

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_words(self, n: int = 1, min_len: int = 2,
                      max_len: int | None = None) -> list[str]:
        """Sample *n* words from freq dict with coverage-aware weighting."""
        key = (min_len, max_len or 20)
        needs_refresh = (
            key not in self._weight_cache
            or self._updates_since_refresh >= self._REFRESH_INTERVAL
        )
        if needs_refresh:
            candidates = self._get_candidates(min_len, max_len)
            weights = self._compute_weights(candidates)
            self._weight_cache[key] = (candidates, weights)
            if needs_refresh and self._updates_since_refresh >= self._REFRESH_INTERVAL:
                # Full refresh: clear all keys so other length ranges refresh too
                old = self._weight_cache.pop(key)
                self._weight_cache.clear()
                self._weight_cache[key] = old
                self._updates_since_refresh = 0

        candidates, weights = self._weight_cache[key]
        chosen = self.rng.choices(candidates, weights=weights, k=n)
        return [w for w, _ in chosen]

    def _apply_case(self, text: str, case: str) -> str:
        if case == "upper":
            return text.upper()
        if case == "title":
            return text.title()
        if case == "lower":
            return text.lower()
        return text  # "mixed" kept as-is

    def sample_case(self) -> str:
        cases = list(CASE_WEIGHTS.keys())
        weights = list(CASE_WEIGHTS.values())
        return self.rng.choices(cases, weights, k=1)[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_tier(self, content_type: str | None = None) -> int:
        tw = TIER_OVERRIDES.get(content_type, TIER_WEIGHTS)
        tiers = list(tw.keys())
        weights = list(tw.values())
        return self.rng.choices(tiers, weights, k=1)[0]

    def generate_text(self, tier: int, case: str = "upper") -> str:
        """Generate target text for tiers 1-2 (pure algorithmic)."""
        if tier == 1:
            words = self._sample_words(1, min_len=2, max_len=8)
        elif tier == 2:
            n_words = self.rng.randint(2, 3)
            words = self._sample_words(n_words, min_len=3, max_len=10)
        else:
            raise ValueError(f"Tier {tier} requires LLM — use get_must_include_words()")

        text = " ".join(words)
        return self._apply_case(text, case)

    def generate_text_fallback(self, tier: int, case: str = "upper") -> str:
        """Fallback for tiers 3-5 when LLM is unavailable.

        Combines frequency-dict words + optional thematic snippets
        to produce a pseudo-natural phrase.
        """
        if tier == 3:
            n = self.rng.randint(3, 5)
        elif tier == 4:
            n = self.rng.randint(3, 6)
        elif tier == 5:
            n = self.rng.randint(5, 8)
        else:
            return self.generate_text(tier, case)

        words = self._sample_words(n, min_len=3, max_len=12)

        # Occasionally mix in a thematic element
        if self.thematic and self.rng.random() < 0.3:
            pool_name = self.rng.choice(["cities", "first_names", "brands_seed", "slogans_seed"])
            pool = self.thematic.get(pool_name, [])
            if pool:
                extra = self.rng.choice(pool)
                words[self.rng.randint(0, len(words) - 1)] = extra

        text = " ".join(words)

        # For tier 4, split into title + subtitle
        if tier == 4 and len(words) >= 4:
            mid = len(words) // 2
            title = " ".join(words[:mid])
            subtitle = " ".join(words[mid:])
            text = f"{self._apply_case(title, 'upper')} / {self._apply_case(subtitle, 'lower')}"
            return text

        return self._apply_case(text, case)

    def get_must_include_words(self, tier: int) -> list[str]:
        """Return must-include words for LLM phrase generation (T3-T5).

        These are coverage-weighted so the LLM is nudged to use rare chars.
        """
        if tier == 3:
            n = self.rng.randint(1, 2)
        elif tier == 4:
            n = self.rng.randint(2, 3)
        elif tier == 5:
            n = self.rng.randint(2, 4)
        else:
            n = 1
        return self._sample_words(n, min_len=3, max_len=12)

    def generate_number_text(self) -> str:
        """Generate a number/date string for mixing into prompts."""
        kind = self.rng.choice(["year", "percent", "price", "date", "hashtag"])
        if kind == "year":
            return str(self.rng.randint(2020, 2030))
        if kind == "percent":
            return f"−{self.rng.choice([10, 15, 20, 25, 30, 40, 50, 70])}%"
        if kind == "price":
            p = self.rng.choice([99, 149, 199, 299, 499, 999, 1490, 2990])
            return f"{p} ₽"
        if kind == "date":
            d = self.rng.randint(1, 28)
            m = self.rng.randint(1, 12)
            return f"{d:02d}.{m:02d}.{self.rng.randint(2024, 2027)}"
        # hashtag
        return f"#{self.rng.randint(1, 999)}"

    def update_coverage(self, text: str):
        """Update character coverage counters after using *text*."""
        for ch in text.lower():
            if ch in CYRILLIC_LOWER:
                self.char_counts[ch] += 1
        self._updates_since_refresh += 1

    def coverage_report(self) -> dict[str, int]:
        return dict(self.char_counts.most_common())
