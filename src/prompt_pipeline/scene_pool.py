"""Scene pool — load, sample, and expand scene descriptions."""

import json
import random
from pathlib import Path


class ScenePool:

    def __init__(self, seed_path: str, expanded_path: str | None = None,
                 seed: int = 42):
        self.rng = random.Random(seed)
        self.scenes: list[dict] = []
        self._by_type: dict[str, list[dict]] = {}

        self._load(seed_path)
        if expanded_path and Path(expanded_path).exists():
            self._load(expanded_path)

        self._index()

    # ------------------------------------------------------------------

    def _load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        new_scenes = data if isinstance(data, list) else data.get("scenes", [])
        # Deduplicate by id
        existing_ids = {s["id"] for s in self.scenes}
        for s in new_scenes:
            if s["id"] not in existing_ids:
                self.scenes.append(s)
                existing_ids.add(s["id"])

    def _index(self):
        self._by_type = {}
        for s in self.scenes:
            ct = s["content_type"]
            self._by_type.setdefault(ct, []).append(s)

    # ------------------------------------------------------------------

    def sample(self, content_type: str, lang: str = "ru") -> dict:
        pool = self._by_type.get(content_type, self.scenes)
        scene = self.rng.choice(pool)
        return {
            "id": scene["id"],
            "description": scene.get(lang, scene.get("ru", "")),
        }

    def content_types(self) -> list[str]:
        return list(self._by_type.keys())

    def __len__(self) -> int:
        return len(self.scenes)

    # ------------------------------------------------------------------
    # Expansion helpers (called from LLM client)
    # ------------------------------------------------------------------

    def add_scenes(self, new_scenes: list[dict]):
        existing_ids = {s["id"] for s in self.scenes}
        for s in new_scenes:
            if s["id"] not in existing_ids:
                self.scenes.append(s)
                existing_ids.add(s["id"])
        self._index()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"scenes": self.scenes}, f, ensure_ascii=False, indent=2)
