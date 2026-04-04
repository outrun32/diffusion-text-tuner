"""Assembler — combines text, scene, and style into a final FLUX prompt."""

from __future__ import annotations

import random

from .config import (
    COLOR_RU,
    EFFECT_SUFFIX_RU,
    FONT_RU,
    SIZE_RU,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
# RU templates (scene is in Russian)
# {color}/{font} = instrumental case ("красным"/"антиквенным") — pair with "шрифтом"
# {color_raw}/{font_raw} = raw English terms — pair with keyword labels "цвет:"/"шрифт:"
_TEMPLATES_RU = [
    "{scene}. {size} текст '{target_text}', {color} {font} шрифтом{effect}",
    "{scene}. Надпись '{target_text}', цвет: {color_raw}, шрифт: {font_raw}{effect}. Высокое качество",
    "{scene}. Текст '{target_text}', шрифт: {font_raw}, цвет: {color_raw}{effect}",
    "{scene}. На изображении {placement} '{target_text}', {color} {font} шрифтом{effect}",
    "{scene}. Текст '{target_text}', {color} {font} шрифтом{effect}. Детализированное изображение",
]

# EN templates (scene is in English)
_TEMPLATES_EN = [
    "{scene}. {size_en} text '{target_text}' in {color_en} {font_en} font{effect_en}",
    "{scene}. The text '{target_text}' in {color_en} {font_en} style{effect_en}, high quality",
    "{scene} with clearly readable text '{target_text}', {font_en} font, {color_en} color{effect_en}",
    "{scene}. A {size_en_lower} {placement_en} reading '{target_text}', {color_en} {font_en}{effect_en}, detailed",
]

# Content-type-specific placement nouns
_PLACEMENT_RU = {
    "poster": ["текст", "надпись", "заголовок"],
    "photo_text": ["надпись поверх фото", "наложенный текст", "цитата"],
    "typography": ["леттеринг", "надпись", "типографика"],
    "product": ["название на упаковке", "этикетка", "бренд"],
    "social_media": ["заголовок", "текст", "надпись"],
    "clothing": ["принт", "надпись на одежде", "текст на принте"],
    "book_cover": ["название", "заголовок на обложке", "текст обложки"],
    "street_art": ["граффити", "надпись", "тег"],
    "niche": ["гравировка", "надпись", "чеканка"],
}

_PLACEMENT_EN = {
    "poster": ["text", "headline", "title"],
    "photo_text": ["overlaid text", "caption", "quote"],
    "typography": ["lettering", "inscription", "type"],
    "product": ["label", "brand name", "product text"],
    "social_media": ["headline", "text", "title"],
    "clothing": ["print", "text on clothing", "graphic text"],
    "book_cover": ["title", "cover text", "name"],
    "street_art": ["graffiti", "tag", "spray text"],
    "niche": ["engraving", "inscription", "embossed text"],
}

# Effect suffix translations for EN
_EFFECT_SUFFIX_EN = {
    "clean": "",
    "embossed": ", embossed",
    "engraved": ", engraved",
    "neon glow": ", with neon glow",
    "shadow": ", with drop shadow",
    "outlined": ", outlined",
    "gradient": ", with gradient",
    "distressed": ", distressed texture",
    "3D": ", 3D style",
    "watercolor": ", watercolor style",
    "chalk": ", chalk drawing",
    "spray-painted": ", spray-painted",
    "laser-etched": ", laser-etched",
    "frosted glass": ", frosted glass effect",
    "holographic": ", holographic",
}


class Assembler:

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def assemble(
        self,
        target_text: str,
        scene: dict,          # {"id": ..., "description": ...}
        style: dict,           # {"font", "color", "effect", "size"}
        content_type: str,
        lang: str = "ru",
    ) -> str:
        if lang == "ru":
            return self._assemble_ru(target_text, scene, style, content_type)
        return self._assemble_en(target_text, scene, style, content_type)

    # ------------------------------------------------------------------
    # Russian
    # ------------------------------------------------------------------

    def _assemble_ru(self, target_text, scene, style, content_type) -> str:
        template = self.rng.choice(_TEMPLATES_RU)
        placement = self.rng.choice(
            _PLACEMENT_RU.get(content_type, ["текст"]))
        return template.format(
            scene=scene["description"],
            target_text=target_text,
            size=SIZE_RU.get(style["size"], ""),
            placement=placement,
            color=COLOR_RU.get(style["color"], style["color"]),
            color_raw=style["color"],
            font=FONT_RU.get(style["font"], style["font"]),
            font_raw=style["font"],
            effect=EFFECT_SUFFIX_RU.get(style["effect"], ""),
        )

    # ------------------------------------------------------------------
    # English
    # ------------------------------------------------------------------

    def _assemble_en(self, target_text, scene, style, content_type) -> str:
        template = self.rng.choice(_TEMPLATES_EN)
        placement = self.rng.choice(
            _PLACEMENT_EN.get(content_type, ["text"]))
        size_en = style["size"].capitalize()
        return template.format(
            scene=scene["description"],
            target_text=target_text,
            size_en=size_en,
            size_en_lower=style["size"],
            placement_en=placement,
            color_en=style["color"],
            font_en=style["font"],
            effect_en=_EFFECT_SUFFIX_EN.get(style["effect"], ""),
        )
