"""Configuration constants for the prompt generation pipeline."""

import re

# ---------------------------------------------------------------------------
# Cyrillic
# ---------------------------------------------------------------------------
CYRILLIC_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
CYRILLIC_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
CYRILLIC_ALL = CYRILLIC_UPPER + CYRILLIC_LOWER
RARE_CHARS = set("щЩъЪёЁэЭюЮфФцЦ")
CYRILLIC_WORD_RE = re.compile(r"^[а-яёА-ЯЁ\-]+$")

# ---------------------------------------------------------------------------
# Content types  (sum = 1.0)
# ---------------------------------------------------------------------------
CONTENT_TYPES = {
    "poster":      0.25,   # Movie / concert / event / ad posters
    "photo_text":  0.20,   # Photo with overlaid text, quotes, memes
    "typography":  0.15,   # Art lettering, calligraphy, decorative text
    "product":     0.10,   # Packaging, labels, brand shots
    "social_media": 0.10,  # YouTube thumbnails, stories, banners
    "clothing":    0.08,   # T-shirt / hoodie prints, merch
    "book_cover":  0.07,   # Book / album / magazine covers
    "street_art":  0.03,   # Graffiti, murals, stencils
    "niche":       0.02,   # Coins, medals, engravings, carvings
}

# ---------------------------------------------------------------------------
# Text tiers  (global default + per-content overrides)
# ---------------------------------------------------------------------------
TIER_WEIGHTS = {
    1: 0.05,   # single word  (1-4 chars)
    2: 0.20,   # 1-3 words
    3: 0.35,   # short phrase / slogan  (LLM)
    4: 0.25,   # title + subtitle       (LLM)
    5: 0.15,   # 1-2 sentences          (LLM)
}

TIER_OVERRIDES = {
    "niche":      {1: 0.30, 2: 0.40, 3: 0.25, 4: 0.05, 5: 0.00},
    "clothing":   {1: 0.15, 2: 0.35, 3: 0.35, 4: 0.10, 5: 0.05},
    "street_art": {1: 0.20, 2: 0.40, 3: 0.30, 4: 0.10, 5: 0.00},
    "typography": {1: 0.10, 2: 0.25, 3: 0.35, 4: 0.20, 5: 0.10},
    "book_cover": {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.35, 5: 0.15},
}

# ---------------------------------------------------------------------------
# Case styles for target text
# ---------------------------------------------------------------------------
CASE_WEIGHTS = {
    "upper": 0.50,
    "title": 0.30,
    "lower": 0.15,
    "mixed": 0.05,   # e.g. "ЗАГОЛОВОК\nподзаголовок"
}

# ---------------------------------------------------------------------------
# Style options
# ---------------------------------------------------------------------------
FONTS = [
    "bold sans-serif", "thin sans-serif", "italic serif", "serif",
    "monospace", "handwritten", "calligraphic", "decorative",
    "stencil", "grunge", "gothic", "minimalist", "retro",
    "futuristic", "brush script",
]

COLORS = [
    "white", "black", "gold", "red", "neon blue", "silver",
    "dark green", "orange", "pastel pink", "metallic grey",
    "deep purple", "teal", "cream", "electric yellow", "burgundy",
]

EFFECTS = [
    "clean", "embossed", "engraved", "neon glow", "shadow",
    "outlined", "gradient", "distressed", "3D", "watercolor",
    "chalk", "spray-painted", "laser-etched", "frosted glass",
    "holographic",
]

SIZES = ["large", "medium", "small", "tiny"]

# Per-content constraints (absent keys fall back to global lists)
STYLE_CONSTRAINTS = {
    "niche": {
        "fonts": ["serif", "stencil", "gothic", "monospace"],
        "effects": ["engraved", "embossed", "3D", "clean"],
        "sizes": ["small", "tiny", "medium"],
    },
    "street_art": {
        "fonts": ["grunge", "stencil", "bold sans-serif", "brush script"],
        "effects": ["spray-painted", "distressed", "clean", "shadow"],
    },
    "typography": {
        "fonts": [
            "calligraphic", "decorative", "brush script",
            "handwritten", "gothic", "futuristic",
        ],
        "effects": [
            "clean", "gradient", "3D", "watercolor",
            "outlined", "holographic",
        ],
        "sizes": ["large", "medium"],
    },
    "book_cover": {
        "fonts": [
            "serif", "italic serif", "decorative", "gothic",
            "minimalist", "calligraphic",
        ],
    },
    "clothing": {
        "fonts": [
            "bold sans-serif", "grunge", "minimalist",
            "stencil", "retro", "futuristic",
        ],
        "sizes": ["large", "medium"],
    },
}

# ---------------------------------------------------------------------------
# Russian translations for style terms (used in RU prompt templates)
# ---------------------------------------------------------------------------
FONT_RU = {
    "bold sans-serif": "жирным гротесковым",
    "thin sans-serif": "тонким гротесковым",
    "italic serif": "курсивным антиквенным",
    "serif": "антиквенным",
    "monospace": "моноширинным",
    "handwritten": "рукописным",
    "calligraphic": "каллиграфическим",
    "decorative": "декоративным",
    "stencil": "трафаретным",
    "grunge": "гранжевым",
    "gothic": "готическим",
    "minimalist": "минималистичным",
    "retro": "ретро",
    "futuristic": "футуристическим",
    "brush script": "кистевым",
}

COLOR_RU = {
    "white": "белым",
    "black": "чёрным",
    "gold": "золотым",
    "red": "красным",
    "neon blue": "неоново-голубым",
    "silver": "серебристым",
    "dark green": "тёмно-зелёным",
    "orange": "оранжевым",
    "pastel pink": "пастельно-розовым",
    "metallic grey": "металлически-серым",
    "deep purple": "тёмно-фиолетовым",
    "teal": "бирюзовым",
    "cream": "кремовым",
    "electric yellow": "ярко-жёлтым",
    "burgundy": "бордовым",
}

EFFECT_SUFFIX_RU = {
    "clean": "",
    "embossed": ", с тиснением",
    "engraved": ", с гравировкой",
    "neon glow": ", с неоновым свечением",
    "shadow": ", с тенью",
    "outlined": ", с обводкой",
    "gradient": ", с градиентом",
    "distressed": ", с потёртостями",
    "3D": ", в 3D стиле",
    "watercolor": ", в акварельной стилистике",
    "chalk": ", мелом",
    "spray-painted": ", аэрозольной краской",
    "laser-etched": ", с лазерной гравировкой",
    "frosted glass": ", с эффектом матового стекла",
    "holographic": ", с голографическим эффектом",
}

SIZE_RU = {
    "large": "Крупный",
    "medium": "Среднего размера",
    "small": "Небольшой",
    "tiny": "Мелкий",
}

# ---------------------------------------------------------------------------
# Rendering difficulty boosts for unique Cyrillic characters
# ---------------------------------------------------------------------------
# Characters are grouped by visual distance from Latin alphabet.
# FLUX/diffusion models were trained on predominantly Latin text, so chars
# with no Latin equivalent are rendered worst and need the most coverage.
#
# Group A — Latin lookalikes (many seen during FLUX pretraining):
#   а→a, в→B, е→E, к→K, м→M, н→H, о→O, р→P, с→C, т→T, у→Y, х→X
#   → no extra boost, covered naturally by frequency dict
#
# Group B — Partial resemblance (Greek/Cyrillic-specific but model has seen some):
#   Б≈6, Г≈Г(Gamma), З≈3, И≈N-mirror, Й≈Й, Л≈Λ, П≈Π, Э≈Ε-mirror, Я≈R-mirror
#   → modest boost ×2.5
#
# Group C — Visually unique, no Latin equivalent, complex glyph structure:
#   Ж, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Ю, Ё, Ф
#   → strong boost ×5.0

CHAR_RENDER_BOOST: dict[str, float] = {
    # Group B — partial resemblance
    "б": 2.5, "г": 2.5, "д": 2.0, "з": 2.5, "и": 1.5,
    "й": 3.0, "л": 2.0, "п": 2.0, "я": 2.0,
    # Group C — visually unique
    "ж": 5.0, "ц": 5.0, "ч": 5.0, "ш": 5.0, "щ": 5.0,
    "ъ": 5.0, "ы": 5.0, "ь": 4.0, "э": 5.0, "ю": 4.0, "ё": 4.5, "ф": 4.5,
}
# Uppercase versions map to same boost (matching is done on lowercased chars)

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
LANG_RU_RATIO = 0.80

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = (
    "Ты — генератор текста для визуального контента. "
    "Генерируй ТОЛЬКО запрошенный текст, без кавычек, объяснений и комментариев."
)

LLM_PHRASE_PROMPTS = {
    3: (
        "Придумай короткий слоган или фразу из 3-7 слов для {content_type_ru}. "
        "Обязательно используй слова: {words}. "
        "Ответь одной строкой."
    ),
    4: (
        "Придумай заголовок и подзаголовок (через символ переноса строки) "
        "для {content_type_ru}. Обязательно используй слова: {words}. "
        "Формат:\nЗАГОЛОВОК\nподзаголовок"
    ),
    5: (
        "Придумай 1-2 коротких предложения для {content_type_ru}. "
        "Обязательно используй слова: {words}."
    ),
}

CONTENT_TYPE_RU = {
    "poster": "рекламного постера",
    "photo_text": "фотографии с текстом",
    "typography": "типографической композиции",
    "product": "упаковки товара",
    "social_media": "обложки для соцсетей",
    "clothing": "принта на одежде",
    "book_cover": "обложки книги",
    "street_art": "уличного арта",
    "niche": "гравировки или чеканки",
}

LLM_SCENE_PROMPT = (
    "Придумай {n} уникальных описаний визуальных сцен для генерации изображений.\n"
    "Категория: {category}.\n"
    "Описывай ТОЛЬКО визуальную сцену и фон. НЕ упоминай текст, буквы или надписи.\n"
    "Каждое описание — 1-2 предложения.\n"
    "Формат: по одному описанию на строку, без нумерации и маркеров."
)
