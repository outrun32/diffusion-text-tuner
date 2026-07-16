"""Extended PaddleOCR test with multiple configurations"""
import os
from pathlib import Path
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

IMAGE_PATH = Path(__file__).resolve().parents[1] / "assets" / "bad_text.png"

configs = [
    {
        "name": "Config 1: cyrillic model + default det",
        "kwargs": {
            "text_recognition_model_name": "cyrillic_PP-OCRv3_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "device": "cpu",
        }
    },
    {
        "name": "Config 2: cyrillic model + lower det threshold",
        "kwargs": {
            "text_recognition_model_name": "cyrillic_PP-OCRv3_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_det_limit_side_len": 960,
            "device": "cpu",
        }
    },
    {
        "name": "Config 3: cyrillic model + no textline orientation",
        "kwargs": {
            "text_recognition_model_name": "cyrillic_PP-OCRv3_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
        }
    },
]

for cfg in configs:
    print("=" * 60)
    print(cfg["name"])
    print("=" * 60)
    try:
        ocr = PaddleOCR(**cfg["kwargs"])
        result = ocr.predict(str(IMAGE_PATH))
        for res in result:
            d = res.to_dict() if hasattr(res, 'to_dict') else {}
            if d:
                texts = d.get('rec_texts', [])
                scores = d.get('rec_scores', [])
                for t, s in zip(texts, scores):
                    print(f"  Text: {t!r:30s}  Confidence: {s:.4f}")
            else:
                try:
                    for t, s in zip(res['rec_texts'], res['rec_scores']):
                        print(f"  Text: {t!r:30s}  Confidence: {s:.4f}")
                except:
                    res.print()
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("=" * 60)
print("REFERENCE:")
print("  Expected (intended): ЛУЧШЕЕ КАПУЧИНО 2025!")
print("  Actual render (errors): ЛУЧЦЕЕ КАПУЧНИNО 2025!")
print("=" * 60)
