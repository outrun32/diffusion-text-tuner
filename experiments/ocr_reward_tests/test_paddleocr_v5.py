"""
Test PaddleOCR PP-OCRv5 with Cyrillic recognition on three test images.
Compare with previous PP-OCRv3 results.
"""
import os
from pathlib import Path
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
IMAGES = [
    (ASSETS_DIR / "bad_text.png",
     "AI-gen cat with broken text (intended: ЛУЧШЕЕ КАПУЧИНО 2025!)"),
    (ASSETS_DIR / "good_text.jpg",
     "AI-gen cat with correct text ПРИВЕТ МИР!"),
    (ASSETS_DIR / "good_handwritten_text.jpg",
     "Handwritten Russian text"),
]

CONFIGS = [
    {
        "name": "PP-OCRv5 (lang='ru' → eslav_PP-OCRv5_mobile_rec + PP-OCRv5_server_det)",
        "kwargs": {
            "lang": "ru",
            # ocr_version defaults to PP-OCRv5 for ru
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
        },
    },
    {
        "name": "PP-OCRv5 cyrillic (text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec')",
        "kwargs": {
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "cyrillic_PP-OCRv5_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
        },
    },
    {
        "name": "PP-OCRv5 server rec (text_recognition_model_name='PP-OCRv5_server_rec')",
        "kwargs": {
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
        },
    },
    {
        "name": "PP-OCRv3 cyrillic (baseline, text_recognition_model_name='cyrillic_PP-OCRv3_mobile_rec')",
        "kwargs": {
            "text_recognition_model_name": "cyrillic_PP-OCRv3_mobile_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
        },
    },
]


def extract_results(result):
    """Extract text and scores from PaddleOCR result."""
    lines = []
    for res in result:
        d = res.json.get("res", res.json)  # results nested under 'res' key
        texts = d.get("rec_texts", [])
        scores = d.get("rec_scores", [])
        # scores may be numpy array
        import numpy as np
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        for t, s in zip(texts, scores):
            lines.append((t, float(s)))
    return lines


def run_config(cfg):
    print(f"\n{'='*70}")
    print(f"  {cfg['name']}")
    print(f"{'='*70}")
    try:
        ocr = PaddleOCR(**cfg["kwargs"])
    except Exception as e:
        print(f"  INIT ERROR: {e}")
        return {}

    all_results = {}
    for img_path, description in IMAGES:
        print(f"\n  --- {os.path.basename(img_path)}: {description} ---")
        if not os.path.exists(img_path):
            print(f"  FILE NOT FOUND: {img_path}")
            continue
        try:
            result = ocr.predict(str(img_path))
            lines = extract_results(result)
            if not lines:
                print("  (no text detected)")
            for text, score in lines:
                print(f"    Text: {text!r:40s}  Confidence: {score:.4f}")
            all_results[os.path.basename(img_path)] = lines
        except Exception as e:
            print(f"  PREDICT ERROR: {e}")
            import traceback
            traceback.print_exc()
    return all_results


# Run all configs and collect results
all_config_results = {}
for cfg in CONFIGS:
    results = run_config(cfg)
    all_config_results[cfg["name"]] = results

# Print comparison summary
print("\n\n")
print("=" * 80)
print("  COMPARISON SUMMARY")
print("=" * 80)

for img_path, description in IMAGES:
    fname = os.path.basename(img_path)
    print(f"\n{'─'*80}")
    print(f"  {fname}: {description}")
    print(f"{'─'*80}")
    print(f"  {'Config':<55s} {'Text':<25s} {'Conf':>6s}")
    print(f"  {'─'*55} {'─'*25} {'─'*6}")
    for cfg_name, results in all_config_results.items():
        short_name = cfg_name.split("(")[0].strip()
        if fname in results:
            for text, score in results[fname]:
                print(f"  {short_name:<55s} {text!r:<25s} {score:>6.4f}")
        else:
            print(f"  {short_name:<55s} {'(error/missing)':<25s} {'':>6s}")

print(f"\n{'='*80}")
print("  MODEL VERSION DETAILS")
print("=" * 80)
print("  PP-OCRv3 cyrillic: cyrillic_PP-OCRv3_mobile_rec (det: PP-OCRv5_server_det default)")
print("  PP-OCRv5 eslav:    eslav_PP-OCRv5_mobile_rec + PP-OCRv5_server_det")
print("  PP-OCRv5 cyrillic: cyrillic_PP-OCRv5_mobile_rec + PP-OCRv5_server_det")
print("  PP-OCRv5 server:   PP-OCRv5_server_rec + PP-OCRv5_server_det (CH/CHT/EN/JP)")
print("=" * 80)
