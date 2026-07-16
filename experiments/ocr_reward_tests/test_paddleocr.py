"""Test PaddleOCR with Cyrillic recognition model on bad_text.webp"""
from pathlib import Path

from paddleocr import PaddleOCR

IMAGE_PATH = Path(__file__).resolve().parents[1] / "assets" / "bad_text.png"

# Use the PaddleOCR v3 API with cyrillic recognition model
ocr = PaddleOCR(
    text_recognition_model_name="cyrillic_PP-OCRv3_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    device="cpu",
)

result = ocr.predict(str(IMAGE_PATH))

print("=" * 60)
print("PaddleOCR Results (cyrillic_PP-OCRv3_mobile_rec)")
print("=" * 60)
for res in result:
    res.print()
    # Also extract structured data
    print("\n--- Structured output ---")
    d = res.to_dict() if hasattr(res, 'to_dict') else None
    if d:
        texts = d.get('rec_texts', [])
        scores = d.get('rec_scores', [])
        for t, s in zip(texts, scores):
            print(f"  Text: {t!r}  Confidence: {s:.4f}")
    else:
        # Try accessing attributes directly
        try:
            texts = res['rec_texts']
            scores = res['rec_scores']
            for t, s in zip(texts, scores):
                print(f"  Text: {t!r}  Confidence: {s:.4f}")
        except Exception as e:
            print(f"  Could not extract structured data: {e}")
            print(f"  Result type: {type(res)}")
            print(f"  Result dir: {[x for x in dir(res) if not x.startswith('_')]}")

print("\n" + "=" * 60)
print("Expected text: ЛУЧШЕЕ КАПУЧИНО 2025!")
print("Actual render (with errors): ЛУЧЦЕЕ КАПУЧНИNО 2025!")
print("=" * 60)
