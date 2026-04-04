"""Test TrOCR Cyrillic handwritten model on three images."""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import time

MODEL_ID = "/tmp/trocr-cyrillic"

IMAGES = [
    {
        "path": "bad_text.png",
        "desc": "AI-generated cat, broken Russian text (intended: ЛУЧШЕЕ КАПУЧИНО 2025!)",
        "crops": [
            ("Sign area",          (80, 510, 900, 930)),
            ("Line 1 (ЛУЧЦЕЕ)",    (140, 520, 810, 650)),
            ("Line 2 (КАПУЧНИNО)", (70, 630, 880, 790)),
            ("Line 3 (2025!)",     (230, 760, 730, 920)),
        ],
    },
    {
        "path": "good_text.jpg",
        "desc": "AI-generated cat, correct Russian text (ЛУЧШЕЕ КАПУЧИНО 2025!)",
        "crops": [
            ("Sign area",         (80, 510, 900, 930)),
            ("Line 1 (ЛУЧШЕЕ)",   (140, 520, 810, 660)),
            ("Line 2 (КАПУЧИНО)",  (70, 640, 880, 790)),
            ("Line 3 (2025!)",    (230, 770, 730, 920)),
        ],
    },
    {
        "path": "good_handwritten_text.jpg",
        "desc": "Handwritten Russian cursive (Лучшее Капучино 2025!)",
        "crops": [
            ("Sign area",          (80, 510, 900, 940)),
            ("Line 1 (Лучшее)",   (100, 510, 840, 670)),
            ("Line 2 (Капучино)",  (80, 650, 860, 800)),
            ("Line 3 (2025!)",    (220, 770, 740, 930)),
        ],
    },
]


def load_model():
    print(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return processor, model


def recognize(processor, model, image: Image.Image) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, num_beams=1, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    processor, model = load_model()

    for info in IMAGES:
        path = info["path"]
        print("=" * 70)
        print(f"IMAGE: {path}")
        print(f"  {info['desc']}")
        print("-" * 70)

        img = Image.open(path).convert("RGB")
        w, h = img.size
        print(f"  Size: {w}x{h}")

        # Full image
        t0 = time.time()
        text_full = recognize(processor, model, img)
        dt = time.time() - t0
        print(f"  [Full image] ({dt:.1f}s) → \"{text_full}\"")

        # Cropped lines
        for label, box in info["crops"]:
            x0 = max(0, box[0])
            y0 = max(0, box[1])
            x1 = min(w, box[2])
            y1 = min(h, box[3])
            crop = img.crop((x0, y0, x1, y1))
            t0 = time.time()
            text_crop = recognize(processor, model, crop)
            dt = time.time() - t0
            print(f"  [{label}] ({dt:.1f}s) → \"{text_crop}\"")

        print()

    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
