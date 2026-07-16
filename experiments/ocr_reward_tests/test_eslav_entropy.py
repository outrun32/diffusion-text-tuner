"""
Probe the raw CTC output of eslav_PP-OCRv5_mobile_rec to extract logit
entropy as a rendering-quality signal.

Idea:
  The official pipeline collapses logits → argmax + max-prob → a single
  confidence score.  Even badly-rendered text gets high confidence because
  the model "knows" the script.

  But the per-timestep probability *distribution* is much more sensitive:
    - Clean text   → sharply-peaked distributions → low entropy per character
    - Garbled text → flat / noisy distributions  → high entropy per character

  We monkey-patch CTCLabelDecode.__call__ to expose the raw preds array,
  then compute:
    - mean_max_prob  : the official confidence metric
    - mean_entropy   : average entropy over all non-blank timesteps
    - min_char_prob  : minimum char probability (worst single character)
    - frac_uncertain : fraction of timesteps where top prob < threshold
"""
import os
from pathlib import Path
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import warnings; warnings.filterwarnings("ignore")
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

import paddlex.inference.models.text_recognition.processors as proc_mod

# ── patch: intercept raw preds before CTC decode ─────────────────────────────
_raw_preds_store = {}

_original_ctc_call = proc_mod.CTCLabelDecode.__call__

def _patched_ctc_call(self, pred, return_word_box=False, **kwargs):
    preds = np.array(pred[0])            # (batch, T, vocab_size)
    _raw_preds_store["last"] = preds     # stash before decode
    return _original_ctc_call(self, pred, return_word_box=return_word_box, **kwargs)

proc_mod.CTCLabelDecode.__call__ = _patched_ctc_call


def char_level_stats(preds_np: np.ndarray) -> dict:
    """
    preds_np : (T, vocab_size) — CTC output for ONE line image.
    Returns a dict of quality metrics (higher entropy = worse rendering).
    """
    # Check whether values look like logits or probabilities
    # Probabilities should sum to ~1 across vocab axis
    row_sums = preds_np.sum(axis=-1)
    is_probs = bool(np.allclose(row_sums, 1.0, atol=0.05))

    if is_probs:
        probs = preds_np
    else:
        probs = softmax(preds_np, axis=-1)

    T, V = probs.shape
    blank_id = 0  # CTC blank is always index 0 in PaddleOCR

    max_probs     = probs.max(axis=-1)          # (T,)
    pred_ids      = probs.argmax(axis=-1)        # (T,)
    non_blank     = pred_ids != blank_id

    # entropy at each timestep  H_t = -sum(p * log(p + eps))
    eps = 1e-9
    entropy_per_t = -(probs * np.log(probs + eps)).sum(axis=-1)  # (T,)

    # metrics over non-blank frames (the actual character positions)
    if non_blank.sum() == 0:
        nb_max_prob = nb_entropy = min_char_prob = frac_uncertain = float("nan")
    else:
        nb_probs   = max_probs[non_blank]
        nb_entropy_vals = entropy_per_t[non_blank]
        nb_max_prob    = float(nb_probs.mean())
        nb_entropy     = float(nb_entropy_vals.mean())
        min_char_prob  = float(nb_probs.min())
        frac_uncertain = float((nb_probs < 0.5).mean())

    return {
        "is_probs"       : is_probs,
        "T"              : T,
        "V"              : V,
        "non_blank_T"    : int(non_blank.sum()),
        "mean_max_prob"  : float(max_probs.mean()),
        "mean_entropy_all" : float(entropy_per_t.mean()),
        "mean_entropy_nonblank" : nb_entropy,
        "min_char_prob"  : min_char_prob,
        "frac_uncertain" : frac_uncertain,
    }


# ── run inference ─────────────────────────────────────────────────────────────
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    text_detection_model_name="PP-OCRv5_server_det",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

BASE = Path(__file__).resolve().parents[1] / "assets"
# (path, short_label, expected_words_for_cer)
IMAGES = [
    # ── good ──────────────────────────────────────────────────────────────────
    (BASE / "good_text.jpg",
     "good_text",
     ["ЛУЧШЕЕ", "КАПУЧИНО", "2025!"]),
    (BASE / "good_handwritten_text.jpg",
     "good_handwritten",
     None),
    # ── bad ───────────────────────────────────────────────────────────────────
    (BASE / "bad_text.png",
     "bad/glitchy_glyphs",
     ["ЛУЧШЕЕ", "КАПУЧИНО", "2025!"]),
    (BASE / "bad_text_отъездом_сюрпризы.jpg",
     "bad/wrong_char(Ъ→Ь)",
     ["ОТЪЕЗДОМ", "СЮРПРИЗЫ"]),
    (BASE / "bad_text_стэном_чёрная.jpg",
     "bad/substitutions(э,ё,р)",
     ["стэном", "чёрная"]),
    (BASE / "bad_text_свобода_радиостанции_для_метеоритов.jpg",
     "bad/heavy_garble",
     ["Свобода", "радиостанции", "для", "метеоритов"]),
]

def char_error_rate(hypothesis_lines: list, reference_words: list) -> float:
    """Simple CER: edit_distance(joined_hyp, joined_ref) / len(ref)."""
    if reference_words is None:
        return float("nan")
    ref = "".join(reference_words).lower()
    hyp = "".join(hypothesis_lines).lower()
    # Levenshtein via DP
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        ndp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            ndp[j] = min(ndp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[m] / max(len(ref), 1)


print("=" * 88)
print(f"{'Image':<32} {'conf':>6} {'entropy':>8} {'min_p':>7} {'frac_unc':>9} {'CER':>6}")
print("-" * 88)

for img_path, label, expected in IMAGES:
    _raw_preds_store.clear()
    result = ocr.ocr(str(img_path))

    # result is a list of dicts (new PaddleOCR v5 format)
    res = result[0] if isinstance(result, list) else result
    if not res:
        print(f"{label:<32}  (no text detected)")
        continue

    # dict format: res["rec_texts"], res["rec_scores"]
    if isinstance(res, dict):
        rec_texts  = res["rec_texts"]
        rec_scores = res["rec_scores"]
    else:
        # old tuple format fallback
        rec_texts  = [line[-1][0] for line in res]
        rec_scores = [line[-1][1] for line in res]

    texts_and_scores = list(zip(rec_texts, rec_scores))
    official_conf    = float(np.mean(rec_scores))

    # _raw_preds_store["last"] has the LAST batch processed; for multi-line
    # images it's the last textline batch.  We re-run per-line analysis if
    # there's more than one line by using the stored preds (batch dimension).
    raw = _raw_preds_store.get("last")   # (batch=N_lines, T, V) or (T, V)

    if raw is None:
        print(f"{label:<28}  (no raw preds captured)")
        continue

    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]       # normalise to (1, T, V)

    all_stats = [char_level_stats(raw[i]) for i in range(raw.shape[0])]
    avg_entropy  = float(np.mean([s["mean_entropy_nonblank"] for s in all_stats]))
    min_char_p   = float(np.mean([s["min_char_prob"]         for s in all_stats]))
    frac_unc     = float(np.mean([s["frac_uncertain"]        for s in all_stats]))

    cer = char_error_rate(rec_texts, expected)

    print(f"{label:<32} {official_conf:>6.3f} {avg_entropy:>8.3f} {min_char_p:>7.3f} {frac_unc:>9.3f} {cer:>6.3f}" if not np.isnan(cer)
          else f"{label:<32} {official_conf:>6.3f} {avg_entropy:>8.3f} {min_char_p:>7.3f} {frac_unc:>9.3f} {'n/a':>6}")

    # per-line breakdown
    texts = rec_texts
    for i, (txt, stats) in enumerate(zip(texts, all_stats)):
        print(f"   line {i}  text={txt!r:<26}  "
              f"max_p={stats['mean_max_prob']:.3f}  "
              f"H={stats['mean_entropy_nonblank']:.3f}  "
              f"V={stats['V']}  T={stats['T']} nb={stats['non_blank_T']}")

print("=" * 88)
print()
print("Columns:")
print("  conf     : mean(max_prob per timestep) — official PaddleOCR score")
print("  entropy  : mean CTC entropy at non-blank frames (higher = more confused)")
print("  min_p    : lowest single-character confidence")
print("  frac_unc : fraction of chars with top-prob < 0.5")
print("  CER      : character error rate vs ground-truth (requires expected=")
