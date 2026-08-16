"""
step3_compare.py  --  Step 3
Rebuild k-NN index on intents_v3.json, re-evaluate, run leakage check,
check confusable pairs. Report old vs new numbers side-by-side.

Usage:
    python step3_compare.py
"""

import json
import os
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

V2_PATH = "data/intents_v2.json"
V3_PATH = "data/intents_v3.json"
RANDOM_STATE = 42

# Confusable pairs to test explicitly
CONFUSABLE_PAIRS = [
    ("media_next",      "next_desktop"),
    ("move_fullscreen", "maximize_button"),
    ("scroll_down",     "move_bottom"),
]

SIM_LEAK_THRESHOLD = 0.80
K_NEIGHBORS        = 7
CONF_THRESHOLD     = 0.43


# ---------------------------------------------------------------------------
# Helpers (inline so this script is standalone)
# ---------------------------------------------------------------------------
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print("[Step3] Loading sentence encoder ...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def encode(texts):
    return get_encoder().encode(texts, show_progress_bar=True, convert_to_numpy=True)


def knn_predict_batch(val_embs, train_embs, train_labels, k=K_NEIGHBORS):
    preds, confs = [], []
    for emb in val_embs:
        sims     = cosine_similarity(emb.reshape(1, -1), train_embs)[0]
        top_k    = np.argsort(sims)[-k:][::-1]
        top_lbls = [train_labels[i] for i in top_k]
        votes    = Counter(top_lbls)
        pred     = votes.most_common(1)[0][0]
        conf     = votes[pred] / k
        preds.append(pred)
        confs.append(conf)
    return np.array(preds), np.array(confs)


# ---------------------------------------------------------------------------
# Keyword override layer
# Applied AFTER k-NN, only within pre-defined confusion groups.
# Only fires when the k-NN already predicted a member of the group,
# ensuring zero interference with unrelated intents.
# ---------------------------------------------------------------------------

# Group 1: window snapping (left vs right)
_SNAP_GROUP = {"move_left", "move_right"}
_SNAP_KEYWORDS = {
    "left":  "move_left",
    "right": "move_right",
}

# Group 2: virtual desktop navigation (next vs prev)
_DESK_GROUP = {"next_desktop", "prev_desktop", "new_desktop"}
_DESK_KEYWORDS = {
    "next":     "next_desktop",
    "forward":  "next_desktop",
    "ahead":    "next_desktop",
    "right":    "next_desktop",   # "go right a desktop" = next
    "previous": "prev_desktop",
    "back":     "prev_desktop",
    "behind":   "prev_desktop",
    "prior":    "prev_desktop",
    "last":     "prev_desktop",
    "left":     "prev_desktop",   # "jump left workspace" = prev
}


def keyword_override(pred: str, command: str) -> str:
    """
    Post-process a k-NN prediction with high-precision keyword rules.
    Only modifies predictions that belong to a known confusion group.
    Returns the (possibly corrected) intent label.
    """
    words = set(command.lower().split())

    if pred in _SNAP_GROUP:
        for kw, intent in _SNAP_KEYWORDS.items():
            if kw in words:
                return intent

    if pred in _DESK_GROUP:
        for kw, intent in _DESK_KEYWORDS.items():
            if kw in words:
                return intent

    return pred   # no override fired


def load_and_split(path):
    with open(path) as f:
        data = json.load(f)
    X, y = [], []
    for item in data["intents"]:
        for ex in item["examples"]:
            X.append(ex)
            y.append(item["intent"])
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def leakage_check(X_train, X_val, y_train, y_val, threshold=SIM_LEAK_THRESHOLD):
    flagged_val = set()
    for val_ex, val_lbl in zip(X_val, y_val):
        for train_ex, train_lbl in zip(X_train, y_train):
            if train_lbl == val_lbl:
                sim = SequenceMatcher(None, val_ex, train_ex).ratio()
                if sim >= threshold:
                    flagged_val.add(val_ex)
                    break
    return len(flagged_val), len(X_val)


def evaluate_dataset(path, label, use_override=False):
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {label} ({path})")
    print(f"{'='*70}")
    X_train, X_val, y_train, y_val = load_and_split(path)
    print(f"  Split: {len(X_train)} train / {len(X_val)} val")

    print("  Encoding training examples ...")
    train_emb = encode(X_train)
    print("  Encoding val examples ...")
    val_emb   = encode(X_val)

    y_pred_raw, confidences = knn_predict_batch(val_emb, train_emb, y_train)

    if use_override:
        y_pred = np.array([keyword_override(p, x)
                           for p, x in zip(y_pred_raw, X_val)])
        n_overridden = int(np.sum(y_pred != y_pred_raw))
        print(f"  Keyword override: {n_overridden} predictions changed")
    else:
        y_pred = y_pred_raw

    y_val_arr = np.array(y_val)

    val_acc          = accuracy_score(y_val, y_pred)
    vlm_fallback     = float(np.mean(confidences < CONF_THRESHOLD))
    n_errors         = int(np.sum(y_pred != y_val_arr))
    correct_mask     = y_pred == y_val_arr
    wrong_mask       = ~correct_mask
    mean_conf_correct = float(confidences[correct_mask].mean()) if correct_mask.any() else 0
    mean_conf_wrong   = float(confidences[wrong_mask].mean())   if wrong_mask.any()  else 0

    print(f"\n  Val Accuracy     : {val_acc*100:.1f}%")
    print(f"  Errors           : {n_errors}/{len(y_val)}")
    print(f"  VLM Fallback Rate: {vlm_fallback*100:.1f}%")
    print(f"  Mean conf correct: {mean_conf_correct:.3f}")
    print(f"  Mean conf wrong  : {mean_conf_wrong:.3f}")

    # Leakage
    n_flagged, n_total = leakage_check(X_train, X_val, y_train, y_val)
    print(f"  Leakage (>={SIM_LEAK_THRESHOLD} sim): {n_flagged}/{n_total} val examples ({n_flagged/n_total*100:.1f}%)")

    # Classification report (compact)
    intent_labels = sorted(set(y_val))
    print("\n  Per-Intent Classification Report:")
    print(classification_report(y_val, y_pred, labels=intent_labels, zero_division=0))

    # Confusable pair analysis
    print(f"\n  CONFUSABLE PAIR ANALYSIS")
    print(f"  {'Pair':<45} {'Correct':>8} {'Confused':>9}  Notes")
    print(f"  {'-'*45} {'-'*8} {'-'*9}  {'-'*20}")
    for lbl_a, lbl_b in CONFUSABLE_PAIRS:
        mask_a = y_val_arr == lbl_a
        mask_b = y_val_arr == lbl_b
        # A examples predicted as B
        a_confused = int(np.sum((y_val_arr == lbl_a) & (y_pred == lbl_b)))
        b_confused = int(np.sum((y_val_arr == lbl_b) & (y_pred == lbl_a)))
        a_correct  = int(np.sum((y_val_arr == lbl_a) & (y_pred == lbl_a)))
        b_correct  = int(np.sum((y_val_arr == lbl_b) & (y_pred == lbl_b)))
        total_a = int(mask_a.sum())
        total_b = int(mask_b.sum())
        pair_str = f"{lbl_a} <-> {lbl_b}"
        note = f"{lbl_a}:{a_correct}/{total_a}  {lbl_b}:{b_correct}/{total_b}"
        confused_total = a_confused + b_confused
        print(f"  {pair_str:<45} {a_correct+b_correct:>8} {confused_total:>9}  {note}")

    return {
        "label": label,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": val_acc,
        "n_errors": n_errors,
        "vlm_fallback_rate": vlm_fallback,
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_wrong":   mean_conf_wrong,
        "leakage_flagged": n_flagged,
        "leakage_total":   n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(V3_PATH):
        print(f"[Step3] ERROR: {V3_PATH} not found. Run expand_local.py first.")
        exit(1)

    results_v2  = evaluate_dataset(V2_PATH, "intents_v2 (original)",          use_override=False)
    results_v3  = evaluate_dataset(V3_PATH, "intents_v3 (expanded)",           use_override=False)
    results_v3k = evaluate_dataset(V3_PATH, "intents_v3 + keyword override",   use_override=True)

    print(f"\n{'='*78}")
    print("  SIDE-BY-SIDE COMPARISON: v2  |  v3  |  v3 + keyword override")
    print(f"{'='*78}")
    print(f"  {'Metric':<28} {'v2':>12} {'v3':>12} {'v3+override':>13}  {'v2->v3+kw':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*13}  {'-'*10}")

    all_r = [results_v2, results_v3, results_v3k]

    def row3(name, k, fmt="{:.1f}%", scale=100):
        vals  = [r[k] * scale for r in all_r]
        delta = vals[2] - vals[0]   # v3+override vs v2
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<28} {fmt.format(vals[0]):>12} {fmt.format(vals[1]):>12} "
              f"{fmt.format(vals[2]):>13}  {sign}{fmt.format(delta):>9}")

    row3("Training examples", "n_train", fmt="{:.0f}", scale=1)
    row3("Val examples",      "n_val",   fmt="{:.0f}", scale=1)
    row3("Val Accuracy",      "val_accuracy")
    row3("Val Errors",        "n_errors", fmt="{:.0f}", scale=1)
    row3("VLM Fallback Rate", "vlm_fallback_rate")
    row3("Mean conf (correct)", "mean_conf_correct", fmt="{:.3f}", scale=1)
    row3("Mean conf (wrong)",   "mean_conf_wrong",   fmt="{:.3f}", scale=1)

    leak_pcts = [r["leakage_flagged"] / r["leakage_total"] * 100 for r in all_r]
    delta_leak = leak_pcts[2] - leak_pcts[0]
    sign = "+" if delta_leak >= 0 else ""
    print(f"  {'Leakage rate (val)':<28} {leak_pcts[0]:>11.1f}% {leak_pcts[1]:>11.1f}% "
          f"{leak_pcts[2]:>12.1f}%  {sign}{delta_leak:>9.1f}%")
    print()
