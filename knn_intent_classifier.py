"""
knn_intent_classifier.py  --  Step 1
k-NN intent classifier using sentence-transformer embeddings.

Usage:
    python knn_intent_classifier.py                     # build + evaluate on intents_v2.json
    python knn_intent_classifier.py --data intents_v3   # use a different dataset
    python knn_intent_classifier.py --eval-only         # skip rebuild, use saved index

Artifacts saved to models/:
    knn_train_embeddings.npy   -- (n_train, 384) float32 matrix
    knn_train_labels.json      -- list of intent labels (parallel to embeddings)
    knn_meta.json              -- dataset path, k, model name, CONF_THRESHOLD
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 22M params, 384-dim, ~90 MB download
K_NEIGHBORS      = 7
CONF_THRESHOLD   = 0.43   # = 3/7 votes; below this -> VLM fallback
RANDOM_STATE     = 42

MODELS_DIR = "models"
EMB_PATH   = os.path.join(MODELS_DIR, "knn_train_embeddings.npy")
LBL_PATH   = os.path.join(MODELS_DIR, "knn_train_labels.json")
META_PATH  = os.path.join(MODELS_DIR, "knn_meta.json")


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[kNN] Loading encoder: {EMBED_MODEL_NAME} ...")
        _encoder = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[kNN] Encoder ready. Embedding dimension: {_encoder.get_sentence_embedding_dimension()}")
    return _encoder


def encode(texts, batch_size=64, show_progress=True):
    enc = get_encoder()
    return enc.encode(texts, batch_size=batch_size,
                      show_progress_bar=show_progress, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
def build_index(dataset_path):
    print(f"\n[kNN] Building index from: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    X_all, y_all = [], []
    for item in data["intents"]:
        for ex in item["examples"]:
            X_all.append(ex)
            y_all.append(item["intent"])

    # Stratified 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"[kNN] Split: {len(X_train)} train / {len(X_val)} val (stratified, every class guaranteed)")

    print("[kNN] Encoding training examples ...")
    train_emb = encode(X_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(EMB_PATH, train_emb.astype(np.float32))
    with open(LBL_PATH, "w") as f:
        json.dump(y_train, f)
    with open(META_PATH, "w") as f:
        json.dump({"dataset": dataset_path, "k": K_NEIGHBORS,
                   "model": EMBED_MODEL_NAME, "conf_threshold": CONF_THRESHOLD,
                   "n_train": len(X_train), "n_val": len(X_val)}, f, indent=2)

    print(f"[kNN] Index saved ({train_emb.shape[0]} vectors, {train_emb.shape[1]}-dim)")
    return X_train, X_val, y_train, y_val, train_emb


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def knn_predict(query_emb, train_emb, train_labels, k=K_NEIGHBORS):
    """
    query_emb : (1, D) or (D,)
    Returns   : (predicted_label, vote_confidence, top_k_labels, top_k_sims)
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    sims = cosine_similarity(query_emb, train_emb)[0]       # (n_train,)
    top_k_idx    = np.argsort(sims)[-k:][::-1]
    top_k_labels = [train_labels[i] for i in top_k_idx]
    top_k_sims   = sims[top_k_idx]

    votes      = Counter(top_k_labels)
    prediction = votes.most_common(1)[0][0]
    confidence = votes[prediction] / k                       # fraction [0.14 .. 1.0] for k=7

    return prediction, confidence, top_k_labels, top_k_sims


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(X_val, y_val, train_emb, train_labels, verbose=True):
    print(f"\n[kNN] Encoding {len(X_val)} val examples ...")
    val_emb = encode(X_val, show_progress=verbose)

    y_pred, confidences = [], []
    for emb in val_emb:
        pred, conf, _, _ = knn_predict(emb, train_emb, train_labels)
        y_pred.append(pred)
        confidences.append(conf)

    confidences = np.array(confidences)
    y_pred      = np.array(y_pred)
    y_val_arr   = np.array(y_val)

    correct_mask = (y_pred == y_val_arr)
    wrong_mask   = ~correct_mask

    val_acc          = accuracy_score(y_val, y_pred)
    vlm_fallback_rate = float(np.mean(confidences < CONF_THRESHOLD))

    # -----------------------------------------------------------------------
    # Classification report
    # -----------------------------------------------------------------------
    intent_labels = sorted(set(y_val))
    report_str = classification_report(y_val, y_pred, labels=intent_labels, zero_division=0)

    print("\n" + "=" * 70)
    print("  k-NN INTENT CLASSIFIER -- EVALUATION RESULTS")
    print(f"  Model: {EMBED_MODEL_NAME}  |  k={K_NEIGHBORS}  |  CONF_THRESHOLD={CONF_THRESHOLD}")
    print("=" * 70)
    print(f"  Val Accuracy     : {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  Total errors     : {wrong_mask.sum()} / {len(y_val)}")
    print(f"  VLM Fallback Rate: {vlm_fallback_rate:.4f} ({vlm_fallback_rate*100:.1f}%)")
    print("=" * 70)
    print("\nPer-Intent Classification Report:")
    print(report_str)

    # -----------------------------------------------------------------------
    # Confidence distribution: correct vs wrong
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  CONFIDENCE DISTRIBUTION  (k-NN vote fraction)")
    print("=" * 70)

    def conf_stats(mask, label):
        vals = confidences[mask]
        if len(vals) == 0:
            print(f"  {label:<10}: (none)")
            return
        buckets = {
            "1/7 (0.14)": np.sum(vals <= 1/7 + 0.01),
            "2/7 (0.29)": np.sum((vals > 1/7 + 0.01) & (vals <= 2/7 + 0.01)),
            "3/7 (0.43)": np.sum((vals > 2/7 + 0.01) & (vals <= 3/7 + 0.01)),
            "4/7 (0.57)": np.sum((vals > 3/7 + 0.01) & (vals <= 4/7 + 0.01)),
            "5/7 (0.71)": np.sum((vals > 4/7 + 0.01) & (vals <= 5/7 + 0.01)),
            "6/7 (0.86)": np.sum((vals > 5/7 + 0.01) & (vals <= 6/7 + 0.01)),
            "7/7 (1.00)": np.sum(vals > 6/7 + 0.01),
        }
        print(f"\n  {label} predictions (n={len(vals)}, mean={vals.mean():.3f}, "
              f"median={np.median(vals):.3f}):")
        for bucket, count in buckets.items():
            bar = "#" * int(count * 40 / max(len(vals), 1))
            print(f"    {bucket}  {bar:<40} {count:>3} ({count/len(vals)*100:.0f}%)")

    conf_stats(correct_mask, "CORRECT")
    conf_stats(wrong_mask,   "WRONG  ")

    # -----------------------------------------------------------------------
    # Misclassified examples
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MISCLASSIFIED EXAMPLES")
    print("=" * 70)
    errors = [(X_val[i], y_val[i], y_pred[i], confidences[i])
              for i in range(len(y_val)) if y_pred[i] != y_val[i]]
    errors.sort(key=lambda x: -x[3])   # highest-confidence wrong = worst

    if errors:
        print(f"\n  {'Conf':>5}  {'True label':<22} {'Predicted':<22} Example")
        print(f"  {'-'*5}  {'-'*22} {'-'*22} {'-'*40}")
        for ex, true, pred, conf in errors:
            print(f"  {conf:>5.2f}  {true:<22} {pred:<22} {ex}")
    else:
        print("  No misclassifications!")

    print()
    return {
        "val_accuracy": val_acc,
        "vlm_fallback_rate": vlm_fallback_rate,
        "n_errors": int(wrong_mask.sum()),
        "n_val": len(y_val),
        "mean_conf_correct": float(confidences[correct_mask].mean()) if correct_mask.any() else 0,
        "mean_conf_wrong":   float(confidences[wrong_mask].mean())   if wrong_mask.any()  else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intents_v2.json",
                        help="Path to intents JSON dataset")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip rebuild; use saved index from models/")
    args = parser.parse_args()

    if args.eval_only and all(os.path.exists(p) for p in [EMB_PATH, LBL_PATH]):
        print("[kNN] Loading saved index ...")
        train_emb    = np.load(EMB_PATH)
        with open(LBL_PATH) as f:
            y_train  = json.load(f)
        with open(args.data) as f:
            data = json.load(f)
        X_all, y_all = [], []
        for item in data["intents"]:
            for ex in item["examples"]:
                X_all.append(ex)
                y_all.append(item["intent"])
        _, X_val, _, y_val = train_test_split(
            X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
        )
    else:
        X_train, X_val, y_train, y_val, train_emb = build_index(args.data)

    metrics = evaluate(X_val, y_val, train_emb, y_train)
    print(f"[kNN] Summary: accuracy={metrics['val_accuracy']*100:.1f}%  "
          f"VLM-fallback={metrics['vlm_fallback_rate']*100:.1f}%  "
          f"errors={metrics['n_errors']}/{metrics['n_val']}")
