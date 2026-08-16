"""
diag_errors.py  --  Pull confidence values for specific intents from v3 eval.
Prints every misclassified example for the 4 target intents with its confidence,
then summarises how many fall above/below CONF_THRESHOLD (0.43).
"""
import json, numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

TARGET_INTENTS  = {"move_left", "move_right", "next_desktop", "prev_desktop"}
V3_PATH         = "data/intents_v3.json"
K               = 7
CONF_THRESHOLD  = 0.43
RANDOM_STATE    = 42

def load_and_split(path):
    with open(path) as f: data = json.load(f)
    X, y = [], []
    for item in data["intents"]:
        for ex in item["examples"]:
            X.append(ex); y.append(item["intent"])
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

def knn_predict_batch(val_embs, train_embs, train_labels, k=K):
    preds, confs = [], []
    for emb in val_embs:
        sims   = cosine_similarity(emb.reshape(1,-1), train_embs)[0]
        top_k  = np.argsort(sims)[-k:][::-1]
        votes  = Counter(train_labels[i] for i in top_k)
        pred   = votes.most_common(1)[0][0]
        conf   = votes[pred] / k
        preds.append(pred); confs.append(conf)
    return preds, confs

from sentence_transformers import SentenceTransformer
print("Loading encoder...")
enc = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder ready.\n")

X_train, X_val, y_train, y_val = load_and_split(V3_PATH)
train_embs = enc.encode(X_train, show_progress_bar=True, convert_to_numpy=True)
val_embs   = enc.encode(X_val,   show_progress_bar=True, convert_to_numpy=True)

preds, confs = knn_predict_batch(val_embs, train_embs, list(y_train))

# ── Pull target-intent errors ────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  MISCLASSIFIED EXAMPLES — target intents ({', '.join(sorted(TARGET_INTENTS))})")
print(f"{'='*72}")
print(f"  {'Conf':>5}  {'VLM?':>5}  {'True label':<16} {'Predicted':<16} Example")
print(f"  {'-'*5}  {'-'*5}  {'-'*16} {'-'*16} {'-'*30}")

above_threshold, below_threshold = [], []
for text, true_lbl, pred_lbl, conf in zip(X_val, y_val, preds, confs):
    if true_lbl not in TARGET_INTENTS:
        continue
    if pred_lbl == true_lbl:
        continue
    would_fallback = conf < CONF_THRESHOLD
    marker = "  YES" if would_fallback else "   no"
    row = f"  {conf:.2f}  {marker}  {true_lbl:<16} {pred_lbl:<16} {text}"
    print(row)
    (below_threshold if would_fallback else above_threshold).append((conf, true_lbl, pred_lbl, text))

# ── Summary ──────────────────────────────────────────────────────────────────
total_errors = len(above_threshold) + len(below_threshold)
print(f"\n{'='*72}")
print(f"  SUMMARY for target intents (move_left/right, next/prev_desktop)")
print(f"{'='*72}")
print(f"  Total errors in these 4 intents : {total_errors}")
print(f"  VLM fallback catches (conf<0.43): {len(below_threshold)}  ({len(below_threshold)/max(total_errors,1)*100:.0f}%)")
print(f"  Slip through  (conf>=0.43, wrong): {len(above_threshold)}  ({len(above_threshold)/max(total_errors,1)*100:.0f}%)")

if above_threshold:
    print(f"\n  SLIP-THROUGH errors (these are the problem — VLM won't help):")
    for conf, tl, pl, txt in sorted(above_threshold, reverse=True):
        print(f"    conf={conf:.2f}  [{tl}] → [{pl}]  '{txt}'")
else:
    print(f"\n  ✅ All errors in these intents are at conf < 0.43 → VLM fallback absorbs them.")
