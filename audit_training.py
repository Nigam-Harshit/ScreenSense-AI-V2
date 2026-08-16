"""
audit_training.py
Runs all 4 data quality checks flagged for review.
"""
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from difflib import SequenceMatcher

DATASET_PATH = "data/intents_v2.json"
RANDOM_STATE = 42

with open(DATASET_PATH) as f:
    data = json.load(f)

X_all, y_all = [], []
for item in data["intents"]:
    for ex in item["examples"]:
        X_all.append(ex)
        y_all.append(item["intent"])

# ============================================================
# CHECK 1: Split leakage — near-duplicate paraphrase analysis
# ============================================================
print("=" * 70)
print("CHECK 1: SPLIT LEAKAGE / NEAR-DUPLICATE ANALYSIS")
print("=" * 70)

# Reproduce the EXACT split used in train_model.py
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE
)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

leaks = []
HIGH_SIM = 0.80   # threshold for "near-duplicate"

for val_ex, val_lbl in zip(X_val, y_val):
    for train_ex, train_lbl in zip(X_train, y_train):
        if train_lbl == val_lbl:
            sim = similarity(val_ex, train_ex)
            if sim >= HIGH_SIM:
                leaks.append({
                    "val":   val_ex,
                    "train": train_ex,
                    "label": val_lbl,
                    "sim":   round(sim, 3)
                })

# Sort by similarity descending
leaks.sort(key=lambda x: -x["sim"])

print(f"\nSimilarity threshold: {HIGH_SIM} (>=80% char-level overlap = near-duplicate)")
print(f"Total near-duplicate (val, train) pairs found: {len(leaks)}")
print(f"Val examples flagged: {len(set(l['val'] for l in leaks))} / {len(X_val)}")
if leaks:
    print(f"\nTop 10 highest-similarity pairs:")
    print(f"  {'Sim':>5}  {'Label':<20} {'Val example':<38} Train example")
    print(f"  {'-'*5}  {'-'*20} {'-'*38} {'-'*35}")
    for l in leaks[:10]:
        print(f"  {l['sim']:>5.3f}  {l['label']:<20} {l['val']:<38} {l['train']}")
else:
    print("  No near-duplicates detected above threshold.")

# ============================================================
# CHECK 2: Misclassified val examples (true vs predicted)
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: MISCLASSIFIED VAL-SET EXAMPLES")
print("=" * 70)

vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_val_vec   = vec.transform(X_val)

model = LogisticRegression(C=5.0, max_iter=2000, solver="saga", random_state=42)
model.fit(X_train_vec, y_train)

y_pred  = model.predict(X_val_vec)
y_proba = model.predict_proba(X_val_vec)
max_conf = np.max(y_proba, axis=1)

errors = [
    (X_val[i], y_val[i], y_pred[i], round(float(max_conf[i]), 3))
    for i in range(len(y_val))
    if y_val[i] != y_pred[i]
]
errors.sort(key=lambda x: -x[3])   # highest-confidence wrong answers first

val_acc = accuracy_score(y_val, y_pred)
print(f"\nVal accuracy: {val_acc:.4f}  |  Errors: {len(errors)} / {len(y_val)}")
print(f"\n{'Conf':>5}  {'True label':<22} {'Predicted':<22} Example")
print(f"{'-'*5}  {'-'*22} {'-'*22} {'-'*40}")
for ex, true, pred, conf in errors:
    print(f"{conf:>5.3f}  {true:<22} {pred:<22} {ex}")

# ============================================================
# CHECK 3: Example diversity — 15 random samples per class
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: RANDOM SAMPLE OF EXAMPLES PER CLASS (diversity check)")
print("=" * 70)
import random
random.seed(42)

intent_map = {}
for item in data["intents"]:
    intent_map[item["intent"]] = item["examples"]

SAMPLE_N = 5  # print 5 per intent — enough to spot template patterns
for intent, examples in sorted(intent_map.items()):
    sample = random.sample(examples, min(SAMPLE_N, len(examples)))
    print(f"\n  [{intent}] ({len(examples)} total)")
    for ex in sample:
        print(f"    - {ex}")

# ============================================================
# CHECK 4: Class balance in val split
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: CLASS BALANCE IN VAL SPLIT")
print("=" * 70)

from collections import Counter
train_counts = Counter(y_train)
val_counts   = Counter(y_val)
all_intents  = sorted(set(y_all))

print(f"\n{'Intent':<25} {'Train':>7} {'Val':>5} {'Val%':>6}  {'Expected Val%':>14}  Status")
print(f"{'-'*25} {'-'*7} {'-'*5} {'-'*6}  {'-'*14}  {'-'*10}")

expected_val_pct = 0.20
issues = []
for intent in all_intents:
    total = train_counts[intent] + val_counts[intent]
    actual_val_pct = val_counts[intent] / total if total > 0 else 0
    status = "OK"
    if val_counts[intent] == 0:
        status = "ABSENT from val"
        issues.append(intent)
    elif abs(actual_val_pct - expected_val_pct) > 0.15:
        status = "IMBALANCED"
        issues.append(intent)
    print(f"{intent:<25} {train_counts[intent]:>7} {val_counts[intent]:>5} {actual_val_pct:>6.1%}  {expected_val_pct:>14.1%}  {status}")

print(f"\nSummary: {len(issues)} intents flagged ({', '.join(issues) if issues else 'none'})")
print(f"NOTE: Current split is random (not stratified) -- see findings below.")

# Verify: what would stratified split look like?
X_strat, X_val_strat, y_strat, y_val_strat = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
)
strat_val_counts = Counter(y_val_strat)
strat_issues = [i for i in all_intents
                if strat_val_counts[i] == 0 or
                   abs(strat_val_counts[i]/(strat_val_counts[i]+Counter(y_strat)[i]) - 0.20) > 0.10]
print(f"\nWith stratified split: {len(strat_issues)} intents would be flagged")
print(f"  Per-intent val counts with stratify: min={min(strat_val_counts.values())}, max={max(strat_val_counts.values())}, avg={np.mean(list(strat_val_counts.values())):.1f}")
