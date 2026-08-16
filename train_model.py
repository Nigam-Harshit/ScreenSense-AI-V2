import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -- W&B ---------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[W&B] wandb not installed -- run `pip install wandb` to enable logging.")


def train():
    # -- Load dataset --------------------------------------------------------
    print("Loading dataset...")
    # Priority: expanded v2 dataset > intents_expanded > intents (legacy)
    for candidate in ["data/intents_v2.json", "data/intents_expanded.json", "data/intents.json"]:
        if os.path.exists(candidate):
            dataset_path = candidate
            break
    else:
        raise FileNotFoundError("No intents dataset found in data/")
    print(f"Using dataset: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    X, y = [], []
    for item in data["intents"]:
        for example in item["examples"]:
            X.append(example)
            y.append(item["intent"])

    num_intents   = len(data["intents"])
    num_examples  = len(X)
    intent_labels = sorted(set(y))

    print(f"Loaded {num_examples} examples across {num_intents} intents.")

    # -- Train / Val split (80/20, stratified) ---------------------------------
    # stratify=y is safe now: intents_v2.json has 25 examples per class,
    # guaranteeing >= 5 val examples per intent regardless of RNG.
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Split: {len(X_train)} train / {len(X_val)} val (stratified)")

    # -- Hyperparameters -------------------------------------------------------
    # CONF_THRESHOLD: commands below this fall through to VLM (Route 3).
    # 0.30 is calibrated to the actual confidence distribution of this model:
    # correct predictions cluster at 0.35-0.90, random/ambiguous ones < 0.30.
    # Must match nlp.py.
    CONF_THRESHOLD = 0.30
    LR_C          = 5.0
    LR_MAX_ITER   = 2000
    LR_SOLVER     = "saga"

    # -- W&B init -------------------------------------------------------------
    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            project="screensense-v2",
            name="intent-classifier-lr",
            config={
                "model":          "LogisticRegression",
                "vectorizer":     "TfidfVectorizer",
                "C":              LR_C,
                "max_iter":       LR_MAX_ITER,
                "solver":         LR_SOLVER,
                "conf_threshold": CONF_THRESHOLD,
                "train_size":     len(X_train),
                "val_size":       len(X_val),
                "num_intents":    num_intents,
                "num_examples":   num_examples,
                "dataset":        dataset_path,
            }
        )
        print("[W&B] Run initialized ->", wandb.run.url)

    # -- Vectoriser -----------------------------------------------------------
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    # -- Classifier -----------------------------------------------------------
    print("Training Logistic Regression Classifier...")
    model = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver=LR_SOLVER,
                               random_state=42)
    model.fit(X_train_vec, y_train)

    # -- Evaluation -----------------------------------------------------------
    y_pred    = model.predict(X_val_vec)
    y_proba   = model.predict_proba(X_val_vec)
    max_conf  = np.max(y_proba, axis=1)

    val_accuracy      = accuracy_score(y_val, y_pred)
    vlm_fallback_rate = float(np.mean(max_conf < CONF_THRESHOLD))

    val_labels = sorted(set(y_val))   # only classes that appear in val set
    report_str = classification_report(y_val, y_pred, labels=val_labels,
                                       zero_division=0)

    print("\n" + "-" * 60)
    print("INTENT CLASSIFIER EVALUATION RESULTS")
    print("-" * 60)
    print(f"Val Accuracy        : {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
    print(f"Avg Max Confidence  : {np.mean(max_conf):.4f}")
    print(f"VLM Fallback Rate   : {vlm_fallback_rate:.4f} ({vlm_fallback_rate*100:.1f}%)")
    print("-" * 60)
    print("\nPer-Intent Classification Report:")
    print(report_str)

    # -- Log to W&B -----------------------------------------------------------
    if run:
        from sklearn.metrics import classification_report as cr_dict
        report_dict = cr_dict(y_val, y_pred, labels=val_labels,
                              output_dict=True, zero_division=0)

        wandb.log({
            "val/accuracy":          val_accuracy,
            "val/avg_confidence":    float(np.mean(max_conf)),
            "val/vlm_fallback_rate": vlm_fallback_rate,
            "val/macro_f1":          report_dict["macro avg"]["f1-score"],
            "val/macro_precision":   report_dict["macro avg"]["precision"],
            "val/macro_recall":      report_dict["macro avg"]["recall"],
        })

        table = wandb.Table(columns=["Intent", "Precision", "Recall", "F1", "Support"])
        for intent in val_labels:
            if intent in report_dict:
                r = report_dict[intent]
                table.add_data(intent, round(r["precision"], 3),
                               round(r["recall"], 3), round(r["f1-score"], 3),
                               int(r["support"]))
        wandb.log({"val/per_intent_metrics": table})
        wandb.log({"val/confidence_histogram": wandb.Histogram(max_conf.tolist())})

        run.finish()
        print("[W&B] Metrics logged. Check your dashboard.")

    # -- Save models ----------------------------------------------------------
    print("\nSaving models to models/ directory...")
    os.makedirs("models", exist_ok=True)

    with open("models/intent_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("models/intent_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Training complete. Models saved to models/.")
    print(f"\n  Val Accuracy  : {val_accuracy*100:.1f}%")
    print(f"  VLM Fallback  : {vlm_fallback_rate*100:.1f}% of commands would hit Route 3")


if __name__ == "__main__":
    train()
