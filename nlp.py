import re
import os
import json
import pickle
import numpy as np

# ── Model paths ────────────────────────────────────────────────────────────────
# k-NN (primary) — built by knn_intent_classifier.py
KNN_EMB_PATH  = "models/knn_train_embeddings.npy"
KNN_LBL_PATH  = "models/knn_train_labels.json"
KNN_META_PATH = "models/knn_meta.json"

# TF-IDF + LR (fallback, used only if k-NN model files are absent)
ML_MODEL_PATH  = "models/intent_model.pkl"
ML_VECTOR_PATH = "models/intent_vectorizer.pkl"

# Confidence gate: predictions below this score -> Route 3 (VLM fallback).
# For k-NN with k=7: 3/7 votes = 0.43 is the lowest majority-confidence level.
# For TF-IDF/LR fallback: 0.30 matches calibrated probability distribution.
CONF_THRESHOLD = 0.43   # matches knn_intent_classifier.py

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_knn_encoder    = None
_knn_embeddings = None   # (n_train, 384) float32
_knn_labels     = None   # list[str]
_knn_k          = 7

_lr_model      = None
_lr_vectorizer = None
_using_knn     = False   # set to True when k-NN loads successfully


# ── k-NN model loader ──────────────────────────────────────────────────────────

def _load_knn():
    global _knn_encoder, _knn_embeddings, _knn_labels, _knn_k, _using_knn
    if _knn_embeddings is not None:
        return True
    if not all(os.path.exists(p) for p in [KNN_EMB_PATH, KNN_LBL_PATH, KNN_META_PATH]):
        return False
    try:
        from sentence_transformers import SentenceTransformer
        _knn_embeddings = np.load(KNN_EMB_PATH)
        with open(KNN_LBL_PATH) as f:
            _knn_labels = json.load(f)
        with open(KNN_META_PATH) as f:
            meta = json.load(f)
        _knn_k = meta.get("k", 7)
        model_name = meta.get("model", "all-MiniLM-L6-v2")
        _knn_encoder = SentenceTransformer(model_name)
        _using_knn = True
        return True
    except Exception as e:
        print(f"[NLP] k-NN load failed ({e}), falling back to TF-IDF/LR")
        _knn_embeddings = None
        return False


def _knn_predict(command):
    """
    Returns (action, confidence).
    Confidence = fraction of k neighbours voting for the winning intent.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    emb  = _knn_encoder.encode([command])            # (1, 384)

    sims = cos_sim(emb, _knn_embeddings)[0]          # (n_train,)
    top_k_idx = np.argsort(sims)[-_knn_k:][::-1]
    top_labels = [_knn_labels[i] for i in top_k_idx]
    from collections import Counter
    votes      = Counter(top_labels)
    action     = votes.most_common(1)[0][0]
    confidence = votes[action] / _knn_k
    return action, confidence


# ── Keyword override layer ─────────────────────────────────────────────────────
# Post-filter applied after k-NN for two confusion groups identified in
# Step 1/3 evaluation. Only fires when the prediction is already a member
# of the group — zero interference with any other intent.

_SNAP_GROUP    = {"move_left", "move_right"}
_SNAP_KEYWORDS = {"left": "move_left", "right": "move_right"}

_DESK_GROUP    = {"next_desktop", "prev_desktop", "new_desktop"}
_DESK_KEYWORDS = {
    "next":     "next_desktop",  "forward":  "next_desktop",
    "ahead":    "next_desktop",  "right":    "next_desktop",
    "previous": "prev_desktop",  "back":     "prev_desktop",
    "behind":   "prev_desktop",  "prior":    "prev_desktop",
    "last":     "prev_desktop",  "left":     "prev_desktop",
}


def _keyword_override(action: str, words: list) -> str:
    """Snap prediction to correct label when a directional keyword is present."""
    word_set = set(words)
    if action in _SNAP_GROUP:
        for kw, intent in _SNAP_KEYWORDS.items():
            if kw in word_set:
                return intent
    if action in _DESK_GROUP:
        for kw, intent in _DESK_KEYWORDS.items():
            if kw in word_set:
                return intent
    return action


# ── TF-IDF / LR fallback loader ───────────────────────────────────────────────

def _load_lr():
    global _lr_model, _lr_vectorizer
    if _lr_model is not None:
        return True
    if not (os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_VECTOR_PATH)):
        print("[NLP] No model files found. Run knn_intent_classifier.py first.")
        return False
    try:
        with open(ML_MODEL_PATH, "rb") as f:
            _lr_model = pickle.load(f)
        with open(ML_VECTOR_PATH, "rb") as f:
            _lr_vectorizer = pickle.load(f)
        return True
    except Exception as e:
        print(f"[NLP] TF-IDF/LR load failed: {e}")
        return False


def _lr_predict(command):
    vec    = _lr_vectorizer.transform([command])
    action = _lr_model.predict(vec)[0]
    proba  = _lr_model.predict_proba(vec)[0]
    return action, float(max(proba))


# ── Slot / entity extractors (regex — intentionally rule-based) ───────────────

def _extract_number(command):
    m = re.search(r"(\d+)", command)
    return int(m.group(1)) if m else None

def _extract_typed_text(command):
    m = re.search(
        r"(?:type|dictate|write|input)(?:\s+(?:this|out|down|my|into the computer))?\s*:?\s*(.*)",
        command
    )
    return m.group(1).strip() if m else None

def _extract_app(words):
    return words[-1] if words else ""

def _extract_split_targets(command, words):
    if "and" in words:
        idx = words.index("and")
        if 0 < idx < len(words) - 1:
            return (words[idx - 1], words[idx + 1])
    if len(words) >= 3:
        return (words[-2], words[-1])
    return None


_SLOT_EXTRACTORS = {
    "set_brightness":  lambda cmd, words: _extract_number(cmd) or 50,
    "set_volume":      lambda cmd, words: _extract_number(cmd) or 50,
    "type_text":       lambda cmd, words: _extract_typed_text(cmd),
    "split_apps":      lambda cmd, words: _extract_split_targets(cmd, words),
    "open_app":        lambda cmd, words: _extract_app(words),
    "focus_window":    lambda cmd, words: _extract_app(words),
    "close_button":    lambda cmd, words: _extract_app(words),
    "minimize_button": lambda cmd, words: _extract_app(words),
    "maximize_button": lambda cmd, words: _extract_app(words),
    "move_left":       lambda cmd, words: _extract_app(words),
    "move_right":      lambda cmd, words: _extract_app(words),
    "move_top":        lambda cmd, words: _extract_app(words),
    "move_bottom":     lambda cmd, words: _extract_app(words),
    "move_fullscreen": lambda cmd, words: _extract_app(words),
}


def _extract_slot(action, command, words):
    extractor = _SLOT_EXTRACTORS.get(action)
    return extractor(command, words) if extractor else None


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_command(command: str) -> tuple:
    """
    Classify a natural-language command and extract its slot value.

    Returns
    -------
    (action, target, confidence)
        action     : str | None  -- intent label, or None if unrecognised
        target     : any         -- extracted slot (number, string, tuple, ...)
        confidence : float       -- [0.0, 1.0]
                                    Values below CONF_THRESHOLD -> VLM Route 3
    """
    command = command.lower().strip()
    words   = command.split()
    if not words:
        return None, None, 0.0

    # ── Try k-NN model (primary) ──────────────────────────────────────────
    if _load_knn():
        try:
            action, confidence = _knn_predict(command)
            action = _keyword_override(action, words)   # directional override
            if confidence < CONF_THRESHOLD:
                print(f"[NLP] Low confidence ({confidence:.2f}) for '{command}' "
                      f"-> '{action}' -- routing to VLM")
                return action, None, confidence
            target = _extract_slot(action, command, words)
            return action, target, confidence
        except Exception as e:
            print(f"[NLP] k-NN prediction error: {e}")

    # ── Fallback: TF-IDF + LR ─────────────────────────────────────────────
    if _load_lr():
        try:
            action, confidence = _lr_predict(command)
            lr_threshold = 0.30   # LR uses different calibration
            if confidence < lr_threshold:
                print(f"[NLP] LR low confidence ({confidence:.2f}) -> VLM")
                return action, None, confidence
            target = _extract_slot(action, command, words)
            return action, target, confidence
        except Exception as e:
            print(f"[NLP] LR prediction error: {e}")

    # ── Last resort: regex ────────────────────────────────────────────────
    print("[NLP] Using regex fallback -- no model loaded.")
    action, target = _regex_fallback(command, words)
    return action, target, 1.0


# ── Regex fallback (used only when no model files are present) ─────────────────

def _regex_fallback(command, words):
    if words[0] == "split" and len(words) >= 3:
        return "split_apps", (words[1], words[2])
    if words[0] == "open":
        return "open_app", words[-1]
    if words[0] in ("focus", "switch"):
        return "focus_window", words[-1]
    if "screenshot" in command:
        return "screenshot", None

    m = re.search(r"(?:set|make|change)?\s*brightness(?:\s*to)?\s*(\d+)", command)
    if m: return "set_brightness", int(m.group(1))
    m = re.search(r"(?:set|turn)?\s*volume(?:\s*to)?\s*(\d+)", command)
    if m: return "set_volume", int(m.group(1))

    if "increase brightness" in command or "brightness up" in command or "brighter" in command:
        return "brightness_up", None
    if "decrease brightness" in command or "brightness down" in command or "dimmer" in command:
        return "brightness_down", None
    if "increase volume" in command or "volume up" in command or "louder" in command:
        return "volume_up", None
    if "decrease volume" in command or "volume down" in command or "quieter" in command:
        return "volume_down", None

    if words[0] == "move" and len(words) >= 2:
        t = words[1]
        if "left"   in command: return "move_left",        t
        if "right"  in command: return "move_right",       t
        if "top"    in command: return "move_top",         t
        if "bottom" in command: return "move_bottom",      t
    if words[0] == "fullscreen":
        return "move_fullscreen", words[-1]

    if "close"    in command: return "close_button",    words[-1]
    if "minimize" in command or "minimise" in command:
        return "minimize_button", words[-1]
    if "maximize" in command: return "maximize_button", words[-1]

    if "mute" in command and "unmute" not in command: return "mute_volume",   None
    if "unmute" in command:                           return "unmute_volume",  None
    if "scroll down" in command or "page down" in command: return "scroll_down", None
    if "scroll up" in command or "page up" in command:     return "scroll_up",   None

    if "pause" in command or ("play " in command and ("music" in command or "video" in command)):
        return "media_play_pause", None
    if "next track" in command or "next song" in command:  return "media_next", None
    if "previous track" in command or "previous song" in command: return "media_prev", None

    if "new desktop" in command or "create desktop" in command: return "new_desktop",  None
    if "next desktop" in command:                               return "next_desktop", None
    if "previous desktop" in command:                           return "prev_desktop", None
    if "close desktop" in command:                              return "close_desktop",None

    m = re.search(r"type\s+(?:this\s*:?\s*)?(.*)", command)
    if m and m.group(1).strip():
        return "type_text", m.group(1).strip()

    if "lock screen" in command or command == "lock": return "lock_screen", None
    if "sleep" in command:                            return "sleep_pc",    None

    return None, None