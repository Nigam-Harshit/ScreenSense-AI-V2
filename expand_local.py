"""
expand_local.py  --  Step 2 (offline replacement for expand_paraphrase.py)
Expands intents_v2.json -> intents_v3.json using:
  1. Domain-specific synonym substitution (static dict, no download)
  2. Prefix / suffix template variation (no download)
  3. Targeted antonym-pair disambiguation examples (handcrafted for the
     6 pairs that caused the most k-NN errors in Step 1)
  4. Diversity filter using the already-cached all-MiniLM-L6-v2 encoder
     (rejects candidates with cosine-similarity > SIM_THRESHOLD to any
     existing example in the same intent)

NO ADDITIONAL DOWNLOADS. Uses only:
  - sentence-transformers (already installed)
  - all-MiniLM-L6-v2 (already cached from Step 1)

Usage:
    python expand_local.py
    python expand_local.py --sim-threshold 0.80 --max-new 20
"""

import argparse
import itertools
import json
import os
import random
import re
import numpy as np

SIM_THRESHOLD  = 0.75    # reject if cosine_sim > this to any existing example
MAX_NEW_PER_INTENT = 20  # cap to keep intents balanced
RANDOM_STATE   = 42
INPUT_PATH     = "data/intents_v2.json"
OUTPUT_PATH    = "data/intents_v3.json"

random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synonym map  (command-vocabulary only — no general-purpose thesaurus needed)
# ─────────────────────────────────────────────────────────────────────────────
SYNONYMS = {
    "increase":    ["raise", "boost", "crank up", "pump up", "step up"],
    "raise":       ["increase", "boost", "turn up", "bump up"],
    "decrease":    ["lower", "reduce", "drop", "cut", "dial down"],
    "lower":       ["decrease", "reduce", "bring down", "turn down"],
    "turn up":     ["raise", "increase", "boost"],
    "turn down":   ["lower", "decrease", "reduce"],
    "open":        ["launch", "start", "run", "boot up", "fire up", "pull up"],
    "launch":      ["open", "start", "run", "fire up"],
    "start":       ["open", "launch", "run", "kick off"],
    "close":       ["quit", "exit", "shut", "kill", "terminate", "end"],
    "quit":        ["close", "exit", "shut down", "end"],
    "exit":        ["close", "quit", "shut"],
    "minimize":    ["minimise", "shrink", "hide", "collapse", "push down"],
    "maximise":    ["maximize", "expand", "enlarge", "full size"],
    "maximize":    ["maximise", "expand", "enlarge", "full size"],
    "expand":      ["maximize", "maximise", "enlarge"],
    "take":        ["capture", "grab", "snap", "save"],
    "capture":     ["take", "grab", "save", "snap"],
    "screenshot":  ["screen capture", "screengrab", "screen shot", "snip"],
    "lock":        ["secure", "protect", "lock down"],
    "sleep":       ["hibernate", "suspend", "rest"],
    "hibernate":   ["sleep", "suspend", "power down"],
    "mute":        ["silence", "quiet", "disable sound", "turn off sound"],
    "unmute":      ["restore sound", "enable sound", "turn sound back on"],
    "pause":       ["stop", "hold", "freeze"],
    "play":        ["resume", "start", "begin"],
    "skip":        ["jump to", "advance to", "go to next", "forward to"],
    "next":        ["following", "subsequent"],
    "previous":    ["prior", "last", "preceding"],
    "scroll":      ["move", "navigate"],
    "snap":        ["move", "push", "throw", "place"],
    "split":       ["divide", "side by side", "snap side by side"],
    "switch":      ["go to", "change to", "move to"],
    "desktop":     ["workspace", "virtual screen", "screen"],
    "workspace":   ["desktop", "virtual desktop"],
    "screen":      ["display", "monitor"],
    "window":      ["app", "application", "program"],
    "brightness":  ["screen brightness", "display brightness", "light level"],
    "volume":      ["sound", "audio level", "audio"],
    "louder":      ["more volume", "higher volume", "sound up"],
    "quieter":     ["less volume", "lower volume", "sound down"],
    "brighter":    ["more brightness", "higher brightness", "light it up"],
    "dimmer":      ["less brightness", "lower brightness", "darker"],
}

PREFIXES = [
    "",                    # no prefix (keep original-ish)
    "please ",
    "can you ",
    "could you ",
    "i need to ",
    "i want to ",
    "go ahead and ",
    "hey ",
]

SUFFIXES = [
    "",
    " please",
    " now",
    " for me",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Targeted antonym-pair disambiguation examples
#    Hand-crafted for the 6 pairs that caused the most k-NN errors in Step 1.
#    These use DISTINCTIVE directional language absent from the current dataset.
# ─────────────────────────────────────────────────────────────────────────────
ANTONYM_BOOST = {
    "scroll_up": [
        "scroll upward", "move the page upward", "navigate up the page",
        "go upward on this page", "scroll up this document", "view content above",
        "move up through the content", "jump to the top section",
        "navigate to the top", "look at the content above this",
        "bring the upper section into view", "scroll up towards the top",
    ],
    "scroll_down": [
        "scroll downward", "move the page downward", "navigate down the page",
        "go downward on this page", "scroll down this document", "view content below",
        "move down through the content", "jump to the bottom section",
        "navigate to the bottom", "look at the content below this",
        "bring the lower section into view", "scroll down towards the bottom",
    ],
    "move_left": [
        "push this window to the left side", "align the app to the left edge",
        "snap to left half of screen", "place the window on the left",
        "left-side snap", "put the window against the left wall",
        "dock the window to the left", "tile the window on the left side",
        "send the app to the left portion", "position window at left",
        "move app leftward", "anchor to left side",
    ],
    "move_right": [
        "push this window to the right side", "align the app to the right edge",
        "snap to right half of screen", "place the window on the right",
        "right-side snap", "put the window against the right wall",
        "dock the window to the right", "tile the window on the right side",
        "send the app to the right portion", "position window at right",
        "move app rightward", "anchor to right side",
    ],
    "next_desktop": [
        "move to the next virtual workspace", "switch to the workspace ahead",
        "go forward one desktop", "advance to the following workspace",
        "jump to the desktop after this one", "cycle to the next screen",
        "forward to next workspace", "go to the workspace in front",
        "step forward to next desktop", "move forward one workspace",
        "advance desktop forward", "switch forward one screen",
    ],
    "prev_desktop": [
        "move to the previous virtual workspace", "switch to the workspace behind",
        "go back one desktop", "go to the preceding workspace",
        "jump to the desktop before this one", "cycle to the previous screen",
        "backward to prior workspace", "go to the workspace behind me",
        "step back to previous desktop", "move back one workspace",
        "retreat to prior desktop", "switch backward one screen",
    ],
    "media_next": [
        "skip to the next track", "jump forward one track",
        "advance to the next song in the queue", "go forward in the playlist",
        "play the song after this one", "next song in the playlist",
        "skip ahead to next", "move forward one song",
        "forward track please", "play track after this",
        "advance the queue one step", "go to the following song",
    ],
    "media_prev": [
        "go back to the previous track", "jump back one track",
        "return to the prior song", "go backward in the playlist",
        "play the song before this one", "previous song in the playlist",
        "skip back to previous", "move backward one song",
        "rewind to prior track please", "play track before this",
        "retreat queue one step", "go to the preceding song",
    ],
    "minimize_button": [
        "push this window to the taskbar", "send app to taskbar",
        "collapse this window to taskbar", "shrink window to tray",
        "minimize to the bottom bar", "hide in the taskbar",
        "reduce window to taskbar icon", "taskbar this window",
        "drop window to taskbar", "put in taskbar",
    ],
    "maximize_button": [
        "expand this window to fill the screen",
        "make the window as large as possible",
        "fill the entire screen with this window",
        "expand to maximum window size",
        "take up the full screen real estate",
        "zoom this window to maximum",
        "blow this up to full window size",
        "make this window biggest it can be",
        "largest window size please",
        "expand window to full size",
    ],
    "move_top": [
        "snap this window to the top half", "move the window to the upper portion",
        "align window to top of screen", "tile to upper half",
        "push the window upward to top", "snap to top edge",
        "place window at top of display", "dock to top half",
        "top half snap please", "anchor window at the top",
    ],
    "move_bottom": [
        "snap this window to the bottom half", "move the window to the lower portion",
        "align window to bottom of screen", "tile to lower half",
        "push the window downward to bottom", "snap to bottom edge",
        "place window at bottom of display", "dock to bottom half",
        "bottom half snap please", "anchor window at the bottom",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rule-based candidate generator
# ─────────────────────────────────────────────────────────────────────────────

def _apply_synonym(text: str) -> list[str]:
    """Replace one word/phrase in text with a synonym. Returns list of variants."""
    results = []
    for word, syns in SYNONYMS.items():
        if word in text:
            for syn in syns:
                candidate = text.replace(word, syn, 1)
                if candidate != text:
                    results.append(candidate)
    return results


def _apply_prefix_suffix(text: str) -> list[str]:
    """Vary prefix and suffix. Skip combinations that produce duplicates."""
    results = []
    for pre in PREFIXES:
        for suf in SUFFIXES:
            # Only add suffix if sentence doesn't already end with "please"
            if suf == " please" and text.endswith("please"):
                continue
            if pre == "please " and suf == " please":
                continue  # "please X please" sounds odd
            candidate = (pre + text + suf).strip()
            if candidate != text:
                results.append(candidate)
    return results


def generate_candidates(text: str) -> list[str]:
    """Generate all rule-based candidate augmentations for a single example."""
    candidates = []
    # Synonym variants
    candidates.extend(_apply_synonym(text))
    # Prefix/suffix variants of original
    candidates.extend(_apply_prefix_suffix(text))
    # Prefix/suffix variants of synonym variants (cross-product, limited)
    for syn_var in _apply_synonym(text)[:3]:   # limit to first 3 synonym variants
        for pre in ["please ", "can you ", "i need to "]:
            c = (pre + syn_var).strip()
            if c not in candidates:
                candidates.append(c)
    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        if c not in seen and c != text:
            seen.add(c)
            unique.append(c)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diversity filter (uses cached sentence-transformer — NO download needed)
# ─────────────────────────────────────────────────────────────────────────────
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print("[Expand] Loading cached encoder (no download) ...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Expand] Encoder ready.")
    return _encoder


def _l2_norm(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / norms


def build_embed_matrix(texts: list[str]) -> np.ndarray:
    enc = get_encoder()
    embs = enc.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return _l2_norm(embs.astype(np.float32))


def is_diverse(candidate: str, exist_embs: np.ndarray,
               threshold: float = SIM_THRESHOLD) -> tuple[bool, float]:
    enc = get_encoder()
    cemb = _l2_norm(enc.encode([candidate], show_progress_bar=False,
                                convert_to_numpy=True))
    sims = (cemb @ exist_embs.T)[0]
    max_sim = float(sims.max())
    return max_sim < threshold, max_sim


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main expansion loop
# ─────────────────────────────────────────────────────────────────────────────

def expand(input_path: str, output_path: str,
           max_new: int, sim_threshold: float) -> None:

    with open(input_path) as f:
        data = json.load(f)

    # Warm up encoder before the loop (prints once, not per intent)
    get_encoder()

    new_intents        = []
    total_before       = 0
    total_after        = 0
    all_new_examples   = []   # for the 10-sample sanity report

    print(f"\n  {'Intent':<25} {'Before':>7} {'After':>6} {'Added':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6}")

    for item in data["intents"]:
        intent   = item["intent"]
        examples = list(item["examples"])
        n_before = len(examples)
        total_before += n_before

        # ── Build pool of all candidates ──────────────────────────────────────
        boost_pool = [c.strip().lower() for c in ANTONYM_BOOST.get(intent, [])]
        rule_pool  = []
        for source_ex in examples:
            rule_pool.extend(generate_candidates(source_ex))

        # Combine boost (priority) + rule-based, deduplicate
        seen_texts = set(examples)
        unique_candidates = []
        for c in (boost_pool + rule_pool):
            if c and c not in seen_texts:
                seen_texts.add(c)
                unique_candidates.append(c)

        if not unique_candidates:
            new_intents.append({"intent": intent, "examples": examples})
            total_after += n_before
            print(f"  {intent:<25} {n_before:>7} {n_before:>6} {0:>6}", flush=True)
            continue

        # ── BATCH encode all at once (one call per intent = fast) ──────────────
        enc = get_encoder()
        exist_embs = _l2_norm(enc.encode(
            examples, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32))                                    # (n_exist, D)

        cand_embs = _l2_norm(enc.encode(
            unique_candidates, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32))                                    # (n_cand, D)

        # ── Sequential diversity filter (greedy accept) ────────────────────────
        # Accept candidate i if cosine_sim to every already-accepted AND every
        # existing example is below sim_threshold. Process boost pool first.
        accepted_texts = []
        accepted_embs_list = []

        for i, cand_text in enumerate(unique_candidates):
            if len(accepted_texts) >= max_new:
                break
            c = cand_embs[i:i+1]                               # (1, D)

            # Check against original training examples
            if float((c @ exist_embs.T).max()) >= sim_threshold:
                continue
            # Check against already-accepted new examples this round
            if accepted_embs_list:
                acc_mat = np.vstack(accepted_embs_list)         # (n_acc, D)
                if float((c @ acc_mat.T).max()) >= sim_threshold:
                    continue

            accepted_texts.append(cand_text)
            accepted_embs_list.append(c)
            src = "antonym_boost" if cand_text in set(boost_pool) else "rule_based"
            all_new_examples.append({"intent": intent, "text": cand_text, "source": src})

        extended = examples + accepted_texts
        n_after  = len(extended)
        total_after += n_after
        print(f"  {intent:<25} {n_before:>7} {n_after:>6} {len(accepted_texts):>6}",
              flush=True)
        new_intents.append({"intent": intent, "examples": extended})

    # Save
    with open(output_path, "w") as f:
        json.dump({"intents": new_intents}, f, indent=2)

    print(f"\n  Saved: {output_path}")
    print(f"  Total: {total_before} -> {total_after} (+{total_after - total_before})")
    print(f"  Avg/intent: {total_before/len(new_intents):.1f} -> "
          f"{total_after/len(new_intents):.1f}")

    # ── Sanity check: 10 random new examples ─────────────────────────────────
    print(f"\n  --- 10 random newly generated examples (sanity check) ---")
    random.shuffle(all_new_examples)
    for ex in all_new_examples[:10]:
        tag = "[boost]" if ex["source"] == "antonym_boost" else "[rule] "
        print(f"  {tag} [{ex['intent']}]  {ex['text']}")

    n_boost = sum(1 for e in all_new_examples if e["source"] == "antonym_boost")
    n_rule  = sum(1 for e in all_new_examples if e["source"] == "rule_based")
    print(f"\n  Source: {n_boost} antonym-boost + {n_rule} rule-based "
          f"= {n_boost + n_rule} total new")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=INPUT_PATH)
    parser.add_argument("--output",         default=OUTPUT_PATH)
    parser.add_argument("--max-new",        type=int,   default=MAX_NEW_PER_INTENT)
    parser.add_argument("--sim-threshold",  type=float, default=SIM_THRESHOLD)
    args = parser.parse_args()

    print(f"\n[Expand] Offline rule-based augmentation (no model download needed)")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Max new/intent: {args.max_new}  |  Sim threshold: {args.sim_threshold}\n")
    expand(args.input, args.output, args.max_new, args.sim_threshold)
