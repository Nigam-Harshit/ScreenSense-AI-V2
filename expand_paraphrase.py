"""
expand_paraphrase.py  --  Step 2
Locally generate paraphrases using humarin/chatgpt_paraphraser_on_T5_base,
filter with cosine-similarity dedup (threshold=0.75), save as intents_v3.json.

Model is downloaded once (~900 MB) and cached in HF cache.
Runs fully offline after first download.

Usage:
    python expand_paraphrase.py
    python expand_paraphrase.py --candidates 4 --max-per-intent 15
"""

import argparse
import json
import os
import random
import numpy as np

PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # reuse same encoder already downloaded

SIM_THRESHOLD    = 0.75   # reject candidate if cosine_sim > this to any existing example
N_CANDIDATES     = 4      # paraphrases to generate per source example
MAX_NEW_PER_INTENT = 15   # cap so no intent balloons relative to others
RANDOM_STATE     = 42

random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Load models (lazy)
# ---------------------------------------------------------------------------
_paraphrase_tokenizer = None
_paraphrase_model     = None
_encoder              = None


def get_paraphrase_model():
    global _paraphrase_tokenizer, _paraphrase_model
    if _paraphrase_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print(f"[Para] Loading paraphrase model: {PARAPHRASE_MODEL}")
        print("[Para]   (first run: ~900 MB download, cached after that)")
        _paraphrase_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        _paraphrase_model     = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL)
        _paraphrase_model.eval()
        print("[Para] Paraphrase model ready.")
    return _paraphrase_tokenizer, _paraphrase_model


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Para] Loading sentence encoder: {EMBED_MODEL_NAME}")
        _encoder = SentenceTransformer(EMBED_MODEL_NAME)
    return _encoder


# ---------------------------------------------------------------------------
# Paraphrase a single sentence
# ---------------------------------------------------------------------------
def paraphrase(text, n=N_CANDIDATES):
    """Generate up to n paraphrases for text."""
    import torch
    tokenizer, model = get_paraphrase_model()

    input_ids = tokenizer(
        f"paraphrase: {text} </s>",
        return_tensors="pt", padding=True, truncation=True, max_length=64
    ).input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            num_return_sequences=n,
            num_beams=n + 1,          # slightly more beams than sequences
            max_length=64,
            temperature=1.5,          # more diverse outputs
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

    return [
        tokenizer.decode(o, skip_special_tokens=True).strip().lower()
        for o in outputs
    ]


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------
def is_diverse(candidate, existing_embs, sim_threshold=SIM_THRESHOLD):
    """Return True if candidate embedding is sufficiently different from all existing."""
    if len(existing_embs) == 0:
        return True
    enc = get_encoder()
    cand_emb = enc.encode([candidate])                         # (1, D)
    sims = (cand_emb @ existing_embs.T) / (
        np.linalg.norm(cand_emb) * np.linalg.norm(existing_embs, axis=1) + 1e-9
    )
    return float(sims.max()) < sim_threshold


# ---------------------------------------------------------------------------
# Main expansion loop
# ---------------------------------------------------------------------------
def expand(dataset_path, out_path, n_candidates, max_new, sim_threshold):
    with open(dataset_path) as f:
        data = json.load(f)

    enc = get_encoder()     # warm up encoder first (it's cheaper)

    new_intents = []
    total_before = 0
    total_after  = 0
    all_new_examples = []   # for the random-sample report

    for item in data["intents"]:
        intent    = item["intent"]
        examples  = list(item["examples"])
        n_before  = len(examples)
        total_before += n_before

        # Encode all existing examples for this intent
        existing_embs = enc.encode(examples, show_progress_bar=False)   # (n, D)
        # L2-normalise for faster cosine via dot product
        existing_embs = existing_embs / (np.linalg.norm(existing_embs, axis=1, keepdims=True) + 1e-9)

        new_for_intent = []
        seen_texts = set(examples)

        for source_ex in examples:
            if len(new_for_intent) >= max_new:
                break
            try:
                candidates = paraphrase(source_ex, n=n_candidates)
            except Exception as e:
                print(f"    [Para] Generation error for '{source_ex}': {e}")
                continue

            for cand in candidates:
                if len(new_for_intent) >= max_new:
                    break
                cand = cand.strip().lower()
                if not cand or cand in seen_texts:
                    continue
                # Diversity check
                cand_emb = enc.encode([cand], show_progress_bar=False)
                cand_emb_norm = cand_emb / (np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9)
                sims = (cand_emb_norm @ existing_embs.T)[0]
                if sims.max() < sim_threshold:
                    new_for_intent.append(cand)
                    seen_texts.add(cand)
                    # Grow the existing embedding matrix so subsequent candidates
                    # are checked against the newly added ones too
                    existing_embs = np.vstack([existing_embs, cand_emb_norm])
                    all_new_examples.append({"intent": intent, "text": cand})

        extended_examples = examples + new_for_intent
        n_after = len(extended_examples)
        total_after += n_after
        print(f"  {intent:<25}  {n_before:>3} -> {n_after:>3}  (+{len(new_for_intent)})")
        new_intents.append({"intent": intent, "examples": extended_examples})

    # Save
    out_data = {"intents": new_intents}
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Total: {total_before} -> {total_after} examples  (+{total_after - total_before})")
    print(f"Avg per intent: {total_before/len(new_intents):.1f} -> {total_after/len(new_intents):.1f}")

    # Print 10 random new examples for sanity check
    print(f"\n--- 10 random newly generated examples (sanity check) ---")
    random.shuffle(all_new_examples)
    for item in all_new_examples[:10]:
        print(f"  [{item['intent']}]  {item['text']}")

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="data/intents_v2.json")
    parser.add_argument("--output",      default="data/intents_v3.json")
    parser.add_argument("--candidates",  type=int, default=N_CANDIDATES)
    parser.add_argument("--max-per-intent", type=int, default=MAX_NEW_PER_INTENT)
    parser.add_argument("--sim-threshold",  type=float, default=SIM_THRESHOLD)
    args = parser.parse_args()

    print(f"\n[Para] Expanding {args.input}")
    print(f"       Candidates/example: {args.candidates}  "
          f"Max new/intent: {args.max_per_intent}  "
          f"Sim threshold: {args.sim_threshold}\n")

    expand(args.input, args.output, args.candidates, args.max_per_intent, args.sim_threshold)
