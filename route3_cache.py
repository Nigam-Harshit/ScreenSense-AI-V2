"""
route3_cache.py  --  Semantic embedding cache for Route 3 VLM fallback.

Cache key  : normalized all-MiniLM-L6-v2 embedding of
             "intent:{action} | target:{target} | cmd:{command} [screen:{screen_hash}]"
Cache value: dict with action, target, reasoning, vlm_confidence

Design decisions (all flagged as "chosen, pending tuning"):
  - Similarity threshold : 0.80  (configurable via route3_config.py)
  - Max entries          : 500   (LRU eviction at this limit)
  - Persistence          : session-only by default (see open design question
                           documented in route3_config.py about expiry policy)

The encoder (all-MiniLM-L6-v2) is loaded lazily on first use to avoid
import-time cost if the cache is disabled via ROUTE3_CACHE_ENABLED = False.
It reuses the same model already used by the NLP k-NN classifier, so no
additional model download is required.
"""

import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Any

from route3_config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_MAX_ENTRIES,
    CACHE_HASH_MAX_HAMMING,
    CACHE_SEMANTIC_ENABLED,
    CACHE_SEMANTIC_MIN_SIM,
)

# ── Lazy encoder ───────────────────────────────────────────────────────────────
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from model_loader import load_sentence_encoder
        _encoder = load_sentence_encoder("all-MiniLM-L6-v2")
    return _encoder


def _embed(text: str) -> np.ndarray:
    enc = _get_encoder()
    v   = enc.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    # L2-normalize so cosine similarity == dot product
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _hamming_distance(h1: str, h2: str) -> int:
    """
    Compute Hamming distance (differing bit count) between two 32-hex perceptual screen hashes.
    Returns a large sentinel (999999) if either hash is missing, invalid, or equals 'display_unavailable'.
    """
    if not h1 or not h2 or h1 == "display_unavailable" or h2 == "display_unavailable":
        return 999999
    try:
        val1 = int(h1, 16)
        val2 = int(h2, 16)
        return bin(val1 ^ val2).count("1")
    except (ValueError, TypeError):
        return 999999


# ── SemanticCache ──────────────────────────────────────────────────────────────

class SemanticCache:
    """
    In-memory LRU cache for Route 3 VLM responses.

    Matching policy (Brief C - P2):
      (a) Exact normalised command text match (or semantic cosine similarity
          >= CACHE_SEMANTIC_MIN_SIM if CACHE_SEMANTIC_ENABLED is True).
      (b) Perceptual screen hash within CACHE_HASH_MAX_HAMMING bits (12 bits),
          never matching 'display_unavailable'.
      (c) Same classifier hint intent when action_hint is provided.

    Similarity threshold, max entries, and max hamming distance are set
    in route3_config.py and passed at construction time for testability.
    """

    def __init__(self,
                 similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
                 max_entries:          int   = CACHE_MAX_ENTRIES,
                 max_hamming:          int   = CACHE_HASH_MAX_HAMMING):
        self._threshold     = similarity_threshold
        self._max           = max_entries
        self._max_hamming   = max_hamming
        # OrderedDict: entry_id str → dict
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._counter       = 0

    def _cache_key_text(self, command: str, screen_hash: str, action: str = None, target: str = None) -> str:
        act = action if action is not None else "none"
        tgt = str(target) if target is not None else "none"
        return f"intent:{act} | target:{tgt} | cmd:{command.lower().strip()} [screen:{screen_hash}]"

    def query(self, command: str, screen_hash: str,
              action_hint: str = None, target_hint: str = None) -> tuple[bool, dict | None]:
        """
        Look up a cache entry.
        Bypasses window control button intents.
        Requires:
          (a) Exact normalised command text match (or semantic command similarity >= CACHE_SEMANTIC_MIN_SIM
              if CACHE_SEMANTIC_ENABLED is True).
          (b) Screen hash Hamming distance <= CACHE_HASH_MAX_HAMMING (12 bits), never matching 'display_unavailable'.
          (c) Stored action matches current classifier action_hint (when action_hint is provided).

        Returns:
          (True,  cached_response_dict)  on hit  — response includes '_cache_similarity' and '_cache_hamming'
          (False, None)                  on miss
        """
        # Hard policy: window control button intents never query the semantic cache
        if action_hint in ("close_button", "minimize_button", "maximize_button"):
            return False, None

        if not screen_hash or screen_hash == "display_unavailable":
            return False, None

        import route3_config
        semantic_enabled = getattr(route3_config, "CACHE_SEMANTIC_ENABLED", False)
        semantic_min_sim = getattr(route3_config, "CACHE_SEMANTIC_MIN_SIM", CACHE_SEMANTIC_MIN_SIM)
        max_hamming = getattr(route3_config, "CACHE_HASH_MAX_HAMMING", self._max_hamming)

        norm_cmd = command.lower().strip()
        q_emb = None
        if semantic_enabled:
            q_emb = _embed(norm_cmd)

        best_sim     = -1.0
        best_resp    = None
        best_id      = None
        best_hamming = 999999

        for entry_id, entry in list(self._store.items()):
            cached_cmd  = entry["command"]
            cached_hash = entry["screen_hash"]
            cached_act  = entry.get("action")
            cached_tgt  = entry.get("target")
            resp        = entry["response"]

            # Condition (c): same classifier hint intent
            if action_hint is not None and cached_act != action_hint:
                continue

            # Target constraint: reject hit if targets are explicitly conflicting
            if target_hint is not None and cached_tgt is not None:
                if str(cached_tgt).lower() != str(target_hint).lower():
                    continue

            # Condition (b): screen hash within CACHE_HASH_MAX_HAMMING bits
            dist = _hamming_distance(cached_hash, screen_hash)
            if dist > max_hamming:
                continue

            # Condition (a): command match
            sim = 1.0
            if semantic_enabled:
                cached_emb = entry.get("cmd_emb")
                if cached_emb is None:
                    cached_emb = _embed(cached_cmd)
                    entry["cmd_emb"] = cached_emb
                sim = _cosine(q_emb, cached_emb)
                if sim < semantic_min_sim:
                    continue
            else:
                if norm_cmd != cached_cmd:
                    continue

            if sim > best_sim or (sim == best_sim and dist < best_hamming):
                best_sim     = sim
                best_hamming = dist
                best_resp    = resp
                best_id      = entry_id

        if best_id is not None and best_resp is not None:
            self._store.move_to_end(best_id)
            return True, {
                **best_resp,
                "_cache_similarity": round(best_sim, 4),
                "_cache_hamming": best_hamming,
            }

        return False, None

    def store(self, command: str, screen_hash: str, response: dict,
              action: str = None, target: str = None) -> None:
        """
        Add a (command, screen_hash, action, target) → response entry.
        Never stores if screen_hash is 'display_unavailable' or empty.
        Bypasses window control button intents.
        """
        if not screen_hash or screen_hash == "display_unavailable":
            return

        act = action or response.get("action")
        tgt = target if target is not None else response.get("target")

        # Hard policy: window control button intents never store in semantic cache
        if act in ("close_button", "minimize_button", "maximize_button"):
            return

        norm_cmd = command.lower().strip()
        cmd_emb = None
        import route3_config
        if getattr(route3_config, "CACHE_SEMANTIC_ENABLED", False):
            cmd_emb = _embed(norm_cmd)

        self._counter += 1
        self._store[str(self._counter)] = {
            "command": norm_cmd,
            "screen_hash": screen_hash,
            "action": act,
            "target": tgt,
            "response": response,
            "cmd_emb": cmd_emb,
        }

        # LRU eviction
        while len(self._store) > self._max:
            self._store.popitem(last=False)   # remove least-recently-used

    @property
    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ── Module-level singleton (session-only) ──────────────────────────────────────
_cache: SemanticCache | None = None

def get_cache() -> SemanticCache:
    """Return the shared session cache instance (lazy init)."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
