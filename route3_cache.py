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
)

# ── Lazy encoder ───────────────────────────────────────────────────────────────
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _embed(text: str) -> np.ndarray:
    enc = _get_encoder()
    v   = enc.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    # L2-normalize so cosine similarity == dot product
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ── SemanticCache ──────────────────────────────────────────────────────────────

class SemanticCache:
    """
    In-memory LRU semantic cache for Route 3 VLM responses.

    The cache is keyed by the embedding of a composite string:
        "intent:{action} | target:{target} | cmd:{command} [screen:{screen_hash}]"
    so that the same command issued against a different screen state
    will NOT match (assuming the screen hash changes meaningfully).

    Similarity threshold and max entries are set in route3_config.py
    and passed at construction time for testability.
    """

    def __init__(self,
                 similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
                 max_entries:          int   = CACHE_MAX_ENTRIES):
        self._threshold     = similarity_threshold
        self._max           = max_entries
        # OrderedDict: entry_id str → (embedding np.ndarray, response dict)
        self._store: OrderedDict[str, tuple[np.ndarray, dict]] = OrderedDict()
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
        Requires cached action to match action_hint when action_hint is provided.

        Returns:
          (True,  cached_response_dict)  on hit  — response includes
                                                    '_cache_similarity' key
          (False, None)                  on miss
        """
        # Hard policy: window control button intents never query the semantic cache
        if action_hint in ("close_button", "minimize_button", "maximize_button"):
            return False, None

        key_text = self._cache_key_text(command, screen_hash, action_hint, target_hint)
        q_emb    = _embed(key_text)

        best_sim   = -1.0
        best_resp  = None
        best_id    = None

        for entry_id, (emb, resp) in self._store.items():
            cached_act = resp.get("action")
            cached_tgt = resp.get("target")

            # Hard constraint: reject hit if cached action mismatches current action_hint
            if action_hint is not None and cached_act != action_hint:
                continue

            # Target constraint: reject hit if targets are explicitly conflicting
            if target_hint is not None and cached_tgt is not None:
                if str(cached_tgt).lower() != str(target_hint).lower():
                    continue

            sim = _cosine(q_emb, emb)
            if sim > best_sim:
                best_sim  = sim
                best_resp = resp
                best_id   = entry_id

        if best_sim >= self._threshold and best_id is not None:
            # Update LRU order: move hit to end (most-recently-used)
            self._store.move_to_end(best_id)
            return True, {**best_resp, "_cache_similarity": round(best_sim, 4)}

        return False, None

    def store(self, command: str, screen_hash: str, response: dict,
              action: str = None, target: str = None) -> None:
        """
        Add a (command, screen_hash, action, target) → response entry.
        Evicts the oldest entry if over the LRU limit.
        Bypasses window control button intents.
        """
        act = action or response.get("action")
        tgt = target if target is not None else response.get("target")

        # Hard policy: window control button intents never store in semantic cache
        if act in ("close_button", "minimize_button", "maximize_button"):
            return

        key_text = self._cache_key_text(command, screen_hash, act, tgt)
        emb      = _embed(key_text)

        self._counter += 1
        self._store[str(self._counter)] = (emb, response)

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
