"""
model_loader.py  --  Offline-first sentence transformer loader.
Tries local cache first to avoid unauthenticated Hugging Face Hub warnings on startup.
Falls back to online load if model is not yet cached.
Never sets or reads HF_TOKEN.
"""

import os
import inspect
from sentence_transformers import SentenceTransformer


def load_sentence_encoder(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer offline-first.
    Attempts local-only loading first (via local_files_only=True if supported,
    or temporary HF_HUB_OFFLINE=1 environment variable).
    Falls back to normal download/load on failure.
    """
    sig = inspect.signature(SentenceTransformer.__init__)
    supports_local_files_only = "local_files_only" in sig.parameters

    # Try 1: local_files_only parameter if supported
    if supports_local_files_only:
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            pass

    # Try 2: HF_HUB_OFFLINE environment variable set temporarily during load
    old_env = os.environ.get("HF_HUB_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return SentenceTransformer(model_name)
    except Exception:
        pass
    finally:
        if old_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_env

    # Fallback: normal load (may connect to Hugging Face Hub if online)
    return SentenceTransformer(model_name)

