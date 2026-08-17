"""
route3_config.py  --  All configuration for Route 3 (VLM Fallback).

Edit values here to tune the system. All design choices are flagged
as "chosen, pending tuning" until real traffic data is available.

Feature flags can be set to False for controlled ablation testing
(mirrors the ablation study approach used for the NLP classifier).
"""

# ── Feature flags ──────────────────────────────────────────────────────────────
ROUTE3_CACHE_ENABLED   = True    # Set False to disable semantic cache (ablation)
ROUTE3_VERIFY_ENABLED  = True    # Set False to disable screenshot-diff verification

# ── Semantic cache ─────────────────────────────────────────────────────────────
# Similarity threshold for a cache hit.
# Chosen: 0.80.  Rationale: high enough to avoid false positives on different
# commands that happen to be textually similar; low enough to catch genuine
# paraphrase reuse.  PENDING TUNING from real cache-hit/miss logs.
CACHE_SIMILARITY_THRESHOLD = 0.80

# Maximum number of entries before LRU eviction.
# Chosen: 500.  PENDING TUNING — depends on observed command variety in practice.
CACHE_MAX_ENTRIES = 500

# Path for optional disk persistence of the cache across sessions.
# Currently NOT loaded on startup (session-only default — see open question below).
CACHE_PERSIST_PATH = "logs/route3_cache.json"

# ── Screenshot verification ─────────────────────────────────────────────────────
# Seconds to wait after action before capturing the post-action screenshot.
# Chosen: 1.5s.  PENDING TUNING — may need to increase for slow animations.
VERIFY_DELAY_SECONDS = 1.5

# Fraction of screen pixels that must change (per-pixel intensity delta > 10)
# for the outcome to be classified as 'verified'.
# Chosen: 2%.  PENDING TUNING from real verification logs.
VERIFY_CHANGE_THRESHOLD = 0.02

# Fraction above which the outcome is 'unverified_unexpected_change'.
# Chosen: 50%.  PENDING TUNING.
VERIFY_UNEXPECTED_THRESHOLD = 0.50

# ── Gemini VLM ─────────────────────────────────────────────────────────────────
# Model to call for vision-language fallback.
# Chosen: gemini-2.0-flash for speed and cost.  PENDING performance review.
VLM_MODEL_NAME = "gemini-2.0-flash"

# Environment variable (or .env key) holding the Gemini API key.
VLM_API_KEY_ENV = "GEMINI_API_KEY"

# ── Logging ────────────────────────────────────────────────────────────────────
ROUTE3_LOG_DIR = "logs"

# ── Open design question: cache invalidation / expiry ─────────────────────────
# Screen states change between sessions.  A cached entry for "close chrome"
# from a session where Chrome was open could fire incorrectly if Chrome is
# not open in a new session (depending on screen hash similarity).
#
# Options (not yet decided — flagged for review once real traffic data exists):
#   (a) Session-only cache: flush on exit.  CURRENTLY USED — safe default.
#   (b) Time-based TTL: evict entries older than N hours.
#   (c) Require exact screen-hash match: cache hit requires both semantic
#       command similarity >= threshold AND identical screen hash.
#       This is the most conservative option (effectively per-screen-state cache).
#
# The current implementation uses option (a).  The cache object is constructed
# in-memory at startup and not persisted back to disk unless explicitly called.
