"""
route3_logger.py  --  Structured JSONL logging for Route 3 (VLM fallback).

Each Route 3 invocation writes two log entries to logs/route3_YYYY-MM-DD.jsonl:
  1. "invocation" entry — written by route3_handle() before dispatch
     Fields: timestamp, call_id, command, action_hint, nlp_confidence,
             screen_hash, cache_hit, action_taken, target_taken,
             reasoning, vlm_confidence, latency_ms, error
  2. "verification" entry — written by _run_action() after dispatch
     Fields: timestamp, call_id, verification_outcome, change_pct

Joining by call_id gives the full picture of what happened for each Route 3 event.

No metrics (hit rate, latency percentiles) are computed here.
That is the job of the evaluation harness (separate task, later).
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from route3_config import ROUTE3_LOG_DIR


class Route3Logger:
    """Thread-safe JSON-Lines logger for Route 3 events."""

    def __init__(self, log_dir: str = ROUTE3_LOG_DIR):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock    = threading.Lock()

    def _log_path(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self._log_dir / f"route3_{today}.jsonl"

    def _write(self, entry: dict) -> None:
        try:
            with self._lock:
                with open(self._log_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never crash the caller due to a logging error
            print(f"[Route3Logger] Write error (non-fatal): {e}")

    def log_invocation(self, *,
                       call_id:       str,
                       command:       str,
                       action_hint:   str,
                       confidence:    float,
                       screen_hash:   str,
                       cache_hit:     bool,
                       action_taken:  str | None,
                       target_taken,
                       reasoning:     str | None  = None,
                       vlm_confidence: float | None = None,
                       latency_ms:    float,
                       error:         str | None  = None) -> None:
        """Log a Route 3 invocation (pre-dispatch)."""
        entry = {
            "type":            "invocation",
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "call_id":         call_id,
            "command":         command,
            "action_hint":     action_hint,
            "nlp_confidence":  round(float(confidence), 4),
            "screen_hash":     screen_hash,
            "cache_hit":       cache_hit,
            "action_taken":    action_taken,
            "target_taken":    str(target_taken) if target_taken is not None else None,
            "latency_ms":      round(float(latency_ms), 1),
        }
        if reasoning      is not None: entry["reasoning"]       = reasoning
        if vlm_confidence is not None: entry["vlm_confidence"]  = round(float(vlm_confidence), 4)
        if error          is not None: entry["error"]           = error
        self._write(entry)

    def log_verification(self, *,
                         call_id:              str,
                         verification_outcome: str,
                         change_pct:           float) -> None:
        """Log the post-action verification outcome (after dispatch)."""
        entry = {
            "type":                 "verification",
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "call_id":              call_id,
            "verification_outcome": verification_outcome,
            "change_pct":           round(float(change_pct), 4),
        }
        self._write(entry)

    def read_today(self) -> list[dict]:
        return self._read_file(self._log_path())

    def read_date(self, date_str: str) -> list[dict]:
        """Read a specific day's log.  date_str: 'YYYY-MM-DD'."""
        return self._read_file(self._log_dir / f"route3_{date_str}.jsonl")

    def _read_file(self, path: Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries


# ── Module-level singleton ─────────────────────────────────────────────────────
_route3_logger: Route3Logger | None = None

def get_route3_logger() -> Route3Logger:
    global _route3_logger
    if _route3_logger is None:
        _route3_logger = Route3Logger()
    return _route3_logger
