"""
command_logger.py  --  Structured execution logger for ScreenSense

Records every command execution to a JSONL log file:
  logs/commands_YYYY-MM-DD.jsonl

Each entry contains:
  timestamp      : ISO-8601 UTC string
  raw_command    : the raw string the user spoke/typed
  action         : intent label returned by nlp.parse_command()
  target         : slot value (app name, number, etc.)
  confidence     : k-NN confidence score [0.0 – 1.0]
  route          : "knn" | "vlm_fallback" | "regex_fallback" | "exit" | "vision" | "fast_path" | "asr_failure"
  result         : "ok" | "not_found" | "error" | "low_confidence" | "button_not_found" | "asr_failure"
  target_hwnd    : (optional) resolved window handle
  target_title   : (optional) resolved window title
  action_taken   : (optional) specific automation action executed
  error_msg      : (optional) exception message if result == "error"
  duration_ms    : wall-clock time for the whole run_command() call

Usage:
    from command_logger import CommandLogger
    logger = CommandLogger()

    with logger.log_command(raw_command, action, target, confidence, route) as entry:
        # run your automation here
        ...   # entry.result is set automatically: "ok" if no exception
              # entry.mark_result("not_found") to override

Design:
    - One JSONL file per calendar day, auto-rotated at midnight
    - Thread-safe (uses a file lock via threading.Lock)
    - Never raises — logging failure is silently swallowed so a logger bug
      never kills the main loop
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("logs")


class _LogEntry:
    """Mutable entry built during a command execution."""
    def __init__(self, raw_command, action, target, confidence, route,
                 target_hwnd=None, target_title=None, action_taken=None):
        self.ts           = datetime.now(timezone.utc).isoformat()
        self.raw_command  = raw_command
        self.action       = action
        self.target       = str(target) if target is not None else None
        self.confidence   = round(float(confidence), 4)
        self.route        = route
        self.result       = "ok"
        self.error_msg    = None
        self.target_hwnd  = target_hwnd
        self.target_title = target_title
        self.action_taken = action_taken
        self._start       = time.perf_counter()

    def set_window(self, hwnd: int | None, title: str | None = None):
        """Record the resolved window handle and title."""
        self.target_hwnd  = hwnd
        self.target_title = title

    def set_action_taken(self, action_taken: str | None):
        """Record the actual automation dispatch action executed."""
        self.action_taken = action_taken

    def mark_result(self, result: str, error_msg: str = None):
        """Call inside the `with` block to override the default 'ok' result."""
        self.result    = result
        self.error_msg = error_msg

    def to_dict(self):
        duration_ms = round((time.perf_counter() - self._start) * 1000, 1)
        d = {
            "timestamp":    self.ts,
            "raw_command":  self.raw_command,
            "action":       self.action,
            "target":       self.target,
            "confidence":   self.confidence,
            "route":        self.route,
            "result":       self.result,
            "duration_ms":  duration_ms,
            "target_hwnd":  self.target_hwnd,
            "target_title": self.target_title,
            "action_taken": self.action_taken,
        }
        if self.error_msg:
            d["error_msg"] = self.error_msg
        return d


class CommandLogger:
    def __init__(self, log_dir: str | Path = LOG_DIR):
        self._log_dir  = Path(log_dir)
        self._lock     = threading.Lock()
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self._log_dir / f"commands_{today}.jsonl"

    def _write(self, entry_dict: dict):
        try:
            with self._lock:
                with open(self._log_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never crash the caller due to a logging error
            print(f"[Logger] Write error (non-fatal): {e}")

    @contextmanager
    def log_command(self,
                    raw_command: str,
                    action:      str | None,
                    target,
                    confidence:  float,
                    route:       str):
        """
        Context manager that wraps a command execution.

        Usage:
            with logger.log_command(cmd, action, target, conf, route) as entry:
                run_automation(action, target)
                # entry.mark_result("not_found") if needed

        The entry is written when the `with` block exits.
        If an exception is raised inside the block, result is set to "error"
        and the exception is re-raised after logging.
        """
        entry = _LogEntry(raw_command, action, target, confidence, route)
        try:
            yield entry
        except Exception as exc:
            entry.mark_result("error", error_msg=str(exc))
            self._write(entry.to_dict())
            raise
        else:
            self._write(entry.to_dict())

    def log_simple(self,
                   raw_command: str,
                   action:      str | None,
                   target,
                   confidence:  float,
                   route:       str,
                   result:      str = "ok",
                   error_msg:   str = None,
                   duration_ms: float = 0.0,
                   target_hwnd: int | None = None,
                   target_title: str | None = None,
                   action_taken: str | None = None):
        """One-shot log for callers that don't use the context manager."""
        entry = _LogEntry(raw_command, action, target, confidence, route,
                          target_hwnd=target_hwnd, target_title=target_title, action_taken=action_taken)
        entry.mark_result(result, error_msg)
        d = entry.to_dict()
        d["duration_ms"] = duration_ms
        self._write(d)

    # ── Convenience readers ──────────────────────────────────────────────────
    def log_wakeword(self,
                     score: float,
                     threshold: float,
                     rms: float,
                     status: str,
                     audio_energy: float = None):
        """Log wake-word detection scores (triggered, rejected, or silence)."""
        today = datetime.now().strftime("%Y-%m-%d")
        ww_path = self._log_dir / f"wakeword_{today}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": round(float(score), 4),
            "threshold": round(float(threshold), 4),
            "rms": round(float(rms), 2),
            "status": status,
        }
        if audio_energy is not None:
            entry["energy"] = round(float(audio_energy), 2)
        try:
            with self._lock:
                with open(ww_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logger] Wakeword write error (non-fatal): {e}")

    def read_today(self) -> list[dict]:
        """Return all log entries from today as a list of dicts."""
        return self._read_file(self._log_path())

    def read_file(self, date_str: str) -> list[dict]:
        """Read a specific day's log. date_str format: 'YYYY-MM-DD'."""
        return self._read_file(self._log_dir / f"commands_{date_str}.jsonl")

    def read_wakeword_file(self, date_str: str) -> list[dict]:
        """Read a specific day's wakeword log. date_str format: 'YYYY-MM-DD'."""
        return self._read_file(self._log_dir / f"wakeword_{date_str}.jsonl")

    def _read_file(self, path: Path) -> list[dict]:
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

    def summary(self, date_str: str = None) -> dict:
        """
        Return aggregate stats for a day (default: today).
        Useful for the evaluation harness.
        """
        entries = self.read_file(date_str) if date_str else self.read_today()
        if not entries:
            return {}

        total     = len(entries)
        by_route  = {}
        by_result = {}
        by_action = {}
        conf_sum  = 0.0

        for e in entries:
            by_route[e.get("route",  "unknown")] = by_route.get(e.get("route",  "unknown"), 0) + 1
            by_result[e.get("result","unknown")] = by_result.get(e.get("result","unknown"), 0) + 1
            by_action[e.get("action","unknown")] = by_action.get(e.get("action","unknown"), 0) + 1
            conf_sum += e.get("confidence", 0.0)

        return {
            "total_commands": total,
            "mean_confidence": round(conf_sum / total, 4),
            "by_route":  by_route,
            "by_result": by_result,
            "top_actions": sorted(by_action.items(), key=lambda x: -x[1])[:10],
        }
