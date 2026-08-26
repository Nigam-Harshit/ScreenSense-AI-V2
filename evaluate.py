"""
evaluate.py  --  Route 3 evaluation harness for ScreenSense V2.

Reads real Route 3 JSONL logs and prints a three-section report:
  Section 1 -- Overall command pipeline stats     (commands_*.jsonl)
  Section 2 -- Route 3 VLM / verification metrics (route3_*.jsonl)
  Section 3 -- Cache invalidation policy analysis (route3_*.jsonl)

Usage:
    python evaluate.py                    # last 7 days (default)
    python evaluate.py --days 30
    python evaluate.py --from 2026-08-01 --to 2026-08-26
    python evaluate.py --json results.json
    python evaluate.py --self-test        # built-in synthetic-data verification

No new dependencies -- stdlib only (json, pathlib, statistics, collections,
datetime, argparse, math, sys).
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

# -- Tuning constants -----------------------------------------------------------

LOG_DIR = Path("logs")

# Verified outcomes faster than this (ms) are flagged as potential false
# positives -- two polls fired within one sleep cycle, UI may still be
# animating.  Principled choice: exactly one poll cycle = VERIFY_POLL_INTERVAL_SECONDS * 1000.
FP_FLAG_THRESHOLD_MS = 250

# Minimum Route 3 invocations required before a policy recommendation is made.
POLICY_MIN_INVOCATIONS = 20

# Section 3 policy thresholds (heuristics -- see rec_rationale in output).
POLICY_HIT_RATE_MIN        = 0.10   # below this -> session-only recommended
POLICY_TTL_CLUSTER_HOURS   = 4.0    # median inter-occurrence gap -> informs TTL window
POLICY_EXACT_HASH_MIN_FRAC = 0.70   # fraction of hits sharing top hash -> exact-hash candidate

# Output formatting
SEP_WIDE  = "-" * 72
SEP_MID   = "-" * 54
SEP_SMALL = "-" * 38


# -- Percentile helper (no scipy/numpy needed) ----------------------------------

def _pct(data: list, p: float):
    """
    Return the p-th percentile of data (p in 0-100).
    Uses linear interpolation. Returns None for empty lists.
    """
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _fmt(val, fmt=".1f", suffix="", null="--"):
    """Format a number, returning null string if val is None."""
    if val is None:
        return null
    return format(val, fmt) + suffix


def _row(label: str, val: str, width_l: int = 34, width_r: int = 14) -> None:
    print(f"  {label:<{width_l}} {val:>{width_r}}")


# -- Date utilities -------------------------------------------------------------

def _date_range(from_date: date, to_date: date):
    """Yield each date from from_date to to_date inclusive."""
    d = from_date
    while d <= to_date:
        yield d
        d += timedelta(days=1)


# -- JSONL loaders --------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
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


def load_command_logs(from_date: date, to_date: date) -> list[dict]:
    """Load all commands_YYYY-MM-DD.jsonl files in the date range."""
    entries = []
    for d in _date_range(from_date, to_date):
        entries.extend(_load_jsonl(LOG_DIR / f"commands_{d}.jsonl"))
    return entries


def load_route3_logs(from_date: date, to_date: date) -> list[dict]:
    """Load all route3_YYYY-MM-DD.jsonl files in the date range."""
    entries = []
    for d in _date_range(from_date, to_date):
        entries.extend(_load_jsonl(LOG_DIR / f"route3_{d}.jsonl"))
    return entries


def split_route3(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate route3 log into (invocations, verifications)."""
    invocations   = [e for e in entries if e.get("type") == "invocation"]
    verifications = [e for e in entries if e.get("type") == "verification"]
    return invocations, verifications


def join_route3(invocations: list[dict],
                verifications: list[dict]) -> list[dict]:
    """
    Join invocation + verification records by call_id.

    Invocations without a matching verification are included with
    verification fields absent (e.g. ROUTE3_VERIFY_ENABLED=False, or
    the app was killed before verification completed).
    """
    ver_by_id = {v["call_id"]: v for v in verifications if "call_id" in v}
    merged = []
    for inv in invocations:
        cid = inv.get("call_id")
        row = dict(inv)
        if cid and cid in ver_by_id:
            # Merge verification fields; don't overwrite invocation fields
            # (type, timestamp, call_id are omitted intentionally)
            for k, v in ver_by_id[cid].items():
                if k not in ("type", "timestamp", "call_id"):
                    row[k] = v
        merged.append(row)
    return merged


# -- Section 1 -- Overall command pipeline --------------------------------------

def analyse_commands(entries: list[dict]) -> dict:
    """Aggregate command log entries into pipeline-level stats."""
    if not entries:
        return {}

    total     = len(entries)
    by_route  = Counter(e.get("route", "unknown") for e in entries)
    by_result = Counter(e.get("result", "unknown") for e in entries)

    # VLM fallback rate: exclude exit commands from denominator
    actionable = total - by_route.get("exit", 0)
    vlm_rate   = (by_route.get("vlm_fallback", 0) / actionable) if actionable else 0.0

    # k-NN confidence -- meaningful only for the knn route
    knn_confs = [e["confidence"] for e in entries
                 if e.get("route") == "knn" and "confidence" in e]

    # Duration per route
    dur_by_route: dict[str, list] = defaultdict(list)
    for e in entries:
        if "duration_ms" in e:
            dur_by_route[e.get("route", "unknown")].append(e["duration_ms"])

    return {
        "total":         total,
        "by_route":      dict(by_route),
        "by_result":     dict(by_result),
        "vlm_rate":      vlm_rate,
        "knn_conf_mean": mean(knn_confs)    if knn_confs else None,
        "knn_conf_p50":  _pct(knn_confs, 50),
        "knn_conf_p95":  _pct(knn_confs, 95),
        "dur_by_route":  {r: {"mean": mean(v), "p95": _pct(v, 95), "n": len(v)}
                          for r, v in dur_by_route.items()},
    }


# -- Section 2 -- Route 3 VLM / verification deep-dive --------------------------

def analyse_route3(joined: list[dict]) -> dict:
    """Aggregate joined Route 3 records into VLM and verification metrics."""
    if not joined:
        return {}

    total      = len(joined)
    cache_hits = sum(1 for r in joined if r.get("cache_hit") is True)
    hit_rate   = cache_hits / total

    # Latency split by cache hit vs. miss
    lat_hit  = [r["latency_ms"] for r in joined
                if r.get("cache_hit") is True  and "latency_ms" in r]
    lat_miss = [r["latency_ms"] for r in joined
                if r.get("cache_hit") is False and "latency_ms" in r]

    # VLM confidence -- cache misses only (hits don't call Gemini)
    vlm_confs = [r["vlm_confidence"] for r in joined
                 if r.get("cache_hit") is False and "vlm_confidence" in r]

    # Top actions taken
    action_counts = Counter(r["action_taken"] for r in joined
                            if r.get("action_taken"))

    # Error rate
    errors = sum(1 for r in joined if r.get("error"))

    # Verification outcome distribution
    # Rows without a verification_outcome field -> "no_verification"
    outcomes = Counter(r.get("verification_outcome", "no_verification")
                       for r in joined)

    # time_to_stable_ms -- verified rows only
    verified_rows  = [r for r in joined if r.get("verification_outcome") == "verified"]
    stable_ms_vals = [r["time_to_stable_ms"] for r in verified_rows
                      if r.get("time_to_stable_ms") is not None]
    timeout_count  = sum(1 for r in verified_rows
                         if r.get("time_to_stable_ms") is None)

    est_saving_ms  = (1500.0 - mean(stable_ms_vals)) if stable_ms_vals else None

    # False-positive risk: verified and stable_ms below one poll cycle
    fp_risk_count  = sum(1 for r in verified_rows
                         if r.get("time_to_stable_ms") is not None
                         and r["time_to_stable_ms"] < FP_FLAG_THRESHOLD_MS)

    # polls_taken (present from the adaptive-polling build onwards)
    polls_vals = [r["polls_taken"] for r in joined if "polls_taken" in r]

    def _lat_bucket(vals):
        return {"mean": mean(vals) if vals else None,
                "p50":  _pct(vals, 50),
                "p95":  _pct(vals, 95),
                "n":    len(vals)}

    return {
        "total":          total,
        "cache_hits":     cache_hits,
        "hit_rate":       hit_rate,
        "lat_hit":        _lat_bucket(lat_hit),
        "lat_miss":       _lat_bucket(lat_miss),
        "vlm_conf":       {"mean": mean(vlm_confs) if vlm_confs else None,
                           "p50":  _pct(vlm_confs, 50),
                           "p95":  _pct(vlm_confs, 95)},
        "top_actions":    action_counts.most_common(8),
        "errors":         errors,
        "error_rate":     errors / total,
        "outcomes":       dict(outcomes),
        "verified_n":     len(verified_rows),
        "stable_ms":      {"mean": mean(stable_ms_vals) if stable_ms_vals else None,
                           "p50":  _pct(stable_ms_vals, 50),
                           "p95":  _pct(stable_ms_vals, 95),
                           "n":    len(stable_ms_vals)},
        "est_saving_ms":  est_saving_ms,
        "timeout_count":  timeout_count,
        "fp_risk_count":  fp_risk_count,
        "polls":          {"mean": mean(polls_vals) if polls_vals else None,
                           "p95":  _pct(polls_vals, 95),
                           "n":    len(polls_vals)},
    }


# -- Section 3 -- Cache invalidation policy analysis ----------------------------

def analyse_cache_policy(invocations: list[dict]) -> dict:
    """
    Compute metrics to inform the cache invalidation policy decision.

    Three proxy metrics are labeled explicitly in output because the Route 3
    log records the screen_hash at query-time (hit or miss), but does NOT
    record the screen_hash that was used when the matching entry was originally
    stored.  Exact hash match between stored and retrieved entries cannot be
    measured directly from these logs -- see route3_cache.py SemanticCache.store()
    for the write path.
    """
    if not invocations:
        return {}

    total    = len(invocations)
    hits     = [r for r in invocations if r.get("cache_hit") is True]
    n_hits   = len(hits)
    hit_rate = n_hits / total

    # Screen hash diversity across all invocations
    all_hashes    = [r["screen_hash"] for r in invocations if r.get("screen_hash")]
    unique_hashes = len(set(all_hashes))
    hash_diversity = unique_hashes / len(all_hashes) if all_hashes else 0.0

    # Screen hashes seen at hit-time (proxy -- not the stored-entry hash)
    hit_hashes         = [r["screen_hash"] for r in hits if r.get("screen_hash")]
    unique_hit_hashes  = len(set(hit_hashes))
    hit_hash_diversity = unique_hit_hashes / len(hit_hashes) if hit_hashes else 0.0

    # Most common hash at hit-time -- proxy for exact-hash policy viability
    hit_hash_counter   = Counter(hit_hashes)
    top_hit_hash_count = hit_hash_counter.most_common(1)[0][1] if hit_hashes else 0
    top_hit_hash_frac  = top_hit_hash_count / n_hits if n_hits else 0.0

    # Inter-occurrence gap analysis: how often does the same command repeat?
    cmd_timestamps: dict[str, list] = defaultdict(list)
    for r in invocations:
        cmd = r.get("command", "").lower().strip()
        ts  = r.get("timestamp", "")
        if cmd and ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                cmd_timestamps[cmd].append(dt)
            except ValueError:
                pass

    gaps_hours = []
    for times in cmd_timestamps.values():
        if len(times) < 2:
            continue
        times.sort()
        for i in range(1, len(times)):
            gaps_hours.append((times[i] - times[i - 1]).total_seconds() / 3600.0)

    median_gap_h    = median(gaps_hours)  if gaps_hours else None
    commands_repeat = sum(1 for ts in cmd_timestamps.values() if len(ts) > 1)

    # Hit rate by calendar day (shows value of cross-session cache persistence)
    daily_hits:  dict[str, int] = defaultdict(int)
    daily_total: dict[str, int] = defaultdict(int)
    for r in invocations:
        day = r.get("timestamp", "")[:10]   # "YYYY-MM-DD"
        if day:
            daily_total[day] += 1
            if r.get("cache_hit"):
                daily_hits[day] += 1
    daily_hit_rates = {day: daily_hits[day] / daily_total[day]
                       for day in sorted(daily_total)}

    # -- Policy recommendation -------------------------------------------------
    # All thresholds are heuristics -- printed alongside raw numbers so the reader
    # can judge the recommendation against the data directly.
    if total < POLICY_MIN_INVOCATIONS:
        recommendation = "INSUFFICIENT DATA"
        rec_rationale  = (f"Only {total} Route 3 invocations; need >= {POLICY_MIN_INVOCATIONS} "
                          f"for a reliable recommendation")
    elif hit_rate < POLICY_HIT_RATE_MIN:
        recommendation = "SESSION-ONLY (option a)"
        rec_rationale  = (f"Hit rate {hit_rate:.1%} < {POLICY_HIT_RATE_MIN:.0%} "
                          f"-- cache provides little value; session-only avoids stale-entry risk")
    elif top_hit_hash_frac >= POLICY_EXACT_HASH_MIN_FRAC:
        recommendation = "EXACT-HASH MATCH (option c)"
        rec_rationale  = (f"{top_hit_hash_frac:.0%} of hits share the dominant screen_hash "
                          f"(proxy metric) -- exact-hash key would preserve hit rate while "
                          f"preventing cross-screen-state reuse")
    elif median_gap_h is not None and median_gap_h <= POLICY_TTL_CLUSTER_HOURS:
        ttl_h = max(int(math.ceil(median_gap_h * 3)), 1)   # 3× median gap, rounded up
        recommendation = f"TTL = {ttl_h}h (option b)"
        rec_rationale  = (f"Median command repeat gap {median_gap_h:.1f}h -- "
                          f"TTL of {ttl_h}h (3× median) would capture most natural reuse "
                          f"while expiring entries before major context shifts")
    else:
        recommendation = "SESSION-ONLY (option a)"
        rec_rationale  = ("No strong signal for TTL or exact-hash; "
                          "session-only is the safest conservative default")

    return {
        "total_invocations":  total,
        "hit_rate":           hit_rate,
        "unique_hashes":      unique_hashes,
        "hash_diversity":     hash_diversity,
        "unique_hit_hashes":  unique_hit_hashes,
        "hit_hash_diversity": hit_hash_diversity,
        "top_hit_hash_frac":  top_hit_hash_frac,
        "median_gap_h":       median_gap_h,
        "commands_repeat":    commands_repeat,
        "daily_hit_rates":    daily_hit_rates,
        "recommendation":     recommendation,
        "rec_rationale":      rec_rationale,
    }


# -- Report printer -------------------------------------------------------------

def print_report(cmd_stats: dict, r3_stats: dict, policy_stats: dict,
                 from_date: date, to_date: date) -> None:

    date_str = (str(from_date) if from_date == to_date
                else f"{from_date}  ->  {to_date}")

    print(f"\n{SEP_WIDE}")
    print(f"  ScreenSense V2 -- Route 3 Evaluation Harness")
    print(f"  Date range : {date_str}")
    print(f"{SEP_WIDE}")

    # -- Section 1 -------------------------------------------------------------
    print(f"\n  SECTION 1 -- OVERALL COMMAND PIPELINE")
    print(f"  {SEP_MID}")

    if not cmd_stats:
        print("  No command log data in date range.\n")
    else:
        s = cmd_stats
        _row("Total commands", str(s["total"]))
        print()
        for route in ("knn", "vlm_fallback", "regex_fallback", "exit", "unknown"):
            n = s["by_route"].get(route, 0)
            if n == 0:
                continue
            pct = n / s["total"] * 100
            _row(f"  Route: {route}", f"{n}  ({pct:.1f}%)")
        print()
        for result in ("ok", "not_found", "error", "low_confidence"):
            n = s["by_result"].get(result, 0)
            if n == 0:
                continue
            pct = n / s["total"] * 100
            _row(f"  Result: {result}", f"{n}  ({pct:.1f}%)")
        print()
        _row("VLM fallback rate",       _fmt(s["vlm_rate"] * 100, ".1f", "%"))
        print()
        _row("k-NN confidence (mean)",  _fmt(s["knn_conf_mean"], ".3f"))
        _row("k-NN confidence (P50)",   _fmt(s["knn_conf_p50"],  ".3f"))
        _row("k-NN confidence (P95)",   _fmt(s["knn_conf_p95"],  ".3f"))
        print()
        print(f"  {'Route':<24} {'Mean dur (ms)':>13} {'P95 dur (ms)':>13} {'N':>6}")
        print(f"  {SEP_SMALL}")
        for route, d in s.get("dur_by_route", {}).items():
            print(f"  {route:<24} {_fmt(d['mean'],'.1f'):>13} "
                  f"{_fmt(d['p95'],'.1f'):>13} {d['n']:>6}")

    # -- Section 2 -------------------------------------------------------------
    print(f"\n{SEP_WIDE}")
    print(f"  SECTION 2 -- ROUTE 3 VLM / VERIFICATION DEEP-DIVE")
    print(f"  {SEP_MID}")

    if not r3_stats:
        print("  No Route 3 log data in date range.\n")
    else:
        s = r3_stats
        _row("Total Route 3 invocations", str(s["total"]))
        _row("Cache hit rate",            _fmt(s["hit_rate"] * 100, ".1f", "%"))
        _row("Error rate",                _fmt(s["error_rate"] * 100, ".1f", "%"))
        print()

        # Latency
        print(f"  {'Latency':<28} {'Mean (ms)':>9} {'P50 (ms)':>9} {'P95 (ms)':>9} {'N':>5}")
        print(f"  {SEP_SMALL}")
        for label, d in (("Cache hit", s["lat_hit"]), ("Cache miss", s["lat_miss"])):
            print(f"  {label:<28} {_fmt(d['mean'],'.1f'):>9} "
                  f"{_fmt(d['p50'],'.1f'):>9} {_fmt(d['p95'],'.1f'):>9} {d['n']:>5}")
        print()

        # VLM confidence
        vc = s["vlm_conf"]
        _row("VLM confidence (mean)",  _fmt(vc["mean"], ".3f"))
        _row("VLM confidence (P50)",   _fmt(vc["p50"],  ".3f"))
        _row("VLM confidence (P95)",   _fmt(vc["p95"],  ".3f"))
        print()

        # Verification outcomes
        print("  Verification outcomes:")
        total_with_ver = sum(n for k, n in s["outcomes"].items()
                             if k != "no_verification")
        for outcome, n in sorted(s["outcomes"].items()):
            denom = total_with_ver if outcome != "no_verification" else s["total"]
            pct   = n / denom * 100 if denom else 0
            _row(f"    {outcome}", f"{n}  ({pct:.1f}%)")
        print()

        # time_to_stable_ms
        sm = s["stable_ms"]
        n_ver = s["verified_n"]
        print(f"  time_to_stable_ms  (verified outcomes, n={n_ver}):")
        _row("    Mean",         _fmt(sm["mean"], ".0f", " ms"))
        _row("    P50",          _fmt(sm["p50"],  ".0f", " ms"))
        _row("    P95",          _fmt(sm["p95"],  ".0f", " ms"))
        _row("    Timeouts",     f"{s['timeout_count']}  "
                                 f"(stable not reached within 1.5s ceiling)")
        _row("    Est. saving",  _fmt(s["est_saving_ms"], ".0f",
                                      " ms vs old fixed-1500ms baseline"))
        print()
        fp = s["fp_risk_count"]
        fp_tag = "!  review manually" if fp > 0 else "none"
        _row(f"  False-positive risk  (<{FP_FLAG_THRESHOLD_MS}ms = 1 poll cycle)",
             f"{fp}  ({fp_tag})")
        print()

        # polls_taken
        pv = s["polls"]
        _row("  polls_taken (mean)", _fmt(pv["mean"], ".1f"))
        _row("  polls_taken (P95)",  _fmt(pv["p95"],  ".1f"))
        print()

        # Top actions
        print("  Top Route 3 actions:")
        for action, n in s["top_actions"]:
            pct = n / s["total"] * 100
            _row(f"    {action}", f"{n}  ({pct:.1f}%)")

    # -- Section 3 -------------------------------------------------------------
    print(f"\n{SEP_WIDE}")
    print(f"  SECTION 3 -- CACHE INVALIDATION POLICY ANALYSIS")
    print(f"  {SEP_MID}")

    if not policy_stats:
        print("  No Route 3 invocation data in date range.\n")
    else:
        s = policy_stats
        _row("Total Route 3 invocations", str(s["total_invocations"]))
        _row("Cache hit rate",            _fmt(s["hit_rate"] * 100, ".1f", "%"))
        print()

        # Screen hash diversity
        _row("Unique screen_hashes (all invocations)", str(s["unique_hashes"]))
        _row("Hash diversity (unique / total)",
             _fmt(s["hash_diversity"] * 100, ".1f", "%"))
        print()

        # Proxy metrics -- explicitly labeled
        print("  Hit-time screen hash analysis  (proxy metrics -- see note below):")
        _row("    Unique hashes at hit-time",   str(s["unique_hit_hashes"]))
        _row("    Hit-time hash diversity",
             _fmt(s["hit_hash_diversity"] * 100, ".1f", "%"))
        _row("    Top hash fraction of hits",
             _fmt(s["top_hit_hash_frac"] * 100, ".1f", "%"))
        print(f"  NOTE: The log records the screen_hash at query-time, not at "
              f"store-time.\n"
              f"        These metrics estimate stored-entry hash diversity via "
              f"hit-time hashes.\n"
              f"        Treat as directional, not exact. See route3_cache.py "
              f"SemanticCache.store()\n"
              f"        for the write-path limitation.")
        print()

        # Repeat command timing
        _row("Commands with >1 occurrence",  str(s["commands_repeat"]))
        _row("Median inter-occurrence gap",
             (_fmt(s["median_gap_h"], ".1f", "h")
              if s["median_gap_h"] is not None else "--  (no repeated commands)"))
        print()

        # Per-day hit rates
        if s["daily_hit_rates"]:
            print("  Hit rate by day  (shows value of cross-session persistence):")
            for day, rate in s["daily_hit_rates"].items():
                bar = "#" * max(int(rate * 24), 0)
                _row(f"    {day}", f"{rate:.1%}  {bar}")
            print()

        # Policy recommendation -- raw numbers always printed alongside
        print(f"  {'-' * 68}")
        print(f"  RECOMMEND (heuristic, review before committing):")
        print(f"    {s['recommendation']}")
        print(f"  Rationale : {s['rec_rationale']}")
        print(f"  Raw inputs: "
              f"hit_rate={_fmt(s['hit_rate']*100,'.1f','%')}  "
              f"top_hash_frac={_fmt(s['top_hit_hash_frac']*100,'.1f','%')} (proxy)  "
              f"median_gap={_fmt(s['median_gap_h'],'.1f','h')}")
        print(f"  {'-' * 68}")
        print()

    print(f"{SEP_WIDE}\n")


# -- JSON export ----------------------------------------------------------------

def export_json(path: str, cmd_stats: dict, r3_stats: dict,
                policy_stats: dict, from_date: date, to_date: date) -> None:
    out = {
        "generated":    datetime.now(timezone.utc).isoformat(),
        "from_date":    str(from_date),
        "to_date":      str(to_date),
        "pipeline":     cmd_stats,
        "route3":       r3_stats,
        "cache_policy": policy_stats,
    }
    # Convert top_actions list-of-tuples to JSON-serialisable list-of-dicts
    if out["route3"] and "top_actions" in out["route3"]:
        out["route3"]["top_actions"] = [
            {"action": a, "count": n} for a, n in out["route3"]["top_actions"]
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved -> {path}\n")


# -- Self-test -- synthetic data verification ------------------------------------

def run_self_test() -> None:
    """
    Generate 25 synthetic log records in memory, run all three analysis
    functions, and assert every computed value against hand-calculated
    expected results.

    This is the canonical verification step described in the implementation
    plan -- it must pass before real traffic touches the harness.
    Exits with code 1 on any failure; prints each assertion result.
    """
    print(f"\n{SEP_WIDE}")
    print("  SELF-TEST -- Synthetic data verification (23 assertions)")
    print(f"  {SEP_MID}\n")

    failures: list[str] = []

    def _check(label: str, got, expected, tol: float = 0.0001) -> None:
        if got is None:
            ok = False
        else:
            try:
                ok = abs(float(got) - float(expected)) <= tol
            except (TypeError, ValueError):
                ok = False
        tag = "  PASS" if ok else "  FAIL"
        print(f"{tag}  {label}")
        if not ok:
            print(f"       got={got!r}  expected={expected!r}")
            failures.append(label)

    def _check_eq(label: str, got, expected) -> None:
        ok = (got == expected)
        tag = "  PASS" if ok else "  FAIL"
        print(f"{tag}  {label}")
        if not ok:
            print(f"       got={got!r}  expected={expected!r}")
            failures.append(label)

    # -- Synthetic command records (20 total) ----------------------------------
    # Breakdown: knn=10, vlm_fallback=6, regex_fallback=2, exit=2
    # knn confidences: 8 × 0.86, 2 × 0.72
    #   mean = (8*0.86 + 2*0.72) / 10 = (6.88 + 1.44) / 10 = 0.832
    # VLM fallback rate = 6 / (20 − 2) = 6/18 = 0.3333...
    _ts = "2026-08-26T06:00:00+00:00"
    cmd_entries = (
        [{"route": "knn",            "result": "ok",        "confidence": 0.86, "duration_ms": 120.0, "timestamp": _ts}] * 8 +
        [{"route": "knn",            "result": "not_found", "confidence": 0.72, "duration_ms": 150.0, "timestamp": _ts}] * 2 +
        [{"route": "vlm_fallback",   "result": "ok",        "confidence": 0.30, "duration_ms": 2200.0,"timestamp": _ts}] * 6 +
        [{"route": "regex_fallback", "result": "ok",        "confidence": 1.00, "duration_ms": 80.0,  "timestamp": _ts}] * 2 +
        [{"route": "exit",           "result": "ok",        "confidence": 1.00, "duration_ms": 10.0,  "timestamp": _ts}] * 2
    )

    print("  -- Section 1 ------------------------------------------------------")
    cs = analyse_commands(cmd_entries)
    _check_eq("total commands",       cs["total"],                       20)
    _check_eq("knn count",            cs["by_route"].get("knn", 0),      10)
    _check_eq("vlm_fallback count",   cs["by_route"].get("vlm_fallback",0), 6)
    _check   ("vlm_rate",             cs["vlm_rate"],                    6/18)
    _check   ("knn_conf_mean",        cs["knn_conf_mean"],               0.832)
    print()

    # -- Synthetic Route 3 records (8 invocations, 6 verifications) ------------
    # Hits (3): call_ids a1, a2, a3 -- latency_ms=15, screen_hash=hash_A
    # Misses (5): call_ids b1-b5 -- latency_ms=1200
    #
    # Verifications (6/8):
    #   a1: verified, stable_ms=300, polls=3
    #   a2: verified, stable_ms=450, polls=4
    #   a3: verified, stable_ms=600, polls=5
    #   b1: verified, stable_ms=200, polls=2   ← fp_risk (200 < 250ms)
    #   b2: unverified_no_change, polls=8
    #   b3: unverified_unexpected_change, polls=5
    #   b4, b5: no verification record -> "no_verification"
    #
    # Hand-calc:
    #   hit_rate = 3/8 = 0.375
    #   lat_hit mean = 15.0   lat_miss mean = 1200.0
    #   verified_n = 4
    #   stable_ms values: [300, 450, 600, 200] -> sorted [200,300,450,600]
    #     mean = 1550/4 = 387.5
    #     P50 = (300+450)/2 = 375.0
    #   est_saving_ms = 1500 − 387.5 = 1112.5
    #   fp_risk_count = 1 (200 < 250)
    #   polls_vals from all joined: [3,4,5,2,8,5] -> mean = 27/6 = 4.5
    #   outcomes: verified=4, no_change=1, unexpected=1, no_verification=2
    inv = [
        {"type":"invocation","call_id":"a1","cache_hit":True, "latency_ms":15.0,
         "screen_hash":"hash_A","timestamp":"2026-08-26T06:00:00+00:00",
         "command":"close chrome","action_taken":"close_button"},
        {"type":"invocation","call_id":"a2","cache_hit":True, "latency_ms":15.0,
         "screen_hash":"hash_A","timestamp":"2026-08-26T06:05:00+00:00",
         "command":"close chrome","action_taken":"close_button"},
        {"type":"invocation","call_id":"a3","cache_hit":True, "latency_ms":15.0,
         "screen_hash":"hash_A","timestamp":"2026-08-26T06:10:00+00:00",
         "command":"close chrome","action_taken":"close_button"},
        {"type":"invocation","call_id":"b1","cache_hit":False,"latency_ms":1200.0,
         "screen_hash":"hash_B","timestamp":"2026-08-26T07:00:00+00:00",
         "command":"open notepad","action_taken":"open_app",   "vlm_confidence":0.85},
        {"type":"invocation","call_id":"b2","cache_hit":False,"latency_ms":1200.0,
         "screen_hash":"hash_B","timestamp":"2026-08-26T08:00:00+00:00",
         "command":"maximize window","action_taken":"maximize_button","vlm_confidence":0.78},
        {"type":"invocation","call_id":"b3","cache_hit":False,"latency_ms":1200.0,
         "screen_hash":"hash_C","timestamp":"2026-08-26T09:00:00+00:00",
         "command":"scroll down","action_taken":"scroll_down", "vlm_confidence":0.90},
        {"type":"invocation","call_id":"b4","cache_hit":False,"latency_ms":1200.0,
         "screen_hash":"hash_C","timestamp":"2026-08-26T10:00:00+00:00",
         "command":"mute volume","action_taken":"mute_volume", "vlm_confidence":0.80},
        {"type":"invocation","call_id":"b5","cache_hit":False,"latency_ms":1200.0,
         "screen_hash":"hash_D","timestamp":"2026-08-26T11:00:00+00:00",
         "command":"close chrome","action_taken":"close_button","vlm_confidence":0.76},
    ]
    ver = [
        {"type":"verification","call_id":"a1","verification_outcome":"verified",
         "change_pct":0.05,"polls_taken":3,"time_to_stable_ms":300},
        {"type":"verification","call_id":"a2","verification_outcome":"verified",
         "change_pct":0.04,"polls_taken":4,"time_to_stable_ms":450},
        {"type":"verification","call_id":"a3","verification_outcome":"verified",
         "change_pct":0.06,"polls_taken":5,"time_to_stable_ms":600},
        {"type":"verification","call_id":"b1","verification_outcome":"verified",
         "change_pct":0.03,"polls_taken":2,"time_to_stable_ms":200},
        {"type":"verification","call_id":"b2","verification_outcome":"unverified_no_change",
         "change_pct":0.00,"polls_taken":8,"time_to_stable_ms":None},
        {"type":"verification","call_id":"b3","verification_outcome":"unverified_unexpected_change",
         "change_pct":0.60,"polls_taken":5,"time_to_stable_ms":None},
    ]

    print("  -- Section 2 ------------------------------------------------------")
    joined = join_route3(inv, ver)
    rs = analyse_route3(joined)

    _check_eq("total route3",          rs["total"],                    8)
    _check_eq("cache_hits",            rs["cache_hits"],               3)
    _check   ("hit_rate",              rs["hit_rate"],                 3/8)
    _check   ("lat_hit mean",          rs["lat_hit"]["mean"],          15.0)
    _check   ("lat_miss mean",         rs["lat_miss"]["mean"],         1200.0)
    _check_eq("verified_n",            rs["verified_n"],               4)
    _check   ("stable_ms mean",        rs["stable_ms"]["mean"],        387.5)
    _check   ("stable_ms P50",         rs["stable_ms"]["p50"],         375.0)
    _check   ("est_saving_ms",         rs["est_saving_ms"],            1112.5)
    _check_eq("fp_risk_count",         rs["fp_risk_count"],            1)
    _check   ("polls mean",            rs["polls"]["mean"],            4.5)
    _check_eq("outcome verified",      rs["outcomes"].get("verified"), 4)
    _check_eq("outcome no_change",     rs["outcomes"].get("unverified_no_change"), 1)
    _check_eq("outcome no_ver",        rs["outcomes"].get("no_verification"),      2)
    print()

    # -- Section 3 assertions --------------------------------------------------
    # inv has 8 rows; 3 hits (all hash_A); unique hashes = {A,B,C,D} = 4
    # "close chrome" appears 4× (a1,a2,a3,b5) with timestamps:
    #   06:00, 06:05, 06:10, 11:00 -> gaps: 5min, 5min, 290min
    #   in hours: 5/60≈0.0833, 5/60≈0.0833, 290/60≈4.833
    #   median of [0.0833, 0.0833, 4.833] = 0.0833h
    print("  -- Section 3 ------------------------------------------------------")
    ps = analyse_cache_policy(inv)

    _check_eq("policy total",           ps["total_invocations"],   8)
    _check   ("policy hit_rate",        ps["hit_rate"],            3/8)
    _check_eq("unique_hashes",          ps["unique_hashes"],       4)
    _check_eq("unique_hit_hashes",      ps["unique_hit_hashes"],   1)
    _check   ("top_hit_hash_frac",      ps["top_hit_hash_frac"],   1.0)
    _check   ("commands_repeat >= 1",    float(ps["commands_repeat"] >= 1), 1.0)
    # median_gap_h: sorted [0.0833, 0.0833, 4.833] -> median=0.0833h
    _check   ("median_gap_h",           ps["median_gap_h"],        5/60, tol=0.002)
    print()

    # -- Summary ---------------------------------------------------------------
    print(f"  {SEP_MID}")
    n_assertions = 23
    if failures:
        print(f"\n  SELF-TEST FAILED -- {len(failures)}/{n_assertions} assertion(s):\n")
        for label in failures:
            print(f"    X  {label}")
        print()
        sys.exit(1)
    else:
        print(f"\n  All {n_assertions} assertions passed.")
        print(f"  Aggregation logic verified against hand-calculated values.")
    print(f"{SEP_WIDE}\n")


# -- CLI entry point ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="ScreenSense V2 -- Route 3 evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python evaluate.py                    # last 7 days\n"
            "  python evaluate.py --days 30\n"
            "  python evaluate.py --from 2026-08-01 --to 2026-08-26\n"
            "  python evaluate.py --json results.json\n"
            "  python evaluate.py --self-test\n"
        ),
    )
    parser.add_argument(
        "--days", type=int, default=7, metavar="N",
        help="Analyse the last N calendar days (default: 7). Ignored if --from is set.",
    )
    parser.add_argument(
        "--from", dest="from_date", metavar="YYYY-MM-DD",
        help="Start date inclusive. Overrides --days.",
    )
    parser.add_argument(
        "--to", dest="to_date", metavar="YYYY-MM-DD",
        help="End date inclusive (default: today). Only meaningful with --from.",
    )
    parser.add_argument(
        "--json", dest="json_path", metavar="PATH",
        help="Save full results dict to a JSON file.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run built-in synthetic-data verification and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    # -- Resolve date range ----------------------------------------------------
    today = date.today()
    if args.from_date:
        try:
            from_date = date.fromisoformat(args.from_date)
        except ValueError:
            print(f"[ERROR] --from date '{args.from_date}' is not a valid "
                  f"YYYY-MM-DD date.")
            sys.exit(1)
        if args.to_date:
            try:
                to_date = date.fromisoformat(args.to_date)
            except ValueError:
                print(f"[ERROR] --to date '{args.to_date}' is not a valid "
                      f"YYYY-MM-DD date.")
                sys.exit(1)
        else:
            to_date = today
    else:
        to_date   = today
        from_date = today - timedelta(days=max(args.days - 1, 0))

    if from_date > to_date:
        print(f"[ERROR] --from ({from_date}) is after --to ({to_date}).")
        sys.exit(1)

    # -- Load ------------------------------------------------------------------
    cmd_entries = load_command_logs(from_date, to_date)
    r3_entries  = load_route3_logs(from_date,  to_date)

    if not cmd_entries and not r3_entries:
        print(f"\n  ScreenSense V2 -- Route 3 Evaluation Harness")
        print(f"  No log data found in '{LOG_DIR}/' for {from_date} -> {to_date}.")
        print(f"  Run main.py to generate logs, then re-run this harness.\n")
        sys.exit(0)

    # -- Analyse ---------------------------------------------------------------
    invocations, verifications = split_route3(r3_entries)
    joined                     = join_route3(invocations, verifications)

    cmd_stats    = analyse_commands(cmd_entries)
    r3_stats     = analyse_route3(joined)
    policy_stats = analyse_cache_policy(invocations)

    # -- Print -----------------------------------------------------------------
    print_report(cmd_stats, r3_stats, policy_stats, from_date, to_date)

    # -- Export ----------------------------------------------------------------
    if args.json_path:
        export_json(args.json_path, cmd_stats, r3_stats, policy_stats,
                    from_date, to_date)


if __name__ == "__main__":
    main()
