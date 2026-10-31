"""
monitor.py - Part 3: acceptance-rate monitoring with capped self-adjustment.

Compares the clear rate over the last 30 days (current window) against the
30 days before that (baseline window), then either:
  - no action        (shift < 5 points)
  - one capped step  (5 to 15 points), logged to adjustments.jsonl
  - escalate         (> 15 points, or fewer than MIN_DECISIONS in either window)

Reads:  decisions.jsonl, thresholds.json (created with defaults if missing)
Writes: thresholds.json, adjustments.jsonl, escalations.jsonl
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DECISIONS_FILE = Path("decisions.jsonl")
THRESHOLDS_FILE = Path("thresholds.json")
ADJUSTMENTS_FILE = Path("adjustments.jsonl")
ESCALATIONS_FILE = Path("escalations.jsonl")

WINDOW_DAYS = 30
MIN_DECISIONS = 20          # per window; below this we escalate instead of acting
NO_ACTION_BELOW = 5.0       # points
ESCALATE_ABOVE = 15.0       # points

# The one threshold this monitor is allowed to touch (see decision notes).
TUNABLE = "high_dbr"
DEFAULT_THRESHOLDS = {"high_dbr": 0.45, "reference_rate": 70.0}  # reference_rate: fixed fallback in percent, used when the baseline window is empty
STEP = 0.01                 # one capped step per run
BOUNDS = {"high_dbr": (0.35, 0.45)}  # upper bound = SAMA 45% total-obligations cap; lower bound is our own safety margin


RIYADH = timezone(timedelta(hours=3))


def now_utc():
    return datetime.now(timezone.utc)


def parse_ts(value):
    """Parse ISO timestamps, including ones ending in Z."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # check_form.py writes local Riyadh time without a zone marker
        dt = dt.replace(tzinfo=RIYADH)
    return dt


def load_decisions():
    rows = []
    if not DECISIONS_FILE.exists():
        return rows
    with DECISIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupted lines, don't crash the monitor
    return rows


def load_thresholds():
    if THRESHOLDS_FILE.exists():
        with THRESHOLDS_FILE.open(encoding="utf-8") as f:
            thresholds = json.load(f)
        for key, value in DEFAULT_THRESHOLDS.items():
            thresholds.setdefault(key, value)  # older files get new keys automatically
        return thresholds
    save_thresholds(DEFAULT_THRESHOLDS)
    return dict(DEFAULT_THRESHOLDS)


def save_thresholds(thresholds):
    with THRESHOLDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def window_stats(rows, start, end):
    """Return (total, clear_count, clear_rate_percent) for decisions in [start, end)."""
    total = 0
    clear = 0
    for r in rows:
        ts = parse_ts(r.get("timestamp"))
        if ts is None or not (start <= ts < end):
            continue
        total += 1
        if str(r.get("decision", "")).upper() == "CLEAR":
            clear += 1
    rate = (clear / total * 100) if total else 0.0
    return total, clear, rate


def main():
    now = now_utc()
    current_start = now - timedelta(days=WINDOW_DAYS)
    baseline_start = now - timedelta(days=2 * WINDOW_DAYS)

    rows = load_decisions()
    cur_total, cur_clear, cur_rate = window_stats(rows, current_start, now + timedelta(seconds=1))
    base_total, base_clear, base_rate = window_stats(rows, baseline_start, current_start)

    # Choose the reference: the baseline window if it has data, otherwise the fixed reference_rate.
    thresholds = load_thresholds()
    reference = thresholds.get("reference_rate")
    if base_total >= MIN_DECISIONS:
        base_source = "window"
        enough = cur_total >= MIN_DECISIONS
    elif reference is not None:
        base_source = "reference_rate"
        base_rate = float(reference)
        enough = cur_total >= MIN_DECISIONS
    else:
        base_source = "none"
        enough = False

    # A shift only means something when there is a reference and enough current decisions.
    shift = (cur_rate - base_rate) if enough else None
    abs_shift = abs(shift) if enough else 0.0
    summary = {
        "run_at": now.isoformat(),
        "current_window": {"days": WINDOW_DAYS, "decisions": cur_total, "clear": cur_clear, "clear_rate": round(cur_rate, 2)},
        "baseline_window": {"days": WINDOW_DAYS, "decisions": base_total, "clear": base_clear, "clear_rate": round(window_stats(rows, baseline_start, current_start)[2], 2)},
        "baseline_source": base_source,
        "reference_rate": reference,
        "shift_points": round(shift, 2) if enough else None,
    }
    print("Acceptance monitor report")
    print(json.dumps(summary, indent=2))

    # Guard: not enough data in either window -> escalate, never adjust.
    if not enough:
        append_jsonl(ESCALATIONS_FILE, {**summary, "reason": f"fewer than {MIN_DECISIONS} decisions in a window; shift not computed"})
        print("ESCALATED: not enough decisions to judge a shift.")
        return

    # Guard: large shift -> escalate to a human.
    if abs_shift > ESCALATE_ABOVE:
        append_jsonl(ESCALATIONS_FILE, {**summary, "reason": f"shift {shift:+.2f} points exceeds {ESCALATE_ABOVE}"})
        print("ESCALATED: shift is larger than the allowed band.")
        return

    # Small shift -> no action.
    if abs_shift < NO_ACTION_BELOW:
        print("No action: shift is within the normal band.")
        return

    # Medium shift (5 to 15 points) -> one capped, logged step on one threshold.
    thresholds = load_thresholds()
    old_value = thresholds[TUNABLE]
    # If acceptance rose, tighten (lower the DBR limit); if it fell, loosen it.
    direction = -1 if shift > 0 else 1
    low, high = BOUNDS[TUNABLE]
    new_value = min(high, max(low, round(old_value + direction * STEP, 4)))

    if new_value == old_value:
        append_jsonl(ESCALATIONS_FILE, {**summary, "reason": f"{TUNABLE} already at its bound, cannot adjust"})
        print("ESCALATED: threshold is already at its allowed limit.")
        return

    thresholds[TUNABLE] = new_value
    save_thresholds(thresholds)
    append_jsonl(ADJUSTMENTS_FILE, {
        **summary,
        "threshold": TUNABLE,
        "old_value": old_value,
        "new_value": new_value,
        "step": direction * STEP,
    })
    print(f"ADJUSTED: {TUNABLE} {old_value} -> {new_value} (logged).")


if __name__ == "__main__":
    main()
