import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta

LOG_FILE = "decisions.jsonl"
LABELS_FILE = "labels.csv"
PROPOSALS_FILE = "proposals.json"

# Thresholds (placeholders: verify DBR limit before the demo)
DBR_LIMIT = 0.45
VELOCITY_LIMIT = 3
VELOCITY_WINDOW = timedelta(days=7)

# Rules that have a numeric threshold, and how far one proposal may move it
TUNABLE = {
    "high_dbr": {"threshold": DBR_LIMIT, "step": 0.05},
    "velocity": {"threshold": VELOCITY_LIMIT, "step": 1},
}

ALL_RULES = [
    "identity_mismatch",
    "income_mismatch",
    "stale_document",
    "duplicate_identity",
    "high_dbr",
    "velocity",
]

FP_LIMIT = 0.20  # propose loosening if more than 20% of a rule's flags are clean


def load_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["form"]: int(row["confirmed_fraud"]) for row in csv.DictReader(f)}


def load_decisions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_time(ts):
    return datetime.fromisoformat(ts)


def names_by_id(entries):
    names = defaultdict(set)
    for e in entries:
        nid = e["extracted"].get("national_id")
        name = e["extracted"].get("full_name")
        if nid and name:
            names[nid].add(name.lower())
    return names


def fired_rules(entries):
    """Return one set of fired rule names per log entry."""
    names = names_by_id(entries)
    results = []

    for e in entries:
        ext = e["extracted"]
        reason = e["reason"]
        nid = ext.get("national_id")
        fired = set()

        # Identity and name match against payroll records
        if "not found in payroll" in reason or "Name mismatch" in reason:
            fired.add("identity_mismatch")

        # Declared income vs salary certificate / payroll
        if "Income mismatch" in reason:
            fired.add("income_mismatch")

        # Document freshness
        if "older than 6 months" in reason or "Invalid form date" in reason:
            fired.add("stale_document")

        # Same ID submitted under more than one name
        if nid and len(names[nid]) > 1:
            fired.add("duplicate_identity")

        # Debt-to-income ratio (shadow mode)
        # Uses the system's obligations (shadow_dbr), not the form's
        shadow = ext.get("shadow_dbr")
        if shadow is not None and shadow > DBR_LIMIT:
            fired.add("high_dbr")

        # Application velocity per ID (shadow mode)
        if nid:
            t = parse_time(e["timestamp"])
            recent = [
                o for o in entries
                if o["extracted"].get("national_id") == nid
                and abs(t - parse_time(o["timestamp"])) <= VELOCITY_WINDOW
            ]
            if len(recent) > VELOCITY_LIMIT:
                fired.add("velocity")

        results.append(fired)
    return results


def evaluate(entries, labels):
    stats = defaultdict(lambda: {"flags": 0, "tp": 0, "fp": 0, "fn": 0})
    for e, fired in zip(entries, fired_rules(entries)):
        if e["form"] not in labels:
            continue  # unlabeled forms can't be scored
        is_fraud = labels[e["form"]] == 1

        for rule in fired:
            stats[rule]["flags"] += 1
            stats[rule]["tp" if is_fraud else "fp"] += 1

        # Fraud that no rule caught. We can't tell which rule should have caught it,
        # so this is tracked for the whole rule set.
        if is_fraud and not fired:
            stats["_all_rules"]["fn"] += 1
    return stats


def make_proposal(rule, reason):
    if rule in TUNABLE:
        cfg = TUNABLE[rule]
        return {
            "rule": rule,
            "action": "loosen",
            "current": cfg["threshold"],
            "proposed": cfg["threshold"] + cfg["step"],
            "reason": reason,
            "auto_apply": False,
        }
    # Binary rules are not threshold-based, so a human reviews them
    return {"rule": rule, "action": "review", "reason": reason, "auto_apply": False}


def propose(stats):
    proposals = []
    for rule in ALL_RULES:
        s = stats[rule]
        if s["flags"] and s["fp"] / s["flags"] > FP_LIMIT:
            rate = s["fp"] / s["flags"]
            proposals.append(make_proposal(rule, f"{rate:.0%} of flags were on clean forms"))

    missed = stats["_all_rules"]["fn"]
    if missed:
        proposals.append({
            "rule": "all",
            "action": "review",
            "reason": f"{missed} confirmed fraud case(s) passed every rule",
            "auto_apply": False,
        })
    return proposals


if __name__ == "__main__":
    labels = load_labels(LABELS_FILE)
    entries = load_decisions(LOG_FILE)
    stats = evaluate(entries, labels)

    print("Rule performance:")
    for rule in ALL_RULES:
        s = stats[rule]
        print(f"  {rule:20} flags={s['flags']} tp={s['tp']} fp={s['fp']}")
    print(f"  {'missed by all rules':20} fn={stats['_all_rules']['fn']}")

    proposals = propose(stats)
    with open(PROPOSALS_FILE, "w", encoding="utf-8") as f:
        json.dump(proposals, f, indent=2, ensure_ascii=False)
    print(f"\n{len(proposals)} proposal(s) written to {PROPOSALS_FILE}")