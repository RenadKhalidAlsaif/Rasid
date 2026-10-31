import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
OUT = Path("decisions_demo.jsonl")
LOAN_TYPES = ["Personal", "Auto", "Home", "Business"]

end = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
start = end - timedelta(days=335)  # about 11 months

rows = []
for m in range(11):
    month_start = start + timedelta(days=30 * m)
    phase = "before_ai" if m < 5 else "ai"
    clear_pct = 60.0 if phase == "before_ai" else min(60 + (m - 4) * 2.0, 72.0)
    for _ in range(random.randint(60, 90)):
        ts = month_start + timedelta(days=random.randint(0, 29), minutes=random.randint(0, 1400))
        p = clear_pct / 100
        r = random.random()
        if r < p:
            decision = "clear"
        elif r < p + (1 - p) * 0.7:
            decision = "reject"
        else:
            decision = "uncertain"
        rows.append({
            "timestamp": ts.isoformat(timespec="seconds"),
            "form": "demo",
            "loan_type": random.choice(LOAN_TYPES),
            "decision": decision,
            "reason": "synthetic demo record",
            "ocr_engine": "synthetic",
            "phase": phase,
            "synthetic": True,
            "extracted": {},
        })

rows.sort(key=lambda r: r["timestamp"])
with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} synthetic rows to {OUT}")