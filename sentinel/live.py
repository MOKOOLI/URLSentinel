"""Out-of-time evaluation on phishing URLs reported in the last hours (OpenPhish)."""
from __future__ import annotations

import csv
import datetime as dt
from collections import Counter
from pathlib import Path

from . import data
from .scanner import Scanner
from .train import REPORT_DIR, update_readme


def run(feed_path=None, refresh=True, show_missed=10):
    urls = data.load_openphish(feed_path, refresh=refresh)
    if not urls:
        raise RuntimeError("the feed is empty")
    sc = Scanner()
    results = [sc.scan(u, top_k=2) for u in urls]
    verdicts = Counter(r["verdict"] for r in results)
    caught = verdicts["PHISHING"] + verdicts["SUSPICIOUS"]
    rate = caught / len(results)
    reasons = Counter(x["reason"] for r in results if r["verdict"] != "SAFE" for x in r["reasons"][:1])
    missed = sorted((r for r in results if r["verdict"] == "SAFE"), key=lambda r: r["phishing_probability"])

    REPORT_DIR.mkdir(exist_ok=True)
    with (REPORT_DIR / "live_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "verdict", "phishing_probability"])
        w.writerows([r["url"], r["verdict"], r["phishing_probability"]] for r in results)

    today = dt.date.today().isoformat()
    block = [f"On **{today}** the model scanned **{len(results):,} live phishing URLs** from the "
             "OpenPhish feed, reported after the training data was collected:", "",
             "| caught as PHISHING | flagged SUSPICIOUS | missed (SAFE) | detection rate |", "|---|---|---|---|",
             f"| {verdicts['PHISHING']:,} | {verdicts['SUSPICIOUS']:,} | {verdicts['SAFE']:,} | **{rate * 100:.1f}%** |", "",
             "Most common reasons for the catch: " + ", ".join(f"{k} ({v})" for k, v in reasons.most_common(4)) + "."]
    update_readme("LIVE", "\n".join(block))
    (REPORT_DIR / "LIVE.md").write_text("# Live evaluation\n\n" + "\n".join(block) + "\n", encoding="utf-8")

    print(f"\n  scanned {len(results):,} live phishing URLs")
    print(f"  PHISHING {verdicts['PHISHING']:,}   SUSPICIOUS {verdicts['SUSPICIOUS']:,}   SAFE (missed) {verdicts['SAFE']:,}")
    print(f"  detection rate: {rate * 100:.1f}%")
    if missed and show_missed:
        print("\n  hardest misses (what to improve next):")
        for r in missed[:show_missed]:
            print(f"    {r['phishing_probability'] * 100:5.1f}%  {r['url'][:100]}")
    print(f"\n  wrote {Path(REPORT_DIR) / 'live_eval.csv'}")
    return {"n": len(results), "detection_rate": rate, "verdicts": dict(verdicts)}
