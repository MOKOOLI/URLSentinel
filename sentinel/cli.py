"""Command-line interface:  python -m sentinel <command>"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from . import __version__

if os.name == "nt":
    os.system("")  # enable ANSI colours on Windows terminals
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # box-drawing chars on Windows consoles
    except AttributeError:
        pass

C = {"SAFE": "\033[32m", "SUSPICIOUS": "\033[33m", "PHISHING": "\033[31m", "dim": "\033[2m", "b": "\033[1m", "0": "\033[0m"}
BANNER = rf"""{C['b']}
  _   _ ___ _    ___         _   _          _
 | | | | _ \ |  / __| ___ _ _| |_(_)_ _  ___| |
 | |_| |   / |__\__ \/ -_) ' \  _| | ' \/ -_) |
  \___/|_|_\____|___/\___|_||_\__|_|_||_\___|_|{C['0']}
  {C['dim']}phishing URL detection · NumPy neural network · v{__version__}{C['0']}
"""


def _print_scan(r):
    v = r["verdict"]
    bar = "█" * int(r["phishing_probability"] * 30)
    print(f"\n  {C['b']}{r['url']}{C['0']}")
    print(f"  {C[v]}{C['b']}{v:<11}{C['0']} {C[v]}{bar:<30}{C['0']} {r['phishing_probability'] * 100:5.1f}%")
    for x in r["reasons"]:
        mark = f"{C['PHISHING']}▲ risk{C['0']}" if x["direction"] == "risk" else f"{C['SAFE']}▼ safe{C['0']}"
        print(f"     {mark}  {x['reason']}")


def _plain_scan(r):
    bar = "█" * int(r["phishing_probability"] * 30)
    lines = [f"{r['url'][:80]}", f"  {r['verdict']:<11} {bar:<30} {r['phishing_probability'] * 100:5.1f}%"]
    lines += [f"     {'▲ risk' if x['direction'] == 'risk' else '▼ safe'}  {x['reason']}" for x in r["reasons"][:3]]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sentinel", description="URLSentinel: phishing URL detector")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="download the PhiUSIIL dataset (~55 MB) into data/")

    t = sub.add_parser("train", help="train on real data (PhiUSIIL by default)")
    t.add_argument("--csv", help="use your own real dataset instead (any CSV with url + label columns)")
    t.add_argument("--url-col", help="name of the URL column in --csv (auto-detected)")
    t.add_argument("--label-col", help="name of the label column in --csv (auto-detected)")
    t.add_argument("--phishing-label", default="1", help="label value that means phishing in --csv (default 1)")
    t.add_argument("--limit", type=int, help="random subsample for a faster run, e.g. 50000")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--hidden", default="64,32", help="hidden layer sizes, e.g. 128,64,32")
    t.add_argument("--quiet", action="store_true")

    lv = sub.add_parser("live", help="test the model on phishing URLs reported in the last hours (OpenPhish)")
    lv.add_argument("--feed", help="use a saved feed file instead of downloading")

    s = sub.add_parser("scan", help="scan one or more URLs")
    s.add_argument("urls", nargs="+")
    s.add_argument("--json", action="store_true")

    b = sub.add_parser("batch", help="scan a text file (one URL per line) and write a CSV")
    b.add_argument("file")
    b.add_argument("-o", "--out", default="reports/batch_results.csv")

    w = sub.add_parser("serve", help="start the web UI + JSON API")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "download":
        from .data import load_phiusiil
        urls, labels = load_phiusiil()
        print(f"  ready: {len(urls):,} URLs ({sum(labels):,} phishing)")
    elif a.cmd == "train":
        print(BANNER)
        from .train import run
        run(csv_path=a.csv, limit=a.limit, epochs=a.epochs, batch_size=a.batch_size,
            hidden=tuple(int(h) for h in a.hidden.split(",")), quiet=a.quiet,
            url_col=a.url_col, label_col=a.label_col, phishing_value=a.phishing_label)
    elif a.cmd == "live":
        print(BANNER)
        from .live import run as live_run
        try:
            live_run(feed_path=a.feed, refresh=a.feed is None)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif a.cmd == "scan":
        from .scanner import Scanner
        sc = _scanner_or_exit(Scanner)
        results = [sc.scan(u) for u in a.urls]
        if a.json:
            print(json.dumps(results, indent=2))
        else:
            print(BANNER)
            for r in results:
                _print_scan(r)
            print()
        return 1 if any(r["verdict"] == "PHISHING" for r in results) else 0
    elif a.cmd == "batch":
        from .scanner import Scanner
        sc = _scanner_or_exit(Scanner)
        with open(a.file, encoding="utf-8") as fh:
            urls = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        results = []
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["url", "verdict", "phishing_probability", "top_reason"])
            for u in urls:
                r = sc.scan(u)
                results.append(r)
                wr.writerow([u, r["verdict"], r["phishing_probability"], r["reasons"][0]["reason"] if r["reasons"] else ""])
                _print_scan(r)
        print(f"\n  wrote {len(urls)} results -> {a.out}")
        if os.path.normpath(a.file) == os.path.normpath("examples/urls.txt"):
            from .train import update_readme
            update_readme("DEMO", "```text\n$ python -m sentinel batch examples/urls.txt\n\n"
                          + "\n\n".join(_plain_scan(r) for r in results) + "\n```")
    elif a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port)
    return 0


def _scanner_or_exit(cls):
    try:
        return cls()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


def run():
    """Entry point with friendly errors instead of tracebacks."""
    try:
        return main()
    except RuntimeError as e:
        print(f"\nerror: {e}\n(check your internet connection, or pass a local file with --csv / --feed)", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(run())
