"""Real-world data sources. Nothing in this project is synthetic.

* PhiUSIIL (Prasad & Chandra, Computers & Security 2024, CC BY 4.0):
  235,795 real URLs, 100,945 phishing + 134,850 legitimate, collected 2022-2023
  from PhishTank, OpenPhish, MalwareWorld and Open PageRank.
  NOTE: in the original file label 1 = legitimate, 0 = phishing. We flip it so
  that throughout this project 1 = phishing.
* OpenPhish community feed: phishing URLs seen live in the last hours, used as an
  out-of-time test set (the model has never seen them). Non-commercial use only.
"""
from __future__ import annotations

import csv
import random
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")

PHIUSIIL_SOURCES = [
    "https://archive.ics.uci.edu/static/public/967/data.csv",  # UCI API "data_url"
]
OPENPHISH_SOURCES = [
    "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt",
    "https://openphish.com/feed.txt",
]
USER_AGENT = "URLSentinel/1.0 (+https://github.com; research project)"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # PhiUSIIL has very long Title fields


def download(sources, dest: Path, min_bytes: int = 1000) -> Path:
    """Stream the first working source to `dest` with a progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in sources:
        tmp = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, tempfile.NamedTemporaryFile(
                    delete=False, dir=dest.parent, suffix=".part") as tmp:  # noqa: SIM117
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while chunk := resp.read(1 << 16):
                    tmp.write(chunk)
                    done += len(chunk)
                    _progress(done, total, url)
            if done < min_bytes:
                raise OSError(f"only {done} bytes received")
            shutil.move(tmp.name, dest)
            print()
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
            print(f"\n  ! {url} failed ({exc})")
    raise RuntimeError("all download sources failed:\n  " + "\n  ".join(errors))


def _progress(done, total, url):
    mb = done / 1e6
    if total:
        pct = done / total
        bar = "#" * int(pct * 30)
        print(f"\r  downloading {bar:<30} {pct * 100:5.1f}%  {mb:6.1f} MB", end="", flush=True)
    else:
        print(f"\r  downloading {url.split('/')[2]}  {mb:6.1f} MB", end="", flush=True)


def _dedupe(urls, labels):
    seen = {}
    for u, y in zip(urls, labels):
        seen.setdefault(u, y)
    return list(seen.keys()), list(seen.values())


def _subsample(urls, labels, limit, seed):
    if not limit or limit >= len(urls):
        return urls, labels
    idx = random.Random(seed).sample(range(len(urls)), limit)
    return [urls[i] for i in idx], [labels[i] for i in idx]


def load_phiusiil(path: Path | None = None, limit: int | None = None, seed: int = 42):
    """Return (urls, labels) with 1 = phishing. Downloads ~55 MB on first use."""
    path = Path(path) if path else DATA_DIR / "phiusiil.csv"
    if not path.exists():
        print("  PhiUSIIL dataset not found locally, downloading from UCI (one time, ~55 MB)")
        download(PHIUSIIL_SOURCES, path, min_bytes=10_000_000)
    urls, labels = [], []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("URL") or "").strip()
            if url:
                urls.append(url)
                labels.append(1 - int(row["label"]))  # original: 1 = legitimate
    urls, labels = _dedupe(urls, labels)
    return _subsample(urls, labels, limit, seed)


def load_csv(path, url_col=None, label_col=None, phishing_value="1", limit=None, seed=42):
    """Generic loader for any real dataset (Kaggle, PhishTank exports, your own logs)."""
    with Path(path).open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        url_col = url_col or next((cols[c] for c in ("url", "urls", "link") if c in cols), None)
        label_col = label_col or next((cols[c] for c in ("label", "class", "type", "target", "status") if c in cols), None)
        if not url_col or not label_col:
            raise ValueError(f"could not find url/label columns in {reader.fieldnames}; pass --url-col/--label-col")
        urls, labels = [], []
        for row in reader:
            url = (row[url_col] or "").strip()
            if url:
                urls.append(url)
                labels.append(int(str(row[label_col]).strip().lower() == str(phishing_value).lower()))
    urls, labels = _dedupe(urls, labels)
    return _subsample(urls, labels, limit, seed)


def load_openphish(path: Path | None = None, refresh: bool = True):
    """Today's live phishing URLs (all label 1)."""
    path = Path(path) if path else DATA_DIR / "openphish_feed.txt"
    if refresh or not path.exists():
        print("  fetching the live OpenPhish community feed")
        download(OPENPHISH_SOURCES, path, min_bytes=200)
    with path.open(encoding="utf-8", errors="replace") as fh:
        urls = [line.strip() for line in fh if line.strip().startswith(("http://", "https://"))]
    return list(dict.fromkeys(urls))
