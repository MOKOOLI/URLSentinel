"""Training pipeline: real data -> audit -> features -> NumPy MLP + sklearn baselines -> report."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from . import data, metrics
from .features import FEATURE_NAMES, vectorize
from .nn import NeuralNetwork

MODEL_PATH = Path("models/sentinel_mlp.npz")
REPORT_DIR = Path("reports")


def _host(url):
    try:
        return (urlparse(url if "://" in url else "http://" + url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return url


def group_split(urls, val=0.15, test=0.15, seed=0):
    """Split by hostname, so every page of a site lands in the same split.

    A random row split lets the model memorise sites it has already seen
    (e.g. 50 pages of the same phishing kit) and inflates the test score.
    """
    buckets = np.array([int(hashlib.md5(f"{seed}:{_host(u)}".encode()).hexdigest(), 16) % 10_000 / 10_000 for u in urls])
    te = np.where(buckets < test)[0]
    va = np.where((buckets >= test) & (buckets < test + val))[0]
    tr = np.where(buckets >= test + val)[0]
    return tr, va, te


def audit(urls, labels):
    """Look for shortcuts in the data before trusting any metric."""
    y = np.array(labels)
    def share(pred):
        hits = np.array([pred(u) for u in urls])
        return float(hits[y == 1].mean()), float(hits[y == 0].mean())
    checks = {
        "starts with https://": share(lambda u: u.lower().startswith("https://")),
        "has www.": share(lambda u: "://www." in u.lower()),
        "has a path beyond '/'": share(lambda u: len(urlparse(u).path.strip("/")) > 0 if "://" in u else "/" in u),
        "has a query string": share(lambda u: "?" in u),
    }
    rows = []
    for name, (p, q) in checks.items():
        flag = abs(p - q) > 0.4
        rows.append({"check": name, "phishing": p, "legitimate": q, "warning": flag})
        print(f"      {name:<24} phishing {p * 100:5.1f}%   legitimate {q * 100:5.1f}%" + ("   <- possible shortcut" if flag else ""))
    return rows


def _baselines(Xtr, ytr, Xte, yte):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("      (scikit-learn not installed, skipping baselines)")
        return {}
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1),
    }
    out = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        out[name] = metrics.report(yte, m.predict_proba(Xte)[:, 1])
        out[name]["train_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"      {name}: done in {out[name]['train_seconds']}s")
    return out


def _plots(net, y_test, p_test, auc, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      (matplotlib not installed, skipping plots)")
        return []
    files = []
    plt.figure(figsize=(7, 4))
    plt.plot(net.history.train_loss, label="train")
    plt.plot(net.history.val_loss, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.title("Training curve (PhiUSIIL)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    files.append(out_dir / "training_curve.png")
    plt.savefig(files[-1], dpi=140)
    plt.close()

    fpr, tpr = metrics.roc_curve(y_test, p_test)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"NumPy MLP (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC, held-out hosts")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    files.append(out_dir / "roc_curve.png")
    plt.savefig(files[-1], dpi=140)
    plt.close()
    return files


def run(csv_path=None, limit=None, epochs=100, hidden=(64, 32), batch_size=256, seed=42, quiet=False,
        url_col=None, label_col=None, phishing_value="1"):
    print("[1/6] loading real data")
    if csv_path:
        urls, labels = data.load_csv(csv_path, url_col, label_col, phishing_value, limit, seed)
        source = Path(csv_path).name
    else:
        urls, labels = data.load_phiusiil(limit=limit, seed=seed)
        source = "PhiUSIIL (UCI #967)"
    y = np.array(labels)
    print(f"      {len(urls):,} unique URLs from {source}: {int(y.sum()):,} phishing / {int((1 - y).sum()):,} legitimate")

    print("[2/6] auditing the data for shortcuts")
    audit_rows = audit(urls, labels)

    print(f"[3/6] extracting {len(FEATURE_NAMES)} features")
    t0 = time.perf_counter()
    X = vectorize(urls)
    print(f"      {time.perf_counter() - t0:.1f}s")
    tr, va, te = group_split(urls, seed=seed)
    print(f"      split by hostname: train {len(tr):,} / val {len(va):,} / test {len(te):,}")

    print("[4/6] training NumPy neural network", [len(FEATURE_NAMES), *hidden, 1])
    t0 = time.perf_counter()
    net = NeuralNetwork([X.shape[1], *hidden, 1], lr=2e-3, l2=1e-4, dropout=0.1, seed=seed)
    net.fit(X[tr], y[tr], X[va], y[va], epochs=epochs, batch_size=batch_size, patience=10, verbose=not quiet)
    p_test = net.predict_proba(X[te])
    results = metrics.report(y[te], p_test)
    results["train_seconds"] = round(time.perf_counter() - t0, 2)
    net.save(MODEL_PATH)

    print("[5/6] training baselines")
    all_results = {"NumPyMLP": results, **_baselines(X[tr], y[tr], X[te], y[te])}

    print("[6/6] writing report")
    REPORT_DIR.mkdir(exist_ok=True)
    plots = _plots(net, y[te], p_test, results["roc_auc"], REPORT_DIR)
    summary = {"source": source, "n_urls": len(urls), "n_phishing": int(y.sum()), "n_test": len(te),
               "audit": audit_rows, "models": all_results}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    md = _markdown(summary, plots)
    (REPORT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    update_readme("RESULTS", _results_block(summary))

    print("\n  model               accuracy  precision  recall    f1      auc")
    for name, r in all_results.items():
        print(f"  {name:<19} {r['accuracy']:.4f}    {r['precision']:.4f}     {r['recall']:.4f}   {r['f1']:.4f}  {r['roc_auc']:.4f}")
    print(f"\n  saved model  -> {MODEL_PATH}\n  saved report -> {REPORT_DIR / 'REPORT.md'}")
    return summary


def _table(models):
    lines = ["| model | accuracy | precision | recall | F1 | ROC-AUC | train (s) |", "|---|---|---|---|---|---|---|"]
    for name, r in models.items():
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1']:.4f} | {r['roc_auc']:.4f} | {r['train_seconds']} |")
    return lines


def _results_block(s):
    lines = [f"Trained on **{s['source']}**: {s['n_urls']:,} real URLs ({s['n_phishing']:,} phishing). "
             f"Test set = {s['n_test']:,} URLs from **hostnames never seen in training**.", ""]
    lines += _table(s["models"])
    warn = [a["check"] for a in s["audit"] if a["warning"]]
    if warn:
        lines += ["", f"Data audit flagged possible shortcuts ({', '.join(warn)}), which is why the "
                  "[live test](#-live-test-on-todays-phishing) below matters more than these numbers."]
    return "\n".join(lines)


def _markdown(s, plots):
    lines = ["# URLSentinel training report", "", _results_block(s), "", "## Data audit", "",
             "| check | phishing | legitimate |", "|---|---|---|"]
    for a in s["audit"]:
        lines.append(f"| {a['check']} | {a['phishing'] * 100:.1f}% | {a['legitimate'] * 100:.1f}% |" + (" ⚠️" if a["warning"] else ""))
    c = s["models"]["NumPyMLP"]["confusion"]
    lines += ["", "## NumPy MLP confusion matrix", "", "| | predicted legit | predicted phishing |", "|---|---|---|",
              f"| **actual legit** | {c['tn']:,} | {c['fp']:,} |", f"| **actual phishing** | {c['fn']:,} | {c['tp']:,} |", ""]
    lines += [f"![{p.stem}]({p.name})" for p in plots]
    return "\n".join(lines) + "\n"


def update_readme(tag, content, path=Path("README.md")):
    """Replace the text between <!-- TAG:START --> and <!-- TAG:END --> in the README."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        path.write_text(f"{head}{start}\n{content}\n{end}{tail}", encoding="utf-8")
