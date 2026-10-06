<div align="center">

# 🛡️ URLSentinel

**Explainable phishing-URL detection with a neural network written from scratch in NumPy,
trained on 235k real URLs and tested against phishing reported live in the last few hours.**

[![CI](https://github.com/YOUR_USERNAME/urlsentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/urlsentinel/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)
![NumPy](https://img.shields.io/badge/neural%20net-pure%20NumPy-013243)
![Data](https://img.shields.io/badge/data-PhiUSIIL%20%2B%20OpenPhish%20live-b8432f)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

URLSentinel looks at a link **without ever opening it** and tells you how likely it is to be phishing,
and *why*. It mixes security know-how (typosquatting, IDN homographs, `@` tricks, free-hosting abuse)
with a hand-built neural network and Shapley-value explanations, and ships as a CLI, a web UI and a JSON API.

<!-- DEMO:START -->
_Run `./run.sh` once: this block is replaced with real output from your trained model._
<!-- DEMO:END -->

## ✨ Highlights

| Area | What is inside |
|---|---|
| **Deep learning from scratch** | Multi-layer perceptron in pure NumPy: He init, ReLU, Adam, inverted dropout, L2, mini-batches, early stopping. Backprop is checked against numerical gradients in the tests. |
| **Explainable AI** | Shapley values (permutation sampling) written by hand. Every verdict lists what pushed it towards *phishing*, and the contributions provably add up to the model's log-odds. |
| **Cyber-security** | 25 lexical features: Shannon entropy, typosquatting (homoglyphs + edit distance), punycode, brand-outside-domain, abused TLDs, shorteners, free site builders (vercel, weebly, webflow…), ports, redirects. |
| **Data science, done honestly** | Real data only. Automatic **shortcut audit**, **hostname-grouped split** (no site in both train and test), metrics from scratch, scikit-learn baselines, and an **out-of-time test** on live phishing. |
| **Engineering** | Zero-dependency web server, JSON API with input limits + security headers, CLI with pipeline-friendly exit codes, 24 offline tests, CI on Linux + Windows, weekly retrain job, Docker, one-command `run.sh`. |

## 📦 Data (100% real)

| Dataset | Role | Size |
|---|---|---|
| [PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (Prasad & Chandra, *Computers & Security* 2024, CC BY 4.0) | train / validation / test | 235,795 URLs (100,945 phishing) |
| [OpenPhish community feed](https://github.com/openphish/public_feed) | live, out-of-time test | latest few hundred phishing URLs, refreshed every 12 h |

Both are downloaded automatically on first run. Bring your own with
`python -m sentinel train --csv my_urls.csv --phishing-label malicious`.

## 🚀 Quick start (Git Bash, macOS, Linux)

```bash
git clone https://github.com/YOUR_USERNAME/urlsentinel.git
cd urlsentinel
./run.sh            # venv + deps + download data + train + live test + demo
./run.sh serve      # web UI  -> http://127.0.0.1:8000
./run.sh live       # re-test on today's phishing
./run.sh test       # test suite (offline)
```

## 📊 Results

<!-- RESULTS:START -->
_Filled in automatically by `python -m sentinel train`._
<!-- RESULTS:END -->

### 🔴 Live test on today's phishing

<!-- LIVE:START -->
_Filled in automatically by `python -m sentinel live`._
<!-- LIVE:END -->

<p>
  <img src="reports/training_curve.png" width="58%">
  <img src="reports/roc_curve.png" width="38%">
</p>

## 🔬 Methodology notes

* **Shortcut audit.** Before training, the pipeline compares simple properties (HTTPS, `www.`, has a path,
  has a query) between classes and flags big gaps. PhiUSIIL's legitimate URLs are mostly bare
  homepages, so a model can "cheat" by learning *"no path = safe"*. The audit makes that visible
  instead of hiding it behind a 99% score.
* **Grouped split.** Rows are split by hostname, so the test set only contains sites the model never saw.
* **Out-of-time test.** PhiUSIIL was collected in 2022-23. The OpenPhish feed is phishing from the last
  hours, so the live detection rate is the honest number for "would this catch a link sent to me today".

## 🧰 Commands

| Command | Description |
|---|---|
| `python -m sentinel train [--limit 50000] [--hidden 128,64,32]` | Train on PhiUSIIL + baselines, write `reports/` and update this README |
| `python -m sentinel train --csv FILE [--url-col C --label-col C --phishing-label V]` | Train on any other real dataset |
| `python -m sentinel live` | Score the live OpenPhish feed, report detection rate + hardest misses |
| `python -m sentinel scan URL [URL ...] [--json]` | Score URLs, exit code `1` if any is phishing |
| `python -m sentinel batch FILE` | Scan a file, write `reports/batch_results.csv` |
| `python -m sentinel serve [--port 8000]` | Web UI + JSON API |

```bash
curl -s -X POST localhost:8000/api/scan -H "Content-Type: application/json" \
     -d '{"url": "https://innstagrram.netlify.app/"}'
# batch: {"urls": ["...", "..."]}   health: GET /api/health
```

## 🏗️ Architecture

```text
URL ──► features.py ──► 25-d vector ──► nn.py (MLP 25→64→32→1) ──► probability
                                                 └─► explain(): Shapley values ─► "why" list

data.py   PhiUSIIL + OpenPhish downloaders, generic CSV loader
train.py  audit · grouped split · training · baselines · plots · README update
live.py   out-of-time evaluation on live phishing
scanner.py ──► cli.py (terminal) · server.py (web UI + API) · batch CSV
```

## ⚠️ Limitations

Lexical only: it never fetches the page, so it is fast and safe but blind to page content, certificates
and domain age. Free-hosting and brand signals can produce false positives on legitimate small sites.
A portfolio / research project, not a replacement for a secure email gateway.

## 🗺️ Roadmap

- [ ] Character-level CNN on the raw URL string (NumPy)
- [ ] Domain-age and TLS features (opt-in, network)
- [ ] Browser extension calling the local API

## 📄 License & credits

Code: MIT. PhiUSIIL: Prasad, A. & Chandra, S. (2024), CC BY 4.0, doi:10.1016/j.cose.2023.103545.
OpenPhish community feed: non-commercial use, see openphish.com/terms.html.
