"""Scoring + human-readable explanations for a single URL."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, HUMAN_LABELS, extract
from .nn import NeuralNetwork

DEFAULT_MODEL = Path("models/sentinel_mlp.npz")


class Scanner:
    def __init__(self, model_path=DEFAULT_MODEL):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No model at {model_path}. Run: python -m sentinel train")
        self.net = NeuralNetwork.load(model_path)

    def scan(self, url: str, top_k: int = 4) -> dict:
        feats = extract(url)
        x = np.array([feats[n] for n in FEATURE_NAMES])
        prob = float(self.net.predict_proba(x)[0])
        contrib = self.net.explain(x)
        base = self.net.baseline
        reasons = []
        for i in np.argsort(-np.abs(contrib)):
            name, c = FEATURE_NAMES[i], float(contrib[i])
            if name not in HUMAN_LABELS or abs(c) < (0.2 if prob >= 0.5 else 1.0):
                continue
            if c > 0 and x[i] > base[i] and name != "has_https":
                reasons.append({"feature": name, "reason": HUMAN_LABELS[name], "direction": "risk", "impact": round(c, 4)})
            elif c < 0 and name == "has_https" and x[i] == 1:
                reasons.append({"feature": name, "reason": HUMAN_LABELS[name], "direction": "safe", "impact": round(c, 4)})
            if len(reasons) == top_k:
                break
        if not any(r["direction"] == "risk" for r in reasons):
            reasons.append({"feature": None, "reason": "no red flags in the URL structure", "direction": "safe", "impact": 0.0})
        return {"url": url, "phishing_probability": round(prob, 4), "verdict": verdict(prob),
                "reasons": reasons, "features": feats}


def verdict(p: float) -> str:
    if p >= 0.85:
        return "PHISHING"
    if p >= 0.5:
        return "SUSPICIOUS"
    return "SAFE"
