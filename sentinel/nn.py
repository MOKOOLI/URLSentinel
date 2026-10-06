"""A small but complete neural network written from scratch with NumPy.

Features: arbitrary hidden layers, ReLU, sigmoid output, binary cross-entropy,
He initialisation, Adam optimiser, inverted dropout, L2 regularisation,
mini-batches, early stopping and built-in feature standardisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class NeuralNetwork:
    def __init__(self, layers, lr=1e-3, l2=1e-4, dropout=0.1, seed=42):
        self.layers = list(layers)
        self.lr, self.l2, self.dropout = lr, l2, dropout
        self.rng = np.random.default_rng(seed)
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(layers[:-1], layers[1:])]
        self.b = [np.zeros((1, b)) for b in layers[1:]]
        self.mu = np.zeros((1, layers[0]))
        self.sigma = np.ones((1, layers[0]))
        self.baseline = np.zeros(layers[0])  # "typical legitimate URL", used for explanations
        self.history = History()

    # ----- forward / backward -------------------------------------------------
    def _forward(self, X, train=False):
        acts, masks = [X], []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i == len(self.W) - 1:
                a = _sigmoid(z)
            else:
                a = np.maximum(z, 0)
                if train and self.dropout > 0:
                    m = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * m
                    masks.append(m)
                else:
                    masks.append(None)
            acts.append(a)
        return acts, masks

    def _backward(self, acts, masks, y):
        n = y.shape[0]
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        delta = (acts[-1] - y) / n  # dL/dz for sigmoid + BCE
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.W[i].T
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)
        return grads_W, grads_b

    @staticmethod
    def _bce(p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ----- public API ---------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, epochs=200, batch_size=64, patience=20, verbose=True):
        self.mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        self.sigma = np.where(sd < 1e-6, 1.0, sd)  # constant features stay unscaled
        self.baseline = np.median(X[np.asarray(y).ravel() == 0], axis=0) if (np.asarray(y) == 0).any() else X.mean(axis=0)
        Xs = (X - self.mu) / self.sigma
        y = y.reshape(-1, 1).astype(float)
        has_val = X_val is not None
        if has_val:
            Xv = (X_val - self.mu) / self.sigma
            yv = y_val.reshape(-1, 1).astype(float)

        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best, best_state, wait = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            idx = self.rng.permutation(len(Xs))
            for start in range(0, len(Xs), batch_size):
                bi = idx[start:start + batch_size]
                acts, masks = self._forward(Xs[bi], train=True)
                gW, gb = self._backward(acts, masks, y[bi])
                t += 1
                for j, (p, g) in enumerate(zip(self.W + self.b, gW + gb)):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mh, vh = m[j] / (1 - b1 ** t), v[j] / (1 - b2 ** t)
                    p -= self.lr * mh / (np.sqrt(vh) + eps)

            tr_loss = self._bce(self._forward(Xs)[0][-1], y)
            self.history.train_loss.append(tr_loss)
            if has_val:
                pv = self._forward(Xv)[0][-1]
                vl = self._bce(pv, yv)
                self.history.val_loss.append(vl)
                self.history.val_acc.append(float(((pv > 0.5) == yv).mean()))
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    wait += 1
                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"  epoch {epoch:4d}  loss {tr_loss:.4f}  val_loss {vl:.4f}  val_acc {self.history.val_acc[-1]:.4f}")
                if wait >= patience:
                    if verbose:
                        print(f"  early stopping at epoch {epoch} (best val_loss {best:.4f})")
                    break
        if best_state:
            self.W, self.b = best_state
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return self._forward((X - self.mu) / self.sigma)[0][-1].ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def explain(self, x, samples=64, seed=0):
        """Shapley-value attribution (permutation sampling), in log-odds.

        Each feature's value is its average marginal effect when it is switched from
        a "typical legitimate URL" baseline to the real value, over random feature orders.
        Positive = pushes towards phishing. The values sum to logit(x) - logit(baseline).
        """
        x = np.asarray(x, dtype=float).ravel()
        d = len(x)
        rng = np.random.default_rng(seed)
        perms = np.array([rng.permutation(d) for _ in range(samples)])
        # Build every intermediate input: for each permutation, switch features on one by one.
        steps = np.tile(self.baseline, (samples, d + 1, 1))
        for k in range(samples):
            for j in range(d):
                steps[k, j + 1:, perms[k, j]] = x[perms[k, j]]
        p = np.clip(self.predict_proba(steps.reshape(-1, d)), 1e-9, 1 - 1e-9).reshape(samples, d + 1)
        logits = np.log(p / (1 - p))
        gains = np.diff(logits, axis=1)
        phi = np.zeros(d)
        np.add.at(phi, perms.ravel(), gains.ravel())
        return phi / samples

    # ----- persistence --------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}
        meta = json.dumps({"layers": self.layers, "lr": self.lr, "l2": self.l2, "dropout": self.dropout})
        np.savez(path, mu=self.mu, sigma=self.sigma, baseline=self.baseline, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        net = cls(meta["layers"], meta["lr"], meta["l2"], meta["dropout"])
        net.W = [data[f"W{i}"] for i in range(len(net.W))]
        net.b = [data[f"b{i}"] for i in range(len(net.b))]
        net.mu, net.sigma = data["mu"], data["sigma"]
        if "baseline" in data:
            net.baseline = data["baseline"]
        return net
