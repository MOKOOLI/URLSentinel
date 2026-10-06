import tempfile
import unittest
from pathlib import Path

import numpy as np

from sentinel import metrics
from sentinel.nn import NeuralNetwork


def xor_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


class NetworkTests(unittest.TestCase):
    def test_gradients_match_numerical(self):
        rng = np.random.default_rng(1)
        net = NeuralNetwork([3, 5, 1], dropout=0.0, l2=0.0)
        X, y = rng.normal(size=(10, 3)), rng.integers(0, 2, (10, 1)).astype(float)
        acts, masks = net._forward(X)
        gW, _ = net._backward(acts, masks, y)
        eps = 1e-6
        for (i, j) in [(0, 0), (1, 3), (2, 4)]:
            net.W[0][i, j] += eps
            lp = net._bce(net._forward(X)[0][-1], y)
            net.W[0][i, j] -= 2 * eps
            lm = net._bce(net._forward(X)[0][-1], y)
            net.W[0][i, j] += eps
            self.assertAlmostEqual(gW[0][i, j], (lp - lm) / (2 * eps), places=5)

    def test_learns_xor(self):
        X, y = xor_data()
        net = NeuralNetwork([2, 16, 8, 1], lr=1e-2, dropout=0.0, seed=3)
        net.fit(X[:600], y[:600], X[600:], y[600:], epochs=300, verbose=False)
        self.assertGreater((net.predict(X[600:]) == y[600:]).mean(), 0.9)

    def test_save_load_roundtrip(self):
        X, y = xor_data(200)
        net = NeuralNetwork([2, 8, 1]).fit(X, y, epochs=5, verbose=False)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.npz"
            net.save(path)
            again = NeuralNetwork.load(path)
        np.testing.assert_allclose(net.predict_proba(X), again.predict_proba(X))

    def test_shapley_sums_to_logit_gap(self):
        X, y = xor_data(300)
        net = NeuralNetwork([2, 8, 1], dropout=0.0).fit(X, y, epochs=20, verbose=False)
        x = X[0]

        def logit(v):
            p = net.predict_proba(v)[0]
            return np.log(p / (1 - p))

        self.assertAlmostEqual(net.explain(x).sum(), logit(x) - logit(net.baseline), places=6)


class MetricTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(metrics.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(metrics.roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_report(self):
        r = metrics.report([1, 0, 1, 0], [0.9, 0.1, 0.4, 0.6])
        self.assertEqual(r["confusion"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertAlmostEqual(r["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
