import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from sentinel import train
from sentinel.scanner import Scanner


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = os.getcwd()
        fixture = Path(__file__).parent / "fixtures" / "real_urls.csv"
        os.chdir(cls.tmp.name)
        train.run(csv_path=fixture, epochs=200, hidden=(16, 8), batch_size=16, quiet=True)
        cls.scanner = Scanner(Path("models/sentinel_mlp.npz"))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def test_report_written(self):
        self.assertTrue(Path("reports/REPORT.md").exists())
        summary = json.loads(Path("reports/metrics.json").read_text())
        self.assertEqual(summary["n_urls"], 80)
        self.assertIn("NumPyMLP", summary["models"])

    def test_scan_shape(self):
        r = self.scanner.scan("https://www.robiox.com.ps/users/884221804/profile")
        self.assertIn(r["verdict"], {"SAFE", "SUSPICIOUS", "PHISHING"})
        self.assertTrue(0 <= r["phishing_probability"] <= 1)
        self.assertTrue(r["reasons"])

    def test_live_eval_offline_feed(self):
        from sentinel import live
        feed = Path("feed.txt")
        feed.write_text("https://innstagrram.netlify.app/\nhttps://securce-chasce.vercel.app/index.html\n# comment\n")
        out = live.run(feed_path=feed, refresh=False, show_missed=0)
        self.assertEqual(out["n"], 2)
        self.assertTrue(Path("reports/live_eval.csv").exists())

    def test_http_api(self):
        from sentinel.server import make_handler
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.scanner))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scan",
                                         data=json.dumps({"url": "http://10.1.2.3/apple/login"}).encode(),
                                         headers={"Content-Type": "application/json"})
            body = json.loads(urllib.request.urlopen(req).read())
            self.assertIn("phishing_probability", body)
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            self.assertIn("URLSentinel", page)
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
