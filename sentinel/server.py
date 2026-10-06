"""Zero-dependency web UI + JSON API (Python standard library only)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlparse

from .scanner import Scanner

PAGE = files("sentinel").joinpath("static/index.html").read_text(encoding="utf-8")


def make_handler(scanner: Scanner):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                return self._send(200, PAGE, "text/html")
            if route.path == "/api/health":
                return self._send(200, json.dumps({"status": "ok"}))
            if route.path == "/api/scan":
                url = parse_qs(route.query).get("url", [""])[0]
                return self._scan(url)
            self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if urlparse(self.path).path != "/api/scan":
                return self._send(404, json.dumps({"error": "not found"}))
            length = min(int(self.headers.get("Content-Length", 0)), 1_000_000)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, json.dumps({"error": "invalid JSON"}))
            if isinstance(payload.get("urls"), list):
                urls = [str(u) for u in payload["urls"][:500]]
                return self._send(200, json.dumps([scanner.scan(u) for u in urls]))
            return self._scan(str(payload.get("url", "")))

        def _scan(self, url):
            if not url or len(url) > 4096:
                return self._send(400, json.dumps({"error": "provide a url (max 4096 chars)"}))
            self._send(200, json.dumps(scanner.scan(url)))

        def log_message(self, fmt, *args):
            print(f"  {self.address_string()}  {fmt % args}")

    return Handler


def serve(host="127.0.0.1", port=8000, model=None):
    scanner = Scanner(model) if model else Scanner()
    httpd = ThreadingHTTPServer((host, port), make_handler(scanner))
    print(f"\n  URLSentinel running at http://{host}:{port}   (Ctrl+C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")
    finally:
        httpd.server_close()
