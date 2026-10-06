"""Lexical feature extraction for URLs. No network access needed, so scanning is safe and instant."""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "zip", "mov", "click", "country",
    "work", "support", "rest", "fit", "loan", "cam", "icu", "buzz", "live",
}
SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "account", "update", "secure", "banking", "confirm",
    "password", "wallet", "suspend", "unlock", "billing", "webscr", "recover", "auth",
)
BRANDS = (
    "paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook", "instagram",
    "whatsapp", "telegram", "roblox", "steam", "metamask", "binance", "coinbase", "outlook",
    "office365", "yahoo", "adobe", "dropbox", "docusign", "linkedin", "wellsfargo", "chase",
    "bank", "dhl", "fedex", "usps", "ebay",
)
# Free website builders / static hosts: real sites use them too, but most of today's
# phishing kits live there (see the OpenPhish feed), so it is a strong signal in context.
FREE_HOSTS = (
    "vercel.app", "netlify.app", "weebly.com", "blogspot.com", "github.io", "webflow.io",
    "pages.dev", "workers.dev", "replit.app", "glitch.me", "firebaseapp.com", "web.app",
    "wixsite.com", "gitbook.io", "r2.dev", "ngrok.io", "ngrok-free.app", "duckdns.org",
    "000webhostapp.com", "square.site", "godaddysites.com", "framer.website", "edgeone.dev",
    "cpanel.site", "b-cdn.net", "backblazeb2.com", "up.railway.app", "onrender.com", "nip.io",
)
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy",
              "u.to", "1url.at", "alturl.com", "lnk.ink", "shorturl.at", "tiny.cc"}
# Common character swaps used by typosquatters.
HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "7": "t", "@": "a"})

IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens", "num_digits",
    "num_subdomains", "num_special", "num_query_params", "host_entropy", "path_entropy",
    "digit_ratio", "has_ip_host", "has_at_symbol", "has_https", "has_port",
    "suspicious_tld", "suspicious_words", "brand_not_in_domain", "typosquat_brand",
    "is_shortener", "has_punycode", "double_slash_in_path", "long_host_token", "free_hosting",
]

HUMAN_LABELS = {
    "url_length": "URL is unusually long",
    "host_length": "hostname is unusually long",
    "num_hyphens": "many hyphens in the URL",
    "num_digits": "many digits in the URL",
    "num_subdomains": "deeply nested subdomains",
    "host_entropy": "hostname looks random (high entropy)",
    "has_ip_host": "raw IP address instead of a domain",
    "has_at_symbol": "'@' symbol can hide the real destination",
    "has_https": "uses HTTPS",
    "has_port": "non-standard port",
    "suspicious_tld": "TLD frequently abused for phishing",
    "suspicious_words": "credential-bait words (login, verify, ...)",
    "brand_not_in_domain": "brand name appears outside the real domain",
    "typosquat_brand": "looks like a misspelled brand (typosquatting)",
    "is_shortener": "URL shortener hides the destination",
    "has_punycode": "punycode / IDN homograph domain",
    "double_slash_in_path": "redirect-style '//' in path",
    "long_host_token": "very long token in hostname",
    "free_hosting": "hosted on a free site builder / static host",
}


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalize(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def registered_domain(host: str) -> str:
    """Rough eTLD+1 (good enough without the public-suffix list)."""
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "ac", "org", "net", "gov"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract(url: str) -> dict[str, float]:
    parsed = urlparse(_normalize(url))
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    labels = host.split(".") if host else []
    tld = labels[-1] if labels else ""
    reg = registered_domain(host)
    reg_name = reg.split(".")[0] if reg else ""
    outside_domain = full.replace(reg, "", 1) if reg else full

    is_ip = bool(IPV4_RE.match(host))
    brand_outside = any(b in outside_domain and b not in reg_name for b in BRANDS)
    squat_name = reg_name.translate(HOMOGLYPHS).replace("-", "")
    typosquat = any(
        (b in squat_name and b not in reg_name) or (b != reg_name and _edit1(reg_name, b))
        for b in BRANDS
    )

    f = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(ch.isdigit() for ch in url),
        "num_subdomains": max(len(labels) - 2, 0) if not is_ip else 0,
        "num_special": sum(url.count(c) for c in "~%&=_!*,;$"),
        "num_query_params": len([q for q in query.split("&") if q]),
        "host_entropy": shannon_entropy(host),
        "path_entropy": shannon_entropy(path),
        "digit_ratio": sum(ch.isdigit() for ch in host) / max(len(host), 1),
        "has_ip_host": float(is_ip),
        "has_at_symbol": float("@" in url),
        "has_https": float(parsed.scheme == "https"),
        "has_port": float(parsed.port not in (None, 80, 443)) if _safe_port(parsed) else 1.0,
        "suspicious_tld": float(tld in SUSPICIOUS_TLDS),
        "suspicious_words": sum(w in full for w in SUSPICIOUS_WORDS),
        "brand_not_in_domain": float(brand_outside),
        "typosquat_brand": float(typosquat),
        "is_shortener": float(reg in SHORTENERS or host in SHORTENERS),
        "has_punycode": float("xn--" in host),
        "double_slash_in_path": float("//" in path),
        "long_host_token": float(any(len(t) > 20 for t in re.split(r"[.-]", host))),
        "free_hosting": float(any(host == h or host.endswith("." + h) for h in FREE_HOSTS)),
    }
    return {k: float(v) for k, v in f.items()}


def _safe_port(parsed) -> bool:
    try:
        parsed.port
        return True
    except ValueError:
        return False


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by exactly one edit (and are not identical)."""
    if a == b or abs(len(a) - len(b)) > 1 or len(b) < 6:  # short brands collide with real words
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = j = diff = 0
    while i < len(short) and j < len(long_):
        if short[i] != long_[j]:
            diff += 1
            j += 1
            if diff > 1:
                return False
        else:
            i += 1
            j += 1
    return True


def vectorize(urls: list[str]) -> np.ndarray:
    rows = []
    for u in urls:
        f = extract(u)
        rows.append([f[n] for n in FEATURE_NAMES])
    return np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES))
