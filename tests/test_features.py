import unittest

from sentinel.features import FEATURE_NAMES, extract, shannon_entropy, vectorize


class FeatureTests(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(shannon_entropy(""), 0.0)
        self.assertEqual(shannon_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(shannon_entropy("ab"), 1.0)

    def test_ip_host(self):
        self.assertEqual(extract("http://10.0.0.1/login")["has_ip_host"], 1.0)
        self.assertEqual(extract("https://github.com")["has_ip_host"], 0.0)

    def test_at_symbol_and_brand(self):
        f = extract("http://paypal.com@evil.tk/login")
        self.assertEqual(f["has_at_symbol"], 1.0)
        self.assertEqual(f["brand_not_in_domain"], 1.0)
        self.assertEqual(f["suspicious_tld"], 1.0)

    def test_real_brand_is_not_flagged(self):
        f = extract("https://www.paypal.com/signin")
        self.assertEqual(f["brand_not_in_domain"], 0.0)
        self.assertEqual(f["typosquat_brand"], 0.0)

    def test_typosquat(self):
        self.assertEqual(extract("http://paypa1.xyz")["typosquat_brand"], 1.0)
        self.assertEqual(extract("http://netflik.com")["typosquat_brand"], 1.0)

    def test_free_hosting(self):
        self.assertEqual(extract("https://securce-chasce.vercel.app/index.html")["free_hosting"], 1.0)
        self.assertEqual(extract("https://vercel.com")["free_hosting"], 0.0)

    def test_roblox_typosquat_from_live_feed(self):
        self.assertEqual(extract("https://www.robiox.com.ps/users/1/profile")["typosquat_brand"], 1.0)

    def test_scheme_optional(self):
        self.assertEqual(extract("github.com")["host_length"], len("github.com"))

    def test_bad_port_does_not_crash(self):
        extract("http://example.com:99999/x")

    def test_vectorize_shape(self):
        self.assertEqual(vectorize(["a.com", "b.org"]).shape, (2, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
