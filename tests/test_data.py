import unittest
from pathlib import Path

from sentinel import data
from sentinel.train import _host, group_split

FIX = Path(__file__).parent / "fixtures"


class DataTests(unittest.TestCase):
    def test_phiusiil_labels_are_flipped_and_deduped(self):
        urls, labels = data.load_phiusiil(FIX / "phiusiil_format.csv")
        self.assertEqual(urls, ["https://www.python.org", "https://innstagrram.netlify.app/"])
        self.assertEqual(labels, [0, 1])  # original file: 1 = legitimate

    def test_generic_csv_autodetects_columns(self):
        urls, labels = data.load_csv(FIX / "real_urls.csv")
        self.assertEqual(len(urls), 80)
        self.assertEqual(sum(labels), 40)

    def test_limit(self):
        urls, _ = data.load_csv(FIX / "real_urls.csv", limit=10)
        self.assertEqual(len(urls), 10)

    def test_group_split_has_no_host_overlap(self):
        urls = [f"https://site{i % 50}.com/page/{i}" for i in range(2000)]
        tr, va, te = group_split(urls)
        hosts = lambda idx: {_host(urls[i]) for i in idx}  # noqa: E731
        self.assertFalse(hosts(tr) & hosts(te))
        self.assertFalse(hosts(tr) & hosts(va))
        self.assertEqual(len(tr) + len(va) + len(te), len(urls))


if __name__ == "__main__":
    unittest.main()
