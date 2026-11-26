import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / ".pylibs").is_dir():
    sys.path.insert(0, str(_ROOT / ".pylibs"))

import numpy as np

from hnm.transfer import extract_linear_height, srgb_decode, transfer_apply


class TestTransfer(unittest.TestCase):
    def test_srgb_eotf_known_values(self):
        self.assertAlmostEqual(float(srgb_decode(np.array([0.0]))[0]), 0.0)
        self.assertAlmostEqual(float(srgb_decode(np.array([1.0]))[0]), 1.0)
        v50 = float(srgb_decode(np.array([0.5]))[0])
        self.assertAlmostEqual(v50, 0.214041, places=5)

    def test_linear_passthrough(self):
        x = np.linspace(0, 1, 11)
        np.testing.assert_array_equal(transfer_apply(x, "linear"), x)

    def test_gamma(self):
        x = np.array([0.25, 1.0])
        np.testing.assert_allclose(transfer_apply(x, "gamma", 2.0), [0.0625, 1.0])

    def test_extract_luma_linearizes_first(self):
        rgba = np.zeros((1, 1, 4), dtype=np.uint8)
        rgba[0, 0] = [128, 128, 128, 255]
        lin = extract_linear_height(rgba, "luma", "srgb")[0, 0]
        single = extract_linear_height(rgba, "r", "srgb")[0, 0]
        self.assertAlmostEqual(lin, single, places=12)
        self.assertLess(lin, 0.5)  # 显示编码 128 不等于线性 0.5

    def test_invert(self):
        rgba = np.full((1, 1, 4), 255, dtype=np.uint8)
        self.assertAlmostEqual(
            float(extract_linear_height(rgba, "r", "linear", invert=True)[0, 0]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
