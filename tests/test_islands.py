import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if (_ROOT / ".pylibs").is_dir():
    sys.path.insert(0, str(_ROOT / ".pylibs"))

import numpy as np

from hnm.islands import IslandSpec, build_merged_height, evaluate_join, resample_island
from hnm.kernel import ScaleRequired
from hnm.samples import island_pair_fields


def spec(h, z=None, flip_cols=False, label="x"):
    return IslandSpec(h, 0.5, 0.5, z, label=label, flip_cols=flip_cols)


class TestResample(unittest.TestCase):
    def test_identity_grid_roundtrip(self):
        h = np.random.default_rng(1).random((32, 40))
        out = resample_island(h, 32, 40)
        np.testing.assert_allclose(out, h, atol=1e-12)

    def test_flip_roundtrip(self):
        h = np.arange(16).reshape(4, 4).astype(float)
        out = resample_island(h, 4, 4, flip_cols=True)
        np.testing.assert_array_equal(out, h[:, ::-1])


class TestIslandJoin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hA, cls.hB, cls.truth = island_pair_fields(128)
        cls.z = cls.truth["zScale"]

    def test_physical_merge_blocked_without_b_scale(self):
        with self.assertRaises(ScaleRequired) as ctx:
            evaluate_join(spec(self.hA, self.z, label="A"), spec(self.hB, None, label="B"))
        self.assertIn("island_B.z_per_height_unit", ctx.exception.missing)

    def test_relative_preview_allowed_without_scale(self):
        merged = build_merged_height(
            spec(self.hA, self.z, label="A"),
            spec(self.hB, None, label="B"),
            require_scales=False,
            assumed_z_b=self.z,
        )
        self.assertFalse(merged["physical"])
        self.assertEqual(merged["heightWorld"].shape[1], merged["colsA"] + merged["colsB"])

    def test_correct_scale_and_orientation_passes(self):
        res = evaluate_join(
            spec(self.hA, self.z, label="A"),
            spec(self.hB, self.z, label="B"),
            slope_tol=0.12,
        )
        p = res["profile"]
        self.assertTrue(res["checks"]["passed"], res["checks"]["messages"])
        self.assertEqual(p["polarity"], "bump")
        # 缝中心高斯导数为 0，跨缝割线坡度应恢复背景平面 beta
        self.assertAlmostEqual(p["slopeCross"], self.truth["beta"], delta=0.03)
        self.assertGreater(p["mirrorCorrelation"], 0.7)

    def test_wrong_scale_fails_slope_check(self):
        res = evaluate_join(
            spec(self.hA, self.z, label="A"),
            spec(self.hB, 2.0 * self.z, label="B"),
            slope_tol=0.12,
        )
        self.assertFalse(res["checks"]["slopeConsistent"])
        self.assertFalse(res["checks"]["passed"])

    def test_wrong_mirror_orientation_fails(self):
        res = evaluate_join(
            spec(self.hA, self.z, label="A"),
            spec(self.hB, self.z, flip_cols=True, label="B"),
            slope_tol=0.12,
        )
        self.assertFalse(res["checks"]["passed"])
        self.assertFalse(res["checks"]["slopeConsistent"])

    def test_quantized_8bit_assets_still_interpretable(self):
        # 模拟真实 PNG 8-bit 量化后，正确标尺仍给出小残差（容差放宽）
        qA = np.round(self.hA * 255) / 255
        qB = np.round(self.hB * 255) / 255
        res = evaluate_join(
            spec(qA, self.z, label="A"),
            spec(qB, self.z, label="B"),
            slope_tol=0.35,
        )
        self.assertTrue(res["checks"]["slopeConsistent"], res["profile"]["slopeResidual"])
        self.assertEqual(res["profile"]["polarity"], "bump")


if __name__ == "__main__":
    unittest.main()
