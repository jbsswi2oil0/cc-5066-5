import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / ".pylibs").is_dir():
    sys.path.insert(0, str(_ROOT / ".pylibs"))

import numpy as np

from hnm.lighting import light_vector, shade


def _normal(sx, sy):
    v = np.array([[[-sx, -sy, 1.0]]])
    return v / np.linalg.norm(v, axis=2, keepdims=True)


class TestLightingDirection(unittest.TestCase):
    def test_light_vector_axes(self):
        np.testing.assert_allclose(light_vector(0, 0), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(light_vector(90, 0), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(light_vector(0, 90), [0, 0, 1], atol=1e-12)

    def test_light_from_right_brightens_right_flank(self):
        right = _normal(0.6, 0.0)   # dH/dX=+0.6 -> nx<0? careful:
        # normals_from_slopes gives nx=-sx. 右坡（朝+X）对应 sx<0 -> nx>0。
        right_facing = _normal(-0.6, 0.0)
        left_facing = _normal(0.6, 0.0)
        img_r = shade(right_facing, azimuth_deg=0, elevation_deg=10, ambient=0.0, specular=0.0)
        img_l = shade(left_facing, azimuth_deg=0, elevation_deg=10, ambient=0.0, specular=0.0)
        self.assertGreater(int(img_r[0, 0, 0]), int(img_l[0, 0, 0]))

    def test_light_from_top_brightens_top_flank(self):
        top_facing = _normal(0.0, -0.6)   # sy<0 -> ny>0（+Y 朝上）
        bottom_facing = _normal(0.0, 0.6)
        img_t = shade(top_facing, azimuth_deg=90, elevation_deg=10, ambient=0.0, specular=0.0)
        img_b = shade(bottom_facing, azimuth_deg=90, elevation_deg=10, ambient=0.0, specular=0.0)
        self.assertGreater(int(img_t[0, 0, 0]), int(img_b[0, 0, 0]))

    def test_opposite_light_reverses_highlight(self):
        flank = _normal(-0.5, 0.2)
        a = shade(flank, azimuth_deg=0, elevation_deg=30, ambient=0.05, specular=0.0)
        b = shade(flank, azimuth_deg=180, elevation_deg=30, ambient=0.05, specular=0.0)
        self.assertGreater(int(a[0, 0, 0]), int(b[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
