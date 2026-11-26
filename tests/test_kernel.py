import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if (_ROOT / ".pylibs").is_dir():
    sys.path.insert(0, str(_ROOT / ".pylibs"))

import numpy as np

from hnm.conventions import resolve_signs
from hnm.kernel import (
    SurfaceScale,
    angular_error_deg,
    decode_normal_rgb,
    encode_normal_rgb,
    encoding_error_metrics,
    height_to_normals,
    index_gradients,
)
from hnm import samples


class TestAnalyticNormals(unittest.TestCase):
    def test_planar_ramp_uniform(self):
        n = 64
        h = samples.ramp_field(n)
        scale = SurfaceScale(1.0, 1.0, 0.4)
        res = height_to_normals(h, scale)
        normals = res["normals"]

        dc = 0.72 * 0.62 / n
        dr = 0.72 * 0.38 / n
        dx = dy = 1.0 / n
        sx = 0.4 * dc / dx
        sy = -0.4 * dr / dy
        expected = np.array([-sx, -sy, 1.0])
        expected /= np.linalg.norm(expected)

        err = np.abs(normals[3:-3, 3:-3] - expected).max()
        self.assertLess(err, 1e-12, "线性斜坡内部必须与解析法线一致")

    def test_planar_ramp_anisotropic_scale(self):
        # 宽 2m / 100 列 vs 高 0.25m / 50 行：不等距像素坐标
        nc, nr = 100, 50
        u = (np.arange(nc) + 0.5) / nc
        v = (np.arange(nr) + 0.5) / nr
        U, V = np.meshgrid(u, v)
        h = 0.1 + 0.5 * U + 0.25 * V
        scale = SurfaceScale(2.0, 0.25, 0.1)
        res = height_to_normals(h, scale)

        dx, dy = 2.0 / nc, 0.25 / nr
        sx = 0.1 * (0.5 / nc) / dx
        sy = -0.1 * (0.25 / nr) / dy
        expected = np.array([-sx, -sy, 1.0])
        expected /= np.linalg.norm(expected)
        err = np.abs(res["normals"][2:-2, 2:-2] - expected).max()
        self.assertLess(err, 1e-12)
        self.assertAlmostEqual(res["scaleInfo"]["pixelPitchX"], 0.02)
        self.assertAlmostEqual(res["scaleInfo"]["pixelPitchY"], 0.005)
        self.assertAlmostEqual(res["scaleInfo"]["texelAspect"], 4.0)

    def test_sine_relief_matches_analytic(self):
        n = 256
        h = samples.sine_field(n)
        W = H = 1.0
        z = 0.108
        fx, fy = 3, 2
        res = height_to_normals(h, SurfaceScale(W, H, z), boundary="periodic")
        U = (np.arange(n) + 0.5) / n
        V = (np.arange(n) + 0.5) / n
        U, V = np.meshgrid(U, V)
        sx = z * 0.28 * (2 * np.pi * fx / W) * np.cos(2 * np.pi * fx * U) * np.cos(
            2 * np.pi * fy * V
        )
        sy = (
            z
            * 0.28
            * (2 * np.pi * fy / H)
            * np.sin(2 * np.pi * fx * U)
            * np.sin(2 * np.pi * fy * V)
        )
        expected = np.stack([-sx, -sy, np.ones_like(sx)], axis=2)
        expected /= np.linalg.norm(expected, axis=2, keepdims=True)
        ang = angular_error_deg(res["normals"][4:-4, 4:-4], expected[4:-4, 4:-4])
        self.assertLess(float(np.max(ang)), 0.2)
        self.assertLess(float(np.mean(ang)), 0.02)

    def test_central_difference_second_order_convergence(self):
        def max_err(n):
            h = samples.sine_field(n)
            res = height_to_normals(
                h, SurfaceScale(1, 1, 0.1), boundary="periodic", method="central"
            )
            U = (np.arange(n) + 0.5) / n
            U, V = np.meshgrid(U, U)
            sx = 0.1 * 0.28 * 6 * np.pi * np.cos(6 * np.pi * U) * np.cos(4 * np.pi * V)
            sy = 0.1 * 0.28 * 4 * np.pi * np.sin(6 * np.pi * U) * np.sin(4 * np.pi * V)
            exp = np.stack([-sx, -sy, np.ones_like(sx)], 2)
            exp /= np.linalg.norm(exp, axis=2, keepdims=True)
            return float(np.max(angular_error_deg(res["normals"][3:-3], exp[3:-3])))

        e64, e128 = max_err(64), max_err(128)
        self.assertLess(e128, 0.4 * e64, f"中心差分应二阶收敛: {e64=} {e128=}")


class TestBoundaryPolicy(unittest.TestCase):
    def setUp(self):
        self.h = np.zeros((8, 8))
        self.h[:, :4] = 0.3
        self.h[:, 4:] = 0.7

    def test_clamp_edges_have_no_wrap_artifact(self):
        dc, _ = index_gradients(self.h, boundary="clamp")
        np.testing.assert_allclose(dc[:, 0], 0.0)
        np.testing.assert_allclose(dc[:, -1], 0.0)
        np.testing.assert_allclose(dc[:, 3], 0.2)
        np.testing.assert_allclose(dc[:, 4], 0.2)

    def test_periodic_edges_see_wrap_step(self):
        dc, _ = index_gradients(self.h, boundary="periodic")
        np.testing.assert_allclose(dc[:, 0], -0.2)
        np.testing.assert_allclose(dc[:, -1], -0.2)


class TestEncodingAndConventions(unittest.TestCase):
    def test_roundtrip_error_bounded(self):
        rng = np.random.default_rng(0)
        n = rng.normal(size=(64, 64, 3))
        n[:, :, 2] = np.abs(n[:, :, 2]) + 1.5  # 限制在迎光半球
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        metrics = encoding_error_metrics(n, resolve_signs("opengl"))
        self.assertLess(metrics["maxAngularErrorDeg"], 1.0)

    def test_directx_only_flips_green(self):
        n = np.array([[[0.25, -0.4, 0.88]]])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        gl = encode_normal_rgb(n, resolve_signs("opengl")).astype(int)
        dx = encode_normal_rgb(n, resolve_signs("directx")).astype(int)
        self.assertAlmostEqual(dx[0, 0, 0], gl[0, 0, 0], delta=1)
        self.assertAlmostEqual(dx[0, 0, 2], gl[0, 0, 2], delta=1)
        self.assertAlmostEqual(dx[0, 0, 1], 255 - gl[0, 0, 1], delta=1)

        back_gl = decode_normal_rgb(gl.astype(np.uint8), resolve_signs("opengl"))
        back_dx = decode_normal_rgb(dx.astype(np.uint8), resolve_signs("directx"))
        np.testing.assert_allclose(back_gl, back_dx, atol=1e-12)

    def test_explicit_axis_flip_explained_by_signs(self):
        n = np.array([[[0.3, 0.5, 0.81]]])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        base = encode_normal_rgb(n, (1, 1, 1)).astype(int)
        flipped = encode_normal_rgb(n, (-1, 1, 1)).astype(int)
        self.assertAlmostEqual(flipped[0, 0, 0], 255 - base[0, 0, 0], delta=1)
        self.assertAlmostEqual(flipped[0, 0, 1], base[0, 0, 1], delta=1)


class TestPhysicalScaleInvariance(unittest.TestCase):
    def _surface(self, n):
        u = (np.arange(n) + 0.5) / n
        U, _ = np.meshgrid(u, u)
        return 0.5 + 0.3 * np.sin(2 * np.pi * 2 * U)

    def test_doubling_resolution_does_not_double_strength(self):
        out = {}
        for n in (64, 128, 256):
            # 该表面周期可平铺，用 periodic 统计全幅，避免按像素裁边
            res = height_to_normals(
                self._surface(n), SurfaceScale(1, 1, 0.1), boundary="periodic"
            )
            out[n] = float(np.mean(np.abs(res["slope_x"])))
        # 物理表面不变时，平均凹凸强度只允许 sinc 离散因子级别的微小变化，
        # 绝不允许随分辨率翻倍（错误的“强度×像素差分”公式会近似线性翻倍）。
        self.assertLess(abs(out[128] / out[64] - 1.0), 0.006)
        self.assertLess(abs(out[256] / out[128] - 1.0), 0.002)
        # 且都收敛到解析平均坡度 z*A*k*mean|cos| = 0.24
        self.assertAlmostEqual(out[256], 0.24, delta=0.002)

    def test_stretch_halves_slope(self):
        h = samples.ramp_field(64)
        base = height_to_normals(h, SurfaceScale(1, 1, 0.2), boundary="clamp")
        stretched = height_to_normals(
            h, SurfaceScale(1, 1, 0.2, stretch_x=2.0, stretch_y=3.0), boundary="clamp"
        )
        # 内部点比较
        ratio_x = (stretched["slope_x"][4:-4, 4:-4] / base["slope_x"][4:-4, 4:-4]).mean()
        ratio_y = (stretched["slope_y"][4:-4, 4:-4] / base["slope_y"][4:-4, 4:-4]).mean()
        self.assertAlmostEqual(ratio_x, 0.5, places=10)
        self.assertAlmostEqual(ratio_y, 1.0 / 3.0, places=10)


if __name__ == "__main__":
    unittest.main()
