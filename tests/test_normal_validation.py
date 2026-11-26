"""尺度一致性验收：所有期望值独立推导，正反例 + 受控错误。

约定
====
* 期望法线/坡度只来自 hnm.normal_validation 的**解析式**，不借用被验收内核的中间量；
* 与渲染响应相关的断言从独立实现的 render_lit（无高光 Lambert）推导；
* 反例必须被**检出**（翻轴、换单位伪凹凸、错标尺跨缝）；
* 受控错误：故意篡改内核输出（翻转一个轴 / 错用尺度）时，角差与渲染响应
  都必须暴露坏结果，证明验收不是空转。
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if (_ROOT / ".pylibs").is_dir():
    sys.path.insert(0, str(_ROOT / ".pylibs"))

import numpy as np

from hnm import normal_validation as nv
from hnm.conventions import resolve_signs
from hnm.kernel import (
    SurfaceScale,
    decode_normal_rgb,
    encode_normal_rgb,
    height_to_normals,
    normals_from_slopes,
)
from hnm.samples import SAMPLE_REGISTRY, ramp_field, sine_field


class TestAnalyticReferences(unittest.TestCase):
    """独立参照必须与真实导数一致（用机器精度斜坡 / 正弦核对）。"""

    def test_ramp_reference_matches_closed_form(self):
        rows, cols, z, W, H = 64, 96, 0.5, 2.0, 1.0
        sf = nv.analytic_ramp_slope(rows, cols, W, H, z)
        # 闭式 dH/dX = z*amp*coefU/W, dH/dY = -z*amp*coefV/H
        np.testing.assert_allclose(sf.sx, z * 0.72 * 0.62 / W, atol=1e-12)
        np.testing.assert_allclose(sf.sy, -z * 0.72 * 0.38 / H, atol=1e-12)
        np.testing.assert_allclose(sf.dx, W / cols, atol=1e-12)

    def test_sine_reference_derivative(self):
        n, z = 64, 0.108
        sf = nv.analytic_sine_slope(n, n, 1.0, 1.0, z)
        u = (np.arange(n) + 0.5) / n
        U, V = np.meshgrid(u, u)
        sx_ref = z * 0.28 * 6 * np.pi * np.cos(6 * np.pi * U) * np.cos(4 * np.pi * V)
        sy_ref = z * 0.28 * 4 * np.pi * np.sin(6 * np.pi * U) * np.sin(4 * np.pi * V)
        np.testing.assert_allclose(sf.sx, sx_ref, atol=1e-12)
        np.testing.assert_allclose(sf.sy, sy_ref, atol=1e-12)

    def test_angular_error_not_folded_by_flip(self):
        # 翻轴必须给出大角差，不能被 arccos 折叠成 0
        n = np.array([[[0.3, 0.5, 0.81]]], dtype=float)
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        flipped = n.copy()
        flipped[..., 1] *= -1
        ang = nv.angular_error_deg(n, flipped)
        self.assertGreater(float(ang[0, 0]), 30.0)


class TestSlopePaths(unittest.TestCase):
    """切线基内求斜率 vs 按真实长度作差：各向同性时逐位相同，侧光响应一致。"""

    def test_paths_identical_isotropic(self):
        h = nv._ramp_texels(64, 64)
        sx_t, sy_t = nv.slope_in_tangent_basis(h, 1 / 64, 1 / 64, 0.5, "clamp")
        sx_l, sy_l = nv.slope_by_true_length(h, 1 / 64, 1 / 64, 0.5, "clamp")
        np.testing.assert_allclose(sx_t, sx_l, atol=1e-12)
        np.testing.assert_allclose(sy_t, sy_l, atol=1e-12)

    def test_paths_identical_anisotropic(self):
        h = nv._ramp_texels(48, 96)
        dx, dy = 2.0 / 96, 1.0 / 48
        sx_t, sy_t = nv.slope_in_tangent_basis(h, dx, dy, 0.4, "clamp")
        sx_l, sy_l = nv.slope_by_true_length(h, dx, dy, 0.4, "clamp")
        np.testing.assert_allclose(sx_t, sx_l, atol=1e-12)

    def test_render_agrees_finite_difference(self):
        rows, cols, z = 64, 64, 0.5
        h = nv._ramp_texels(rows, cols)
        dx = 2.0 / cols
        n_tan = normals_from_slopes(*nv.slope_in_tangent_basis(h, dx, 1 / rows, z, "clamp"))
        n_len = normals_from_slopes(*nv.slope_by_true_length(h, dx, 1 / rows, z, "clamp"))
        ndl_t = nv.render_lit(n_tan, nv.GRAZING_AZIMUTH_DEG, nv.GRAZING_ELEVATION_DEG)
        ndl_l = nv.render_lit(n_len, nv.GRAZING_AZIMUTH_DEG, nv.GRAZING_ELEVATION_DEG)
        self.assertLess(float(np.max(np.abs(ndl_t - ndl_l))), 1e-9)


class TestVerificationCases(unittest.TestCase):
    """四个正反例的端到端断言，直接对应执行证据。"""

    def test_clamp_ramp_passes_and_exposes_faults(self):
        r = nv.verify_clamp_ramp()
        self.assertTrue(r["passed"], r)
        self.assertLess(r["angle"]["max"], 0.5)
        # 反例：翻轴被检出
        self.assertTrue(r["axisFlipDetected"], r["axisFlipAngleMax"])
        self.assertGreater(r["axisFlipAngleMax"], 1.0)
        # 反例：换单位伪凹凸被检出
        self.assertTrue(r["unitMismatchDetected"])
        self.assertGreater(r["unitSlopeRelMismatch"], 0.05)

    def test_periodic_boundary_interior_and_distinct_edges(self):
        r = nv.verify_periodic_boundary()
        self.assertTrue(r["passed"], r)
        self.assertLessEqual(r["interior"]["max"], 0.5)
        self.assertGreater(r["boundaryPolicyDeltaDeg"], 0.5)

    def test_stretch_and_mirror_same_texels(self):
        r = nv.verify_stretch_and_mirror()
        self.assertLess(r["stretchAngle"]["max"], 0.5)
        self.assertLess(r["mirrorAngle"]["max"], 0.5)
        self.assertTrue(r["stretchHalfEnergy"])
        self.assertTrue(r["renderPathsAgree"])

    def test_unit_scaling_vs_detail_and_pseudo_bump(self):
        r = nv.verify_unit_vs_detail()
        # 真实换单位：法线不变
        self.assertTrue(r["unitInvariant"])
        self.assertLessEqual(r["unitInvarianceAngleMax"], 1e-6)
        # 真实细节变化：换解析期望也能对上
        self.assertTrue(r["detailIsExplained"])
        # 只换单位造成的伪凹凸：必须检出
        self.assertTrue(r["unitPseudoBumpDetected"])
        # 渲染响应必须分辨出伪凹凸（z 放大显著抬高斜光响应）
        self.assertGreater(r["ndlCmRelabel"] - r["ndl1m"], 0.15)


class TestControlledFault(unittest.TestCase):
    """受控错误：篡改内核输出，验收必须暴露坏结果（角差与渲染响应双向）。

    用更陡的双轴斜坡（z=0.9），使翻轴与尺度错配带来的法线差异显著，
    避免退化几何（纯单轴缓坡翻转几乎不改变法线）掩盖错误。
    """

    ROWS = COLS = 64
    Z = 0.9

    def _ramp(self):
        u = (np.arange(self.COLS) + 0.5) / self.COLS
        v = (np.arange(self.ROWS) + 0.5) / self.ROWS
        U, V = np.meshgrid(u, v)
        return np.ascontiguousarray(np.clip(0.1 + 0.8 * (0.62 * U + 0.38 * V), 0.0, 1.0))

    def _exp(self):
        return nv.analytic_ramp_slope(self.ROWS, self.COLS, 1.0, 1.0, self.Z).normals()

    def test_axis_flip_exposed_by_angle_and_render(self):
        h = self._ramp()
        n = height_to_normals(h, SurfaceScale(1.0, 1.0, self.Z), boundary="clamp")["normals"]
        signs = resolve_signs("opengl")
        tex = encode_normal_rgb(n, signs)
        measured_ok = decode_normal_rgb(tex, signs, True)
        tex_bad = tex.copy()
        tex_bad[..., 1] = 255 - tex_bad[..., 1]
        measured_bad = decode_normal_rgb(tex_bad, signs, True)

        exp = self._exp()
        v_ok = nv.verify_normals_against_reference(measured_ok, exp, "clamp", quantized=True)
        v_bad = nv.verify_normals_against_reference(measured_bad, exp, "clamp", quantized=True)
        self.assertTrue(v_ok["passed"])
        self.assertFalse(v_bad["passed"])
        self.assertGreater(v_bad["angle"]["max"], 5.0)
        self.assertGreater(v_bad["grazingMeanAbsDelta"], 3.0 * v_ok["grazingMeanAbsDelta"])

    def test_wrong_scale_exposed(self):
        h = self._ramp()
        exp = self._exp()
        good = height_to_normals(h, SurfaceScale(1.0, 1.0, self.Z), boundary="clamp")["normals"]
        bad = height_to_normals(h, SurfaceScale(1.0, 1.0, self.Z * 10), boundary="clamp")["normals"]
        v_good = nv.verify_normals_against_reference(good, exp, "clamp", quantized=True)
        v_bad = nv.verify_normals_against_reference(bad, exp, "clamp", quantized=True)
        self.assertTrue(v_good["passed"])
        self.assertFalse(v_bad["passed"])
        self.assertGreater(v_bad["angle"]["max"], 20.0)
        self.assertGreater(v_bad["grazingMeanAbsDelta"], 3.0 * v_good["grazingMeanAbsDelta"])


class TestSampleLevel(unittest.TestCase):
    """对注册的真实样例（含 8-bit PNG 读回）做独立复核。"""

    def _load(self, sid):
        from hnm.images import load_rgba
        from hnm.samples import ASSET_DIR
        from hnm.transfer import extract_linear_height

        e = next(x for x in SAMPLE_REGISTRY if x["id"] == sid)
        path = ASSET_DIR / e["file"]
        if not path.exists():
            self.skipTest(f"样例资产未生成: {e['file']}")
        rgba = load_rgba(path)
        h = extract_linear_height(rgba, e["channel"], e["transfer"])
        return e, h

    def test_plane_ramp_readback_matches_reference(self):
        e, h = self._load("plane_ramp")
        n = height_to_normals(
            h, SurfaceScale(e["widthWorld"], e["heightWorld"], e["zScale"], e["unit"]),
            boundary=e["boundary"],
        )["normals"]
        exp = nv.reference_normals_for_sample(
            "plane_ramp", h.shape[0], h.shape[1], e["zScale"], e["widthWorld"], e["heightWorld"]
        )
        v = nv.verify_normals_against_reference(n, exp, "clamp", quantized=True)
        self.assertTrue(v["passed"], v)

    def test_sine_periodic_readback_matches_reference(self):
        e, h = self._load("sine_relief")
        n = height_to_normals(
            h, SurfaceScale(e["widthWorld"], e["heightWorld"], e["zScale"], e["unit"]),
            boundary=e["boundary"],
        )["normals"]
        exp = nv.reference_normals_for_sample(
            "sine_relief", h.shape[0], h.shape[1], e["zScale"], e["widthWorld"], e["heightWorld"]
        )
        v = nv.verify_normals_against_reference(n, exp, "periodic", quantized=True)
        self.assertTrue(v["passed"], v)

    def test_stretch_and_mirror_samples_same_texels(self):
        _, h_s = self._load("scale_ramp_stretched")
        _, h_m = self._load("scale_ramp_mirrored")
        # 镜像样例是拉长样例列序镜像（忽略最后的等值列差异）
        self.assertTrue(np.array_equal(h_s[:, ::-1], h_m))
        # 拉长样例的坡度能量应约为镜像样例的一半
        e_s = next(x for x in SAMPLE_REGISTRY if x["id"] == "scale_ramp_stretched")
        e_m = next(x for x in SAMPLE_REGISTRY if x["id"] == "scale_ramp_mirrored")
        r_s = height_to_normals(h_s, SurfaceScale(e_s["widthWorld"], e_s["heightWorld"],
                                                  e_s["zScale"], e_s["unit"]), boundary="clamp")
        r_m = height_to_normals(h_m, SurfaceScale(e_m["widthWorld"], e_m["heightWorld"],
                                                  e_m["zScale"], e_m["unit"]), boundary="clamp")
        ratio = np.mean(np.abs(r_s["slope_x"][2:-2, 2:-2])) / np.mean(np.abs(r_m["slope_x"][2:-2, 2:-2]))
        self.assertAlmostEqual(float(ratio), 0.5, delta=0.02)


class TestRunAllVerification(unittest.TestCase):
    """汇总入口：所有证据为真、passed 与逐项一致。"""

    def test_run_all_passed_and_evidence_consistent(self):
        r = nv.run_all_verification()
        ev = r["evidence"]
        self.assertTrue(r["passed"], ev)
        self.assertTrue(ev["clampRampPassed"])
        self.assertTrue(ev["axisFlipDetected"])
        self.assertTrue(ev["unitPseudoBumpDetected"])
        self.assertTrue(ev["unitInvariant"])
        self.assertTrue(ev["periodicBoundaryDistinct"])
        self.assertTrue(ev["correctCrossesSeam"])
        self.assertTrue(ev["wrongScaleBreaksSeam"])
        self.assertTrue(ev["relativeOnlyProvableRange"])
        self.assertLess(ev["clampRampAngleMaxDeg"], 0.5)
        self.assertLess(ev["periodicInteriorAngleMaxDeg"], 0.5)
        # 证据自洽：好/坏跨缝比值应可分
        self.assertGreater(ev["wrongScaleSeamRatio"], ev["correctSeamRatio"])


class TestNoSelfFulfillingProof(unittest.TestCase):
    """期望值独立：即使内核整体翻转一个轴，独立参照仍能判其为错。"""

    def test_reference_is_independent_of_kernel(self):
        rows, cols, z = 48, 72, 0.5
        h = ramp_field(max(rows, cols))[:rows, :cols]  # 裁剪会破坏 UV，仅取形状用于方向判定
        # 直接用解析斜坡本身的法线作为“被验收对象”，再翻绿轴
        exp = nv.analytic_ramp_slope(rows, cols, 2.0, 1.0, z).normals()
        flipped = exp.copy()
        flipped[..., 1] *= -1
        good = nv.verify_normals_against_reference(exp, exp, "clamp", quantized=False)
        bad = nv.verify_normals_against_reference(flipped, exp, "clamp", quantized=False)
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])

    def test_broken_engine_is_caught_not_whistled_through(self):
        """受控篡改内核尺度：验收必须判坏，证明不是自洽伪报。

        把 physical_slopes 放大 3×（典型“强度×像素差分”错误），再用独立解析
        参照核对内核输出的法线 —— 角差应显著、裁决应为 FAIL。
        """
        from hnm import kernel as k

        orig = k.physical_slopes

        def driven_broken(h, scale, boundary="clamp", method="central"):
            sx, sy, dx, dy = orig(h, scale, boundary, method)
            return sx * 3.0, sy * 3.0, dx, dy

        k.physical_slopes = driven_broken
        try:
            h = ramp_field(64)
            res = k.height_to_normals(h, SurfaceScale(1.0, 1.0, 0.4))
            exp = nv.analytic_ramp_slope(64, 64, 1.0, 1.0, 0.4).normals()
            v = nv.verify_normals_against_reference(res["normals"], exp, "clamp", quantized=False)
            self.assertFalse(v["passed"])
            self.assertGreater(v["angle"]["max"], 5.0)
        finally:
            k.physical_slopes = orig


if __name__ == "__main__":
    unittest.main()
