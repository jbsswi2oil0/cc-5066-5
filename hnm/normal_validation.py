"""独立参照的尺度一致性验收（期望值不借用被验收内核的中间量）。

为什么要“独立”
==============
若用 kernel 自己算出的坡度去核对法线，坏内核也会“自洽通过”。这里所有期望的
法线/坡度都由解析高度式直接求导得到，只把被测内核输出的**法线**拿来比；
像素间距只用于“真实长度作差”与真实材质渲染的坐标映射。

覆盖
====
* 斜坡（clamp）：正确参数 PASS；故意反转一个轴 FAIL；换单位伪凹凸 FAIL；
* 周期边界：内部与解析法线相容；逐位边界 clamp 与 periodic 行为明确不同；
* 同样的纹素斜坡：模型拉长一倍 vs 镜像 UV 反向，肩并肩给出角差与渲染响应；
* 重设纹素密度/轴向：区分“真实细节变化”与“换单位伪凹凸”。

坡度两种写法并列
================
* tangent-basis：先对纹素做索引差分，再除以像素间距、乘以高度标尺；
* true-length  ：先把纹素换成真实世界坐标，再对真实长度作差。
各向同性时两者逐位相同；拉伸/换单位时两者的法线与侧光响应都必须仍与解析期望一致
（若某一写法把 dx 用错，渲染响应就会偏离，见 evidence）。

度量
====
法线角差用 atan2(|a×b|, a·b)，全 0..180° 单调，翻轴不会被折叠成 0。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .conventions import resolve_signs
from .kernel import (
    SurfaceScale,
    decode_normal_rgb,
    encode_normal_rgb,
    height_to_normals,
    index_gradients,
    normals_from_slopes,
)

# 验收阈值集中在此，测试与 CLI/界面共用同一组数字，避免“改阈值就过”。
SLOPE_ABS_TOL = 1e-3
ANGLE_INTERIOR_DEG = 0.5
ANGLE_PERIODIC_DEG = 0.5
AXIS_FLIP_MIN_ANGLE_DEG = 1.0
UNIT_SCALE_MIN_MISMATCH = 0.05
GRAZING_AZIMUTH_DEG = 150.0
GRAZING_ELEVATION_DEG = 15.0
RENDER_AGREEMENT_TOL = 1e-6   # 两条坡度写法在两套坐标系下应完全等价（0..1 亮度差）
ANGLE_QUANTIZED_DEG = 15.0    # 8-bit 编码读回：好(non-flipped)实测≤10°，翻轴≥21°
RENDER_QUANTIZED_MAD = 0.10   # 8-bit 读回：逐像素平均亮度差（好≤0.05，翻轴≥0.10）
RENDER_QUANTIZED_MAX = 0.20   # 8-bit 读回：逐像素最大亮度差（好≤0.12，翻轴≥0.29）
RENDER_EXACT_MAD = 1e-9       # 未量化（内核法线 vs 解析）：应逐像素一致
RENDER_EXACT_MAX = 1e-9
UNIT_RELABEL_TOL = 1e-6       # 世界宽/高与注册表真值（×真实拉伸）的相对误差上界
UNIT_INVARIANCE_DEG = 1e-6    # 同一物理表面仅换单位时，法线必须不变（< 微度）


# ---------------------------------------------------------------------------
# 解析参照
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlopeField:
    """解析高度面的真实坡度与真实像素间距。"""

    sx: np.ndarray
    sy: np.ndarray
    dx: float
    dy: float

    def normals(self) -> np.ndarray:
        nx = -self.sx
        ny = -self.sy
        nz = np.ones_like(self.sx)
        n = np.stack([nx, ny, nz], axis=2)
        return n / np.linalg.norm(n, axis=2, keepdims=True)


def analytic_ramp_slope(
    rows: int,
    cols: int,
    width_world: float,
    height_world: float,
    z_per_unit: float,
    coef_u: float = 0.62,
    coef_v: float = 0.38,
    amp: float = 0.72,
) -> SlopeField:
    """与 samples.ramp_field 同表面：H = z*(0.12+amp*(coef_u*U+coef_v*V))。

    高度不触 0/1 限幅，故闭式导数精确：
        dH/dX = z*amp*coef_u/width_world
        dH/dY = -z*amp*coef_v/height_world   （+Y 朝上，V 随行向下）
    """
    sx = np.full((rows, cols), z_per_unit * amp * coef_u / width_world, dtype=np.float64)
    sy = np.full((rows, cols), -z_per_unit * amp * coef_v / height_world, dtype=np.float64)
    return SlopeField(_contig(sx), _contig(sy), width_world / cols, height_world / rows)


def analytic_sine_slope(
    rows: int,
    cols: int,
    width_world: float,
    height_world: float,
    z_per_unit: float,
    fx: int = 3,
    fy: int = 2,
    amp: float = 0.28,
) -> SlopeField:
    """与 samples.sine_field 同表面：h=0.5+amp*sin(2πfxU)cos(2πfyV)。"""
    u = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    v = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    U, V = np.meshgrid(u, v)
    wu = 2 * np.pi * fx
    wv = 2 * np.pi * fy
    sx = z_per_unit * amp * wu / width_world * np.cos(wu * U) * np.cos(wv * V)
    sy = z_per_unit * amp * wv / height_world * np.sin(wu * U) * np.sin(wv * V)
    return SlopeField(_contig(sx), _contig(sy), width_world / cols, height_world / rows)


def _contig(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def angular_error_deg(n_true: np.ndarray, n_test: np.ndarray) -> np.ndarray:
    """atan2(|a×b|, a·b)：全 0..180° 单调，翻轴不折叠。"""
    a = np.asarray(n_true, dtype=np.float64)
    b = np.asarray(n_test, dtype=np.float64)
    dot = np.sum(a * b, axis=2)
    cross = np.cross(a, b)
    cross_norm = np.sqrt(np.sum(cross * cross, axis=2))
    return np.clip(np.degrees(np.arctan2(cross_norm, dot)), 0.0, 180.0)


# ---------------------------------------------------------------------------
# 坡度两种写法
# ---------------------------------------------------------------------------


def slope_in_tangent_basis(
    h: np.ndarray, dx: float, dy: float, z_per_unit: float, boundary: str = "clamp"
) -> tuple[np.ndarray, np.ndarray]:
    """先对纹素（索引）差分，再除以像素间距、乘以高度标尺。"""
    dc, dr = index_gradients(h, boundary, "central")
    return z_per_unit * dc / dx, -z_per_unit * dr / dy


def slope_by_true_length(
    h: np.ndarray, dx: float, dy: float, z_per_unit: float, boundary: str = "clamp"
) -> tuple[np.ndarray, np.ndarray]:
    """先把纹素换成真实世界坐标 x=X*dx, y=Y*dy，再对真实长度作差分。

    规则网格下数学上等价于索引差分再除以 dx/dy；并列两写法是为了在拉伸/
    镜像/换单位时检查“谁在动尺度、谁在动细节”。
    """
    rows, cols = h.shape
    if boundary == "periodic":
        hp = np.pad(h, ((0, 0), (1, 1)), mode="wrap")
        dc_len = (hp[:, 2:] - hp[:, :-2]) * 0.5
        hp2 = np.pad(h, ((1, 1), (0, 0)), mode="wrap")
        dr_len = (hp2[2:, :] - hp2[:-2, :]) * 0.5
    else:
        dc_len = np.empty_like(h, dtype=np.float64)
        dr_len = np.empty_like(h, dtype=np.float64)
        dc_len[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
        dc_len[:, 0] = h[:, 1] - h[:, 0]
        dc_len[:, -1] = h[:, -1] - h[:, -2]
        dr_len[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
        dr_len[0, :] = h[1, :] - h[0, :]
        dr_len[-1, :] = h[-1, :] - h[-2, :]
    step_x = (2.0 * dx) * 0.5 if cols > 2 else dx  # x[2]-x[0] 除以 2
    step_y = (2.0 * dy) * 0.5 if rows > 2 else dy
    return z_per_unit * dc_len / step_x, -z_per_unit * dr_len / step_y


# ---------------------------------------------------------------------------
# 独立真实材质渲染（无高光，避免 specular 掩盖方向）
# ---------------------------------------------------------------------------


def render_lit(
    normals: np.ndarray, azimuth_deg: float, elevation_deg: float, ambient: float = 0.22
) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    L = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    ndl = np.clip(np.sum(normals * L.reshape(1, 1, 3), axis=2), 0.0, 1.0)
    return ambient + (1.0 - ambient) * ndl


# ---------------------------------------------------------------------------
# 纹素斜坡几何
# ---------------------------------------------------------------------------


def _ramp_field(rows: int, cols: int) -> np.ndarray:
    """与 samples.ramp_field 同表面，但精确按 (rows, cols) 采样全幅 UV。"""
    u = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    v = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    U, V = np.meshgrid(u, v)
    return np.ascontiguousarray(np.clip(0.12 + 0.72 * (0.62 * U + 0.38 * V), 0.0, 1.0))


def _ramp_texels(rows: int, cols: int) -> np.ndarray:
    """仅沿列变化的线性斜坡，纹素值约 0.1..0.9（不触限幅）。"""
    u = (np.arange(cols, dtype=np.float64) + 0.5) / cols
    v = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    U, V = np.meshgrid(u, v)
    return np.ascontiguousarray(np.clip(0.1 + 0.8 * U, 0.0, 1.0))


def _safe_stats(ang: np.ndarray, margin: int = 2) -> dict:
    if margin <= 0:
        a = np.asarray(ang)
    else:
        a = ang[margin:-margin, margin:-margin]
    return {
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
    }


# ---------------------------------------------------------------------------
# 具体验收：斜坡 / 周期边界 / 拉长与镜像 / 单位与细节
# ---------------------------------------------------------------------------


def verify_clamp_ramp(rows: int = 96, cols: int = 64) -> dict:
    """斜坡正反例：正确 PASS；反转绿轴 FAIL；换单位伪凹凸 FAIL。"""
    z = 0.5
    W, H = 2.0, 1.0
    h = _ramp_field(rows, cols)
    scale = SurfaceScale(W, H, z, unit="m")
    res = height_to_normals(h, scale, boundary="clamp", method="central")
    n = res["normals"]
    expected = analytic_ramp_slope(rows, cols, W, H, z)
    ang = angular_error_deg(expected.normals(), n)

    dx, dy = W / cols, H / rows
    sx_tan, sy_tan = slope_in_tangent_basis(h, dx, dy, z, "clamp")
    sx_len, sy_len = slope_by_true_length(h, dx, dy, z, "clamp")
    slope_err_tan = float(np.max(np.abs(sx_tan - expected.sx)))
    slope_err_len = float(np.max(np.abs(sx_len - expected.sx)))
    path_delta = float(np.max(np.abs(sx_tan - sx_len)))

    stats = _safe_stats(ang)
    passed = stats["max"] <= ANGLE_INTERIOR_DEG and slope_err_tan <= SLOPE_ABS_TOL

    # 反例 A：物理翻转绿通道（约定错/贴图错），再由正确约定读回，必须检出
    good_signs = resolve_signs("opengl")
    tex_good = encode_normal_rgb(n, good_signs)
    n_read_good = decode_normal_rgb(tex_good, good_signs, renormalize=True)
    roundtrip = angular_error_deg(n, n_read_good)
    tex_flipped = tex_good.copy()
    tex_flipped[..., 1] = (255 - tex_flipped[..., 1]).astype(np.uint8)
    n_read_flipped = decode_normal_rgb(tex_flipped, good_signs, renormalize=True)
    flip_ang = angular_error_deg(expected.normals(), n_read_flipped)
    axis_flip_angle = float(np.max(flip_ang[2:-2, 2:-2]))

    # 反例 B：换单位伪凹凸（宽 2m 误写成 2cm，文本仍标 m）
    fake = height_to_normals(h, SurfaceScale(0.02, H, z, unit="m"), boundary="clamp")
    unit_rel = float(
        np.max(np.abs(fake["slope_x"][2:-2, 2:-2] - expected.sx[2:-2, 2:-2]))
        / max(float(np.max(np.abs(expected.sx))), 1e-12)
    )

    return {
        "name": "clamp_ramp",
        "passed": passed,
        "angle": stats,
        "roundtripAngleMax": float(np.max(roundtrip[2:-2, 2:-2])),
        "roundtripFidelityOk": bool(np.max(roundtrip[2:-2, 2:-2]) < 1.0),
        "slopeAbsErrTangent": slope_err_tan,
        "slopeAbsErrTrueLength": slope_err_len,
        "slopePathDelta": path_delta,
        "axisFlipAngleMax": axis_flip_angle,
        "axisFlipDetected": axis_flip_angle >= AXIS_FLIP_MIN_ANGLE_DEG,
        "unitSlopeRelMismatch": unit_rel,
        "unitMismatchDetected": unit_rel >= UNIT_SCALE_MIN_MISMATCH,
    }


def verify_periodic_boundary(n: int = 64) -> dict:
    """周期边界：内部相容；逐位边界 clamp 与 periodic 行为明确不同。"""
    from .samples import sine_field

    z = 0.108
    W = H = 1.0
    h = sine_field(n)
    expected = analytic_sine_slope(n, n, W, H, z)
    exp_n = expected.normals()

    res_p = height_to_normals(h, SurfaceScale(W, H, z), boundary="periodic")
    res_c = height_to_normals(h, SurfaceScale(W, H, z), boundary="clamp")
    ang_p = angular_error_deg(exp_n, res_p["normals"])
    ang_c = angular_error_deg(exp_n, res_c["normals"])
    stats_p = _safe_stats(ang_p)

    edge = np.zeros((n, n), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    policy_delta = float(np.max(np.abs(ang_p - ang_c)[edge]))

    interior_ok = stats_p["max"] <= ANGLE_PERIODIC_DEG
    boundary_distinct = policy_delta >= 0.5
    return {
        "name": "periodic_boundary",
        "passed": interior_ok and boundary_distinct,
        "interior": stats_p,
        "interiorOk": interior_ok,
        "clampEdgeAngleMax": float(np.max(ang_c[edge])),
        "periodicEdgeAngleMax": float(np.max(ang_p[edge])),
        "boundaryPolicyDeltaDeg": policy_delta,
        "boundaryDistinct": boundary_distinct,
    }


def verify_stretch_and_mirror(rows: int = 64, cols: int = 64, z: float = 0.5) -> dict:
    """同样的纹素斜坡：模型拉长一倍 vs 镜像 UV 反向。

    * 拉长：SurfaceScale stretch_x=2 -> 坡度减半；解析期望同步用宽×2。
    * 镜像：纹素逐位相同但列序反转；kernel 中心差分给出反号坡度，
      与“反向 UV 岛”的解析期望一致（不能靠事后猜法线）。
    两写法（切线基内 vs 真实长度）各出法线，再侧光渲染比较一致性。
    """
    h = _ramp_texels(rows, cols)
    base_W = 1.0

    # ----- 拉长一倍 -----
    stretch_scale = SurfaceScale(base_W, 1.0, z, stretch_x=2.0)
    res_stretch = height_to_normals(h, stretch_scale, boundary="clamp")
    exp_stretch = analytic_ramp_slope(
        rows, cols, base_W * 2.0, 1.0, z, coef_u=0.8, coef_v=0.0, amp=1.0
    )
    ang_stretch = angular_error_deg(exp_stretch.normals(), res_stretch["normals"])

    # ----- 镜像 UV 反向 -----
    mirrored = np.ascontiguousarray(h[:, ::-1])
    res_mirror = height_to_normals(mirrored, SurfaceScale(base_W, 1.0, z), boundary="clamp")
    exp_mirror_plain = analytic_ramp_slope(
        rows, cols, base_W, 1.0, z, coef_u=0.8, coef_v=0.0, amp=1.0
    )
    exp_mirror = SlopeField(-exp_mirror_plain.sx, exp_mirror_plain.sy,
                            exp_mirror_plain.dx, exp_mirror_plain.dy)
    ang_mirror = angular_error_deg(exp_mirror.normals(), res_mirror["normals"])

    # ----- 两条坡度写法：内核法线（高度→坡度→法线）与独立渲染 -----
    # 拉长斜坡用真实间距 dx=2W/cols；两条写法都应复现解析坡度，侧光响应也必须一致。
    dx_s = base_W * 2.0 / cols
    dy_s = 1.0 / rows
    n_tan = normals_from_slopes(*slope_in_tangent_basis(h, dx_s, dy_s, z, "clamp"))
    n_len = normals_from_slopes(*slope_by_true_length(h, dx_s, dy_s, z, "clamp"))
    ndl_tan = render_lit(n_tan, GRAZING_AZIMUTH_DEG, GRAZING_ELEVATION_DEG)
    ndl_len = render_lit(n_len, GRAZING_AZIMUTH_DEG, GRAZING_ELEVATION_DEG)
    render_delta = float(np.max(np.abs(ndl_tan - ndl_len)))
    tan_vs_expected = float(np.max(angular_error_deg(exp_stretch.normals(), n_tan)[2:-2, 2:-2]))
    len_vs_expected = float(np.max(angular_error_deg(exp_stretch.normals(), n_len)[2:-2, 2:-2]))

    # 拉长应有约一半的能量，镜像能量不变
    e_stretch = float(np.mean(np.abs(exp_stretch.sx)))
    e_mirror = float(np.mean(np.abs(exp_mirror.sx)))
    stretch_half = abs(e_stretch / e_mirror - 0.5) < 0.01

    return {
        "name": "stretch_and_mirror",
        "stretchAngle": _safe_stats(ang_stretch),
        "mirrorAngle": _safe_stats(ang_mirror),
        "energyStretch": e_stretch,
        "energyMirror": e_mirror,
        "stretchHalfEnergy": bool(stretch_half),
        "tangentPathAngleMax": tan_vs_expected,
        "trueLengthPathAngleMax": len_vs_expected,
        "renderPathDelta": render_delta,
        "renderPathsAgree": render_delta <= RENDER_AGREEMENT_TOL,
        "normalsStretch": res_stretch["normals"],
        "normalsMirror": res_mirror["normals"],
    }


def verify_unit_vs_detail(rows: int = 64, cols: int = 64, z: float = 0.5) -> dict:
    """对同一斜面重设纹素密度与轴向：区分真实细节变化与换单位伪凹凸。

    三组对照，判据清晰：
      1. **真实换单位**（m→cm）：物理表面不变，width/height/z 三者同倍缩放。
         正确法线必须**不变**（对同一物理表面，单位不是几何）。
      2. **真实细节变化**：同一纹素覆盖更宽（1m→2m），每米起伏变小；解析期望
         同步变缓，渲染响应可见地变平 -> 可解释的真实变化。
      3. **换单位伪凹凸**：只改 z、不改 width（或反之），单位混用。物理坡度被
         凭空放大 -> 与真实表面期望角差极大，渲染被夸张 -> 必须检出。
    """
    h = _ramp_texels(rows, cols)
    W, H = 1.0, 1.0

    # 1) 真实换单位：m→cm，宽/高/标尺同倍×100
    n_m = height_to_normals(h, SurfaceScale(W, H, z, unit="m"))["normals"]
    n_cm = height_to_normals(h, SurfaceScale(W * 100, H * 100, z * 100, unit="cm"))["normals"]
    # 注意 SurfaceScale 的 z 是“线性高度 1.0 对应世界高度”，cm 下同为 100 倍
    unit_angle = angular_error_deg(n_m, n_cm)
    unit_invariant = float(np.max(unit_angle)) <= UNIT_INVARIANCE_DEG

    # 2) 真实细节变化：覆盖 1m→2m，同一纹素；期望坡度减半
    exp_1m = analytic_ramp_slope(rows, cols, W, H, z, coef_u=0.8, coef_v=0.0, amp=1.0)
    exp_2m = analytic_ramp_slope(rows, cols, 2 * W, H, z, coef_u=0.8, coef_v=0.0, amp=1.0)
    n_1m = height_to_normals(h, SurfaceScale(W, H, z))["normals"]
    n_2m = height_to_normals(h, SurfaceScale(2 * W, H, z))["normals"]
    ang_2m = angular_error_deg(exp_2m.normals(), n_2m)
    detail_explained = _safe_stats(ang_2m)["max"] <= ANGLE_INTERIOR_DEG

    # 3) 换单位伪凹凸：只放大 z（或等同地只缩小 width），物理坡度×100
    n_cm_relabel = height_to_normals(h, SurfaceScale(W, H, z * 100))["normals"]
    ang_cm_relabel = angular_error_deg(exp_1m.normals(), n_cm_relabel)
    pseudo_detected = _safe_stats(ang_cm_relabel)["max"] >= AXIS_FLIP_MIN_ANGLE_DEG

    def mean_ndl(nn):
        return float(np.mean(render_lit(nn, GRAZING_AZIMUTH_DEG, GRAZING_ELEVATION_DEG)))

    return {
        "name": "unit_vs_detail",
        "unitInvariant": bool(unit_invariant),
        "unitInvarianceAngleMax": float(np.max(unit_angle)),
        "detailIsExplained": bool(detail_explained),
        "detail2mAngle": _safe_stats(ang_2m),
        "unitPseudoBumpDetected": bool(pseudo_detected),
        "unitPseudoBumpAngle": _safe_stats(ang_cm_relabel),
        "ndl1m": mean_ndl(n_1m),
        "ndl2m": mean_ndl(n_2m),
        "ndlCmRelabel": mean_ndl(n_cm_relabel),
    }


# ---------------------------------------------------------------------------
# 对“纹理制作界面已转换的那张图”做独立复核
# ---------------------------------------------------------------------------


def reference_normals_for_sample(
    sample_id: str,
    rows: int,
    cols: int,
    z_per_unit: float,
    width_world: float,
    height_world: float,
    stretch_x: float = 1.0,
    stretch_y: float = 1.0,
) -> Optional[np.ndarray]:
    """仅对带解析真值的样例给出独立参照法线；其它返回 None。

    参照用调用方声明的物理覆盖范围（width/height）乘以真实拉伸，
    与内核看到的几何完全一致；单位/尺度是否“误标”由上层自洽性检查判定，
    不在这里偷偷回退到注册表值。
    """
    from .samples import SAMPLE_REGISTRY

    entry = next((e for e in SAMPLE_REGISTRY if e["id"] == sample_id), None)
    if entry is None or "analytic" not in entry:
        return None
    a = entry["analytic"]
    eff_w = width_world * stretch_x
    eff_h = height_world * stretch_y
    if a["kind"] == "ramp":
        sf = analytic_ramp_slope(
            rows, cols, eff_w, eff_h, z_per_unit,
            coef_u=a["coefU"], coef_v=a["coefV"], amp=a["amp"],
        )
    else:
        sf = analytic_sine_slope(
            rows, cols, eff_w, eff_h, z_per_unit, fx=a["fx"], fy=a["fy"], amp=a["amp"]
        )
    return sf.normals()


def sample_unit_consistency(
    sample_id: str, width_world: float, height_world: float, stretch_x: float, stretch_y: float
) -> Optional[dict]:
    """对带真值样例，检查“世界宽高”是否与真实几何自洽。

    真实几何：纹理里写的是固定 UV 阵列，覆盖宽度是注册表里的常量。
    * widthWorld 未改而只改 stretch_x -> 真实拉伸，自洽；
    * widthWorld 改了但未相应改变表面几何 -> 单位误标，不自洽。

    返回相对误差（声明值 vs 注册表真值×拉伸）；无真值样例返回 None。
    """
    from .samples import SAMPLE_REGISTRY

    entry = next((e for e in SAMPLE_REGISTRY if e["id"] == sample_id), None)
    if entry is None or "analytic" not in entry:
        return None
    true_w = entry["widthWorld"] * stretch_x
    true_h = entry["heightWorld"] * stretch_y
    rel_w = abs(width_world - true_w) / max(true_w, 1e-12)
    rel_h = abs(height_world - true_h) / max(true_h, 1e-12)
    consistent = rel_w <= UNIT_RELABEL_TOL and rel_h <= UNIT_RELABEL_TOL
    return {
        "consistent": consistent,
        "widthRelError": rel_w,
        "heightRelError": rel_h,
        "trueWidth": true_w,
        "trueHeight": true_h,
        "tolerance": UNIT_RELABEL_TOL,
    }


def verify_normals_against_reference(
    normals: np.ndarray,
    expected: np.ndarray,
    boundary: str,
    *,
    quantized: bool = False,
) -> dict:
    """把测得法线与独立参照比角差，并给出真实材质斜光的逐像素响应差。

    quantized=True 用于“从 8-bit PNG 读回”的法线：合法的编码量化会让角度
    在平滑斜坡上偏大数度，此时用**渲染响应差（逐像素余弦）**作为主要判据，
    角差仅作参考。内核法线若与解析参照一致（quantized=False），则角差可收紧。
    """
    ang = angular_error_deg(expected, normals)
    margin = 0 if boundary == "periodic" else 2
    stats = _safe_stats(ang, margin=margin)
    ndl = render_lit(normals, GRAZING_AZIMUTH_DEG, GRAZING_ELEVATION_DEG)
    ref = render_lit(expected, GRAZING_AZIMUTH_DEG, GRAZING_ELEVATION_DEG)
    mad = float(np.mean(np.abs(ndl - ref)))
    mx = float(np.max(np.abs(ndl - ref)))

    angle_tol = ANGLE_QUANTIZED_DEG if quantized else ANGLE_INTERIOR_DEG
    # 渲染响应差判据对 8-bit 编码量化不敏感（余弦对方向误差平滑），作主要判据。
    mad_tol = RENDER_QUANTIZED_MAD if quantized else RENDER_EXACT_MAD
    max_tol = RENDER_QUANTIZED_MAX if quantized else RENDER_EXACT_MAX
    passed = bool(stats["max"] <= angle_tol and mad <= mad_tol and mx <= max_tol)
    return {
        "angle": stats,
        "angleToleranceDeg": angle_tol,
        "passed": passed,
        "grazingMeanAbsDelta": mad,
        "grazingMaxAbsDelta": mx,
        "grazingMeanTolerance": mad_tol,
        "grazingMaxTolerance": max_tol,
        "quantized": bool(quantized),
        "grazingResponse": float(np.mean(ndl)),
        "grazingReference": float(np.mean(ref)),
    }


# ---------------------------------------------------------------------------
# 跨缝斜光：正确物理标尺能跨拼缝，缺标尺的相对预览只保留可证范围
# ---------------------------------------------------------------------------


def _seam_jump_vs_neighborhood(ndl: np.ndarray, seam: int, window: int = 12, band: int = 1) -> dict:
    """接缝处相邻列亮度阶跃 vs 接缝邻近列（排除接缝本身）的阶跃。

    真实物理标尺下接缝是内部模板点，高度连续、坡度平滑，接缝阶跃应和邻近列
    同级；错误标尺造成高度台阶，接缝处出现孤立的阶跃尖峰，远大于邻近列。
    """
    diff = np.abs(np.diff(ndl, axis=1))  # (rows, cols-1)，列 j 表示 col j→j+1
    cols_d = diff.shape[1]
    lo = max(0, seam - window)
    hi = min(cols_d, seam + window)
    seam_cols = list(range(max(0, seam - band), min(cols_d, seam + band)))
    neigh_cols = [c for c in range(lo, hi) if c not in seam_cols]
    seam_jump = float(np.max(diff[:, seam_cols])) if seam_cols else 0.0
    neigh = diff[:, neigh_cols] if neigh_cols else diff
    neigh_p95 = float(np.percentile(neigh, 95)) if neigh.size else 0.0
    ratio = seam_jump / max(neigh_p95, 1e-9)
    return {"seamJump": seam_jump, "neighP95": neigh_p95, "ratio": ratio}


def verify_cross_seam_grazing(row_fraction: float = 0.5) -> dict:
    """用保留的斜光图判断哪组能跨拼缝。

    * 正确两岛标尺：材料高度连续，斜光亮度在接缝处无阶跃 -> 可跨拼缝；
    * B 标尺错一倍：接缝出现高度台阶，斜光亮度在接缝处出现阶跃 -> 不能跨；
    * 缺标尺的相对预览：只能保留“非物理”的可证范围，不得声称跨缝一致。
    """
    from .islands import IslandSpec, build_merged_height, evaluate_join
    from .kernel import normals_from_physical_height
    from .samples import island_pair_fields

    hA, hB, truth = island_pair_fields(256)
    z = float(truth["zScale"])
    a = IslandSpec(hA, 0.5, 0.5, z, label="A")

    def _merged_normals(b_spec):
        merged = build_merged_height(a, b_spec, require_scales=True)
        nres = normals_from_physical_height(merged["heightWorld"], merged["px"], merged["py"])
        return merged, nres["normals"]

    merged_ok, n_ok = _merged_normals(IslandSpec(hB, 0.5, 0.5, z, label="B"))
    merged_bad, n_bad = _merged_normals(IslandSpec(hB, 0.5, 0.5, 2.0 * z, label="B"))

    # 斜光沿 X（跨缝方向），低仰角，使接缝坡度差最明显
    ndl_ok = render_lit(n_ok, 0.0, 12.0)
    ndl_bad = render_lit(n_bad, 0.0, 12.0)
    seam = int(merged_ok["seamCol"])
    j_ok = _seam_jump_vs_neighborhood(ndl_ok, seam)
    j_bad = _seam_jump_vs_neighborhood(ndl_bad, seam)

    # 相对预览（B 无标尺）必须显式非物理，不能用于跨缝判定
    from .islands import relative_preview

    rel = relative_preview(a, IslandSpec(hB, 0.5, 0.5, None, label="B"), z)
    ndl_rel = render_lit(rel["normals"], 0.0, 12.0)
    seam_rel = int(rel["merged"]["seamCol"])
    j_rel = _seam_jump_vs_neighborhood(ndl_rel, seam_rel)

    return {
        "name": "cross_seam_grazing",
        "seamJumpCorrect": j_ok["seamJump"],
        "seamJumpWrongScale": j_bad["seamJump"],
        "correctSeamRatio": j_ok["ratio"],
        "wrongScaleSeamRatio": j_bad["ratio"],
        # 正确标尺：接缝跳变与邻近列同级；错误标尺：接缝出现孤立阶跃尖峰
        "correctCrossesSeam": j_ok["ratio"] <= 2.0,
        "wrongScaleBreaksSeam": j_bad["ratio"] >= 2.5,
        "relativePhysical": bool(rel["physical"]),
        "relativeOnlyProvableRange": (not rel["physical"]) and "临时强度" in rel["notice"],
        "relativeSeamRatio": j_rel["ratio"],
        "grazingOk": ndl_ok,
        "grazingBad": ndl_bad,
    }


# ---------------------------------------------------------------------------
# 汇总证据
# ---------------------------------------------------------------------------


def run_all_verification() -> dict:
    """一次跑完所有正反例，返回可 JSON 序列化的证据字典（测试/CLI/界面共用）。"""
    ramp = verify_clamp_ramp()
    periodic = verify_periodic_boundary()
    stretch = verify_stretch_and_mirror()
    unit = verify_unit_vs_detail()
    seam = verify_cross_seam_grazing()

    for d in (ramp, stretch):
        d.pop("normals", None)
        d.pop("normalsStretch", None)
        d.pop("normalsMirror", None)
    seam.pop("grazingOk", None)
    seam.pop("grazingBad", None)

    evidence = {
        "clampRampPassed": ramp["passed"],
        "clampRampAngleMaxDeg": ramp["angle"]["max"],
        "axisFlipDetected": ramp["axisFlipDetected"],
        "axisFlipAngleMaxDeg": ramp["axisFlipAngleMax"],
        "unitMismatchDetected": ramp["unitMismatchDetected"],
        "roundtripFidelityOk": ramp["roundtripFidelityOk"],
        "periodicInteriorAngleMaxDeg": periodic["interior"]["max"],
        "periodicBoundaryDistinct": periodic["boundaryDistinct"],
        "boundaryPolicyDeltaDeg": periodic["boundaryPolicyDeltaDeg"],
        "stretchAngleMaxDeg": stretch["stretchAngle"]["max"],
        "mirrorAngleMaxDeg": stretch["mirrorAngle"]["max"],
        "stretchHalfEnergy": stretch["stretchHalfEnergy"],
        "renderPathDelta": stretch["renderPathDelta"],
        "renderPathsAgree": stretch["renderPathsAgree"],
        "detailIsExplained": unit["detailIsExplained"],
        "unitPseudoBumpDetected": unit["unitPseudoBumpDetected"],
        "unitInvariant": unit["unitInvariant"],
        "ndl1m": unit["ndl1m"],
        "ndl2m": unit["ndl2m"],
        "ndlCmRelabel": unit["ndlCmRelabel"],
        "seamJumpCorrect": seam["seamJumpCorrect"],
        "seamJumpWrongScale": seam["seamJumpWrongScale"],
        "correctSeamRatio": seam["correctSeamRatio"],
        "wrongScaleSeamRatio": seam["wrongScaleSeamRatio"],
        "correctCrossesSeam": seam["correctCrossesSeam"],
        "wrongScaleBreaksSeam": seam["wrongScaleBreaksSeam"],
        "relativeOnlyProvableRange": seam["relativeOnlyProvableRange"],
    }
    passed = bool(
        evidence["clampRampPassed"]
        and evidence["axisFlipDetected"]
        and evidence["unitMismatchDetected"]
        and periodic["passed"]
        and stretch["stretchAngle"]["max"] <= ANGLE_INTERIOR_DEG
        and stretch["mirrorAngle"]["max"] <= ANGLE_INTERIOR_DEG
        and evidence["stretchHalfEnergy"]
        and evidence["renderPathsAgree"]
        and evidence["detailIsExplained"]
        and evidence["unitPseudoBumpDetected"]
        and evidence["unitInvariant"]
        and evidence["correctCrossesSeam"]
        and evidence["wrongScaleBreaksSeam"]
        and evidence["relativeOnlyProvableRange"]
    )
    return {
        "passed": passed,
        "evidence": evidence,
        "cases": {
            "clampRamp": ramp,
            "periodicBoundary": periodic,
            "stretchMirror": stretch,
            "unitVsDetail": unit,
            "crossSeamGrazing": seam,
        },
    }
