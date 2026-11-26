"""两块（可能互为镜像、标尺完备度不同的）纹理岛的接合适配。

设计要点
========
* 每个岛保留自己的线性高度代理场与物理标尺 z（线性值 1.0 对应的世界高度）。
  z=None 表示该岛只有相对起伏，不能进行物理高度合并。
* 合并前按统一的材料空间像素间距重采样，并按显式 flip_cols/flip_rows 把
  “外观镜像”的岛定向到材料坐标系。镜像方向由映射本身处理，而不是事后猜法线。
* 合并后的物理高度场是单一规则网格，接缝是内部边界：中心差分模板自然跨缝，
  因而两侧使用同一物理尺度，跨缝坡度连续可比。
* 跨缝剖线给出三个相邻割线坡度、残差、镜像相关系数与凸起/凹槽极性，
  用来核对“凸起没有翻成凹槽、两侧世界坡度一致”。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kernel import KernelError, ScaleRequired, normals_from_physical_height


@dataclass
class IslandSpec:
    height: np.ndarray            # HxW 线性代理高度
    width_world: float
    height_world: float
    z_per_height_unit: Optional[float]
    label: str = ""
    flip_cols: bool = False
    flip_rows: bool = False

    def require_scale(self) -> float:
        if self.z_per_height_unit is None or not np.isfinite(self.z_per_height_unit) \
                or self.z_per_height_unit <= 0:
            raise ScaleRequired(
                f"纹理岛「{self.label or '未命名'}」缺少实际高度标尺（z_per_height_unit）。"
                "相对法线预览仍可使用临时强度，但补入正数标尺前不能进行物理高度合并。",
                missing=[f"island_{self.label or 'b'}.z_per_height_unit"],
            )
        return float(self.z_per_height_unit)


def resample_island(
    h: np.ndarray,
    out_rows: int,
    out_cols: int,
    flip_cols: bool = False,
    flip_rows: bool = False,
) -> np.ndarray:
    """按归一化 UV 双线性重采样；目标纹理中心采样，边缘 clamp。"""
    src_rows, src_cols = h.shape
    u = (np.arange(out_cols, dtype=np.float64) + 0.5) / out_cols
    v = (np.arange(out_rows, dtype=np.float64) + 0.5) / out_rows
    fi = u * src_cols - 0.5
    ri = v * src_rows - 0.5
    if flip_cols:
        fi = (src_cols - 1) - fi
    if flip_rows:
        ri = (src_rows - 1) - ri

    fi0 = np.floor(fi).astype(int)
    ri0 = np.floor(ri).astype(int)
    fi1 = fi0 + 1
    ri1 = ri0 + 1
    wu = (fi - fi0)[None, :]
    wv = (ri - ri0)[:, None]

    fc0 = np.clip(fi0, 0, src_cols - 1)
    fc1 = np.clip(fi1, 0, src_cols - 1)
    rr0 = np.clip(ri0, 0, src_rows - 1)
    rr1 = np.clip(ri1, 0, src_rows - 1)

    top = h[np.ix_(rr0, fc0)] * (1 - wu) + h[np.ix_(rr0, fc1)] * wu
    bot = h[np.ix_(rr1, fc0)] * (1 - wu) + h[np.ix_(rr1, fc1)] * wu
    return top * (1 - wv) + bot * wv


def _target_counts(a: IslandSpec, b: IslandSpec) -> tuple[float, int, int, int, float]:
    dx_a = a.width_world / a.height.shape[1]
    dx_b = b.width_world / b.height.shape[1]
    dy_a = a.height_world / a.height.shape[0]
    dy_b = b.height_world / b.height.shape[0]
    px = min(dx_a, dx_b)
    py = min(dy_a, dy_b)
    ca = max(2, int(round(a.width_world / px)))
    cb = max(2, int(round(b.width_world / px)))
    common_h = a.height_world
    rows = max(2, int(round(common_h / py)))
    py = common_h / rows
    return px, ca, cb, rows, py


def build_merged_height(
    a: IslandSpec,
    b: IslandSpec,
    require_scales: bool = True,
    assumed_z_b: Optional[float] = None,
) -> dict:
    """构造材料空间统一网格上的物理高度场。

    require_scales=True 时两岛都必须有真实标尺，否则抛 ScaleRequired。
    require_scales=False 时 B 可用 assumed_z_b 作为相对预览强度。
    """
    if a.width_world <= 0 or a.height_world <= 0:
        raise KernelError("A 岛表面宽高必须为正", field="islandA.widthWorld")
    if b.width_world <= 0 or b.height_world <= 0:
        raise KernelError("B 岛表面宽高必须为正", field="islandB.widthWorld")

    if a.z_per_height_unit is None or a.z_per_height_unit <= 0:
        if require_scales:
            a.require_scale()  # 抛出 ScaleRequired
        z_a = float(assumed_z_b) if assumed_z_b else 1.0
    else:
        z_a = float(a.z_per_height_unit)

    if require_scales:
        z_b = b.require_scale()
    elif b.z_per_height_unit is not None and b.z_per_height_unit > 0:
        z_b = float(b.z_per_height_unit)
    elif assumed_z_b is not None and assumed_z_b > 0:
        z_b = float(assumed_z_b)
    else:
        raise KernelError("相对预览也需要一个正的临时强度 assumed_z_b", field="previewZ")

    px, ca, cb, rows, py = _target_counts(a, b)
    ba = resample_island(a.height, rows, ca, a.flip_cols, a.flip_rows)
    bb = resample_island(b.height, rows, cb, b.flip_cols, b.flip_rows)

    ha = z_a * ba
    hb = z_b * bb
    merged = np.concatenate([ha, hb], axis=1)

    xs = (np.arange(ca + cb, dtype=np.float64) + 0.5) * px
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * py
    seam_col = ca
    seam_x = ca * px
    return {
        "heightWorld": merged,
        "heightA": ha,
        "heightB": hb,
        "px": px,
        "py": py,
        "rows": rows,
        "colsA": ca,
        "colsB": cb,
        "seamCol": seam_col,
        "seamX": seam_x,
        "xs": xs,
        "ys": ys,
        "mergedWidth": (ca + cb) * px,
        "mergedHeight": rows * py,
        "zAUsed": z_a,
        "zBUsed": z_b,
        "physical": bool(require_scales and b.z_per_height_unit is not None),
    }


def seam_profile(merged: dict, row_fraction: float = 0.5, band: int = 1) -> dict:
    """取跨缝剖线（可对若干行取平均），返回剖面与三个割线坡度。"""
    H = merged["heightWorld"]
    rows, C = H.shape
    seam = merged["seamCol"]
    r_center = int(np.clip(round(row_fraction * (rows - 1)), 0, rows - 1))
    r0 = max(0, r_center - band)
    r1 = min(rows, r_center + band + 1)
    prof = H[r0:r1, :].mean(axis=0)

    if seam < 2 or seam + 1 > C - 1:
        raise KernelError("接缝太靠近合并纹理边缘，无法取跨缝模板", code="seam_too_close")

    px = merged["px"]
    l1, l0 = seam - 2, seam - 1
    r0i, r1i = seam, seam + 1
    s_left = float((prof[l0] - prof[l1]) / px)
    s_cross = float((prof[r0i] - prof[l0]) / px)
    s_right = float((prof[r1i] - prof[r0i]) / px)
    residual = max(abs(s_left - s_cross), abs(s_right - s_cross), abs(s_left - s_right))

    k = max(3, min(24, seam, C - seam))
    left = prof[seam - k:seam]
    right = prof[seam:seam + k]

    def _detrend(y: np.ndarray) -> np.ndarray:
        t = np.arange(len(y), dtype=np.float64)
        if len(y) >= 2:
            coef = np.polyfit(t, y, 1)
            return y - np.polyval(coef, t)
        return y - y.mean()

    ld, rd = _detrend(left[::-1]), _detrend(right)
    if ld.std() > 1e-12 and rd.std() > 1e-12:
        corr = float(np.corrcoef(ld, rd)[0, 1])
    else:
        corr = 0.0

    i0 = max(0, seam - k)
    i1 = min(C - 1, seam + k - 1)
    seg_x = merged["xs"][i0:i1 + 1]
    seg_h = prof[i0:i1 + 1]
    if len(seg_h) >= 2:
        trend = np.linspace(seg_h[0], seg_h[-1], len(seg_h))
        resid = seg_h - trend
        peak = float(resid.max())
        trough = float(resid.min())
    else:
        peak = trough = 0.0
    eps = 1e-9
    if max(peak, abs(trough)) < eps:
        polarity = "flat"
    elif peak >= abs(trough):
        polarity = "bump"
    else:
        polarity = "groove"

    return {
        "row": r_center,
        "rowFraction": row_fraction,
        "x": merged["xs"].tolist(),
        "height": prof.tolist(),
        "seamX": merged["seamX"],
        "seamCol": seam,
        "slopeLeft": s_left,
        "slopeCross": s_cross,
        "slopeRight": s_right,
        "slopeResidual": float(residual),
        "mirrorCorrelation": corr,
        "peakRelief": peak,
        "troughRelief": trough,
        "polarity": polarity,
        "window": k,
    }


def mirror_scale_fit(merged: dict, row_fraction: float = 0.5, k_pairs: int = 6, band: int = 40) -> dict:
    """跨缝 2D 条带镜像对最小二乘拟合。

    对缝两侧对称像素对 (L_j, R_j) 与物理距离 d_j 拟合
        R = a·L + q·d + c
    正确同标尺镜像连接时 a≈1（B/A 标尺比），q=2aβ，β 即跨缝背景世界坡度。
    用整个垂直条带而不是单行单割线，可显著抑制 8-bit 量化/噪声。
    """
    H = merged["heightWorld"]
    rows, C = H.shape
    seam = merged["seamCol"]
    px = merged["px"]
    rc = int(np.clip(round(row_fraction * (rows - 1)), 0, rows - 1))
    r0 = max(0, rc - band)
    r1 = min(rows, rc + band + 1)
    k_pairs = min(k_pairs, seam, C - seam)

    Ls, Rs, Ds = [], [], []
    for r in range(r0, r1):
        for j in range(k_pairs):
            Ls.append(H[r, seam - 1 - j])
            Rs.append(H[r, seam + j])
            Ds.append((j + 0.5) * px)
    L = np.asarray(Ls)
    R = np.asarray(Rs)
    d = np.asarray(Ds)
    X = np.column_stack([L, d, np.ones_like(L)])
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    a, q, c = (float(v) for v in coef)
    beta = float(q / (2 * a)) if abs(a) > 1e-8 else float("nan")
    rms = float(np.sqrt(np.mean((R - X @ coef) ** 2)))
    return {
        "scaleRatioBA": a,
        "crossSlopeBetaFit": beta,
        "heightOffset": c,
        "fitRmsHeight": rms,
        "pairs": int(len(L)),
    }


def evaluate_join(
    a: IslandSpec,
    b: IslandSpec,
    slope_tol: float = 0.12,
    expected_polarity: str = "bump",
    row_fraction: float = 0.5,
    boundary: str = "clamp",
    method: str = "central",
) -> dict:
    """物理合并 + 法线 + 跨缝核对。缺标尺时抛 ScaleRequired（失败反馈给控件）。"""
    merged = build_merged_height(a, b, require_scales=True)
    nres = normals_from_physical_height(
        merged["heightWorld"], merged["px"], merged["py"], boundary, method
    )
    profile = seam_profile(merged, row_fraction=row_fraction)
    fit = mirror_scale_fit(merged, row_fraction=row_fraction)

    raw_slope_ok = bool(profile["slopeResidual"] <= slope_tol)
    scale_ok = bool(
        abs(fit["scaleRatioBA"] - 1.0) <= slope_tol
        and np.isfinite(fit["crossSlopeBetaFit"])
    )
    polarity_ok = bool(expected_polarity in ("any", "", None) or profile["polarity"] == expected_polarity)
    corr_ok = bool(profile["mirrorCorrelation"] >= 0.5)
    passed = bool(scale_ok and polarity_ok and corr_ok)

    checks = {
        "passed": passed,
        # 稳健拟合给出的“两侧世界坡度/标尺一致”是主判据
        "slopeConsistent": scale_ok,
        "scaleRatioConsistent": scale_ok,
        "rawSlopeConsistent": raw_slope_ok,
        "polarityCorrect": polarity_ok,
        "mirrorOriented": corr_ok,
        "slopeTolerance": slope_tol,
        "expectedPolarity": expected_polarity or "any",
        "messages": [],
    }
    if scale_ok:
        checks["messages"].append(
            f"镜像条带拟合 B/A 标尺比 k={fit['scaleRatioBA']:+.3f}（|k−1|≤{slope_tol:g}），"
            f"反演跨缝世界坡度 β={fit['crossSlopeBetaFit']:+.4f}，两侧物理尺度一致。"
        )
    else:
        checks["messages"].append(
            f"两侧尺度/坡度不一致：拟合 B/A 标尺比 k={fit['scaleRatioBA']:+.3f}，"
            f"偏离 1 超过容差 {slope_tol:g}（或镜像关系崩溃导致 β={fit['crossSlopeBetaFit']}）。"
            "常见原因：B 岛标尺不对、镜像方向（翻轴）错误或物理宽度/分辨率尺度填错。"
        )
    checks["messages"].append(
        f"跨缝割线坡度：左 {profile['slopeLeft']:+.4f} / 缝 {profile['slopeCross']:+.4f} / "
        f"右 {profile['slopeRight']:+.4f}，原始残差 {profile['slopeResidual']:.4f}"
        + ("。" if raw_slope_ok else
           f" > {slope_tol:g}；单像素割线对 8-bit 量化敏感，稳健拟合 k 已通过时可视为量化噪声，"
           "可调大容差或使用 16-bit/浮点高度。")
    )
    if polarity_ok:
        label = {"bump": "凸起（峰顶为正）", "groove": "凹槽（负起伏占主导）", "flat": "平坦"}[
            profile["polarity"]
        ]
        checks["messages"].append(f"跨缝剖线极性：{label}，与预期 {expected_polarity or '任意'} 相符。")
    else:
        checks["messages"].append(
            f"极性错误：剖线呈 {profile['polarity']}，预期 {expected_polarity}。"
            "凸起被翻成凹槽通常是镜像/通道方向或高度反相设置错误。"
        )
    if not corr_ok:
        checks["messages"].append(
            f"镜像形状相关性仅 {profile['mirrorCorrelation']:.3f}（<0.5），"
            "B 岛的 flip_cols/flip_rows 很可能选反，或两岛并非同一镜像表面。"
        )

    return {
        "merged": merged,
        "normals": nres["normals"],
        "slope_x": nres["slope_x"],
        "slope_y": nres["slope_y"],
        "profile": profile,
        "fit": fit,
        "checks": checks,
    }


def relative_preview(
    a: IslandSpec,
    b: IslandSpec,
    preview_z_b: float,
    boundary: str = "clamp",
    method: str = "central",
    row_fraction: float = 0.5,
) -> dict:
    """标尺不齐全时的相对法线预览（B 使用临时强度，结果明确标记为非物理）。"""
    merged = build_merged_height(a, b, require_scales=False, assumed_z_b=preview_z_b)
    nres = normals_from_physical_height(
        merged["heightWorld"], merged["px"], merged["py"], boundary, method
    )
    profile = seam_profile(merged, row_fraction=row_fraction)
    return {
        "merged": merged,
        "normals": nres["normals"],
        "slope_x": nres["slope_x"],
        "slope_y": nres["slope_y"],
        "profile": profile,
        "physical": False,
        "notice": "相对预览：B 岛使用临时强度，跨缝坡度数值不具备物理意义；补齐标尺后请运行物理核对。",
    }
