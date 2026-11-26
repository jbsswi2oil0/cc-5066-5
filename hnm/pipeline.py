"""请求参数 -> 内核调用 -> JSON 可序列化结果（含 PNG data URL）。

server 与 cli 共用这一层；所有可预期失败都抛 KernelError/ScaleRequired，
由边界层转成结构化错误反馈给界面控件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import samples as sample_mod
from .conventions import convention_metadata, resolve_signs
from .images import load_rgba, load_rgba_bytes, decode_data_url, png_data_url
from .islands import IslandSpec, evaluate_join, relative_preview
from .kernel import (
    KernelError,
    ScaleRequired,
    SurfaceScale,
    decode_normal_rgb,
    encoding_error_metrics,
    height_to_normals,
    slope_strength_stats,
)
from .lighting import shade
from .transfer import extract_linear_height


def _num(params: dict, key: str, default: Optional[float] = None) -> float:
    if key not in params or params[key] is None or params[key] == "":
        if default is not None:
            return default
        raise KernelError(f"缺少参数 {key}", field=key, code="missing_parameter")
    try:
        return float(params[key])
    except (TypeError, ValueError):
        raise KernelError(f"参数 {key} 不是数字: {params[key]!r}", field=key)


def load_source_rgba(sample_id: Optional[str], image_data_url: Optional[str]) -> np.ndarray:
    if sample_id:
        for entry in sample_mod.SAMPLE_REGISTRY + list(sample_mod.ISLAND_REGISTRY.values()):
            if entry["id"] == sample_id:
                path = sample_mod.ASSET_DIR / entry["file"]
                if not path.exists():
                    raise KernelError(f"样例资产尚未生成: {path.name}，请先运行 python -m hnm.cli assets")
                return load_rgba(path)
        raise KernelError(f"未知样例: {sample_id}", field="sampleId")
    if image_data_url:
        try:
            return load_rgba_bytes(decode_data_url(image_data_url))
        except KernelError:
            raise
        except Exception as exc:  # Pillow 解码失败等
            raise KernelError(f"无法解码上传图像: {exc}", field="image", code="bad_image")
    raise KernelError("必须提供 sampleId 或 image", field="image", code="missing_image")


def list_samples() -> dict:
    return {
        "samples": sample_mod.SAMPLE_REGISTRY,
        "islands": sample_mod.ISLAND_REGISTRY,
        "islandTruth": _truth_if_exists(),
        "canonicalFrame": (
            "规范切线系：+X 朝画面右（列+），+Y 朝画面上（行-），+Z 朝表面外；右手系。"
        ),
    }


def _truth_if_exists() -> Optional[dict]:
    try:
        _, _, truth = sample_mod.island_pair_fields(256)
        return truth
    except Exception:
        return None


def convert_single(params: dict) -> dict:
    rgba = load_source_rgba(params.get("sampleId"), params.get("image"))
    h = extract_linear_height(
        rgba,
        channel=params.get("channel", "r"),
        transfer=params.get("transfer", "linear"),
        gamma=_num(params, "gamma", 2.2),
        invert=bool(params.get("invert", False)),
    )
    rows, cols = h.shape

    scale = SurfaceScale(
        width_world=_num(params, "widthWorld", 1.0),
        height_world=_num(params, "heightWorld", 1.0),
        z_per_height_unit=_num(params, "zScale", 0.05),
        unit=params.get("unit", "m") or "m",
        stretch_x=_num(params, "stretchX", 1.0),
        stretch_y=_num(params, "stretchY", 1.0),
    )
    boundary = params.get("boundary", "clamp")
    method = params.get("method", "central")
    result = height_to_normals(h, scale, boundary=boundary, method=method)
    n = result["normals"]

    convention_id = params.get("convention", "opengl")
    fx = bool(params.get("flipX", False))
    fy = bool(params.get("flipY", False))
    fz = bool(params.get("flipZ", False))
    signs = resolve_signs(convention_id, fx, fy, fz)
    err = encoding_error_metrics(n, signs)
    normal_png = err.pop("encoded")

    light = params.get("light") or {}
    decoded = decode_normal_rgb(normal_png, signs, True)
    shaded = shade(
        decoded,
        azimuth_deg=float(light.get("azimuth", 315.0)),
        elevation_deg=float(light.get("elevation", 45.0)),
        ambient=float(light.get("ambient", 0.22)),
        specular=float(light.get("specular", 0.25)),
    )

    return {
        "ok": True,
        "width": cols,
        "height": rows,
        "normalPng": png_data_url(normal_png),
        "shadedPng": png_data_url(shaded),
        "convention": convention_metadata(convention_id, fx, fy, fz),
        "scale": result["scaleInfo"],
        "stats": slope_strength_stats(result),
        "encoding": err,
        "heightPreviewPng": png_data_url(_height_to_gray(h)),
    }


def _height_to_gray(h: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(h)), float(np.max(h))
    g = np.zeros_like(h) if hi - lo < 1e-12 else (h - lo) / (hi - lo)
    return np.uint8(np.clip(g, 0, 1) * 255 + 0.5)


def _island_from_params(spec: dict, fallback_id: str) -> IslandSpec:
    rgba = load_source_rgba(spec.get("sampleId") or fallback_id, spec.get("image"))
    h = extract_linear_height(
        rgba,
        channel=spec.get("channel", "r"),
        transfer=spec.get("transfer", "linear"),
        gamma=float(spec.get("gamma", 2.2) or 2.2),
        invert=bool(spec.get("invert", False)),
    )
    z_raw = spec.get("zScale", None)
    if z_raw is None or z_raw == "":
        z = None
    else:
        try:
            z = float(z_raw)
        except (TypeError, ValueError):
            raise KernelError(f"高度标尺不是数字: {z_raw!r}", field=spec.get("zField", "zScale"))
        if z <= 0:
            raise KernelError("高度标尺必须为正数", field=spec.get("zField", "zScale"))
        z = float(z)
    return IslandSpec(
        height=h,
        width_world=float(spec.get("widthWorld", 0.5)),
        height_world=float(spec.get("heightWorld", 0.5)),
        z_per_height_unit=z,
        label=spec.get("label", ""),
        flip_cols=bool(spec.get("flipCols", False)),
        flip_rows=bool(spec.get("flipRows", False)),
    )


def join_islands(params: dict) -> dict:
    a_spec = dict(params.get("islandA") or {})
    b_spec = dict(params.get("islandB") or {})
    a_spec.setdefault("sampleId", "island_A_calibrated")
    b_spec.setdefault("sampleId", "island_B_relative")
    a_spec["label"] = "A"
    b_spec["label"] = "B"
    b_spec["zField"] = "islandB.zScale"

    a = _island_from_params(a_spec, "island_A_calibrated")
    b = _island_from_params(b_spec, "island_B_relative")

    boundary = params.get("boundary", "clamp")
    method = params.get("method", "central")
    row_fraction = float(params.get("rowFraction", 0.5))
    convention_id = params.get("convention", "opengl")
    signs = resolve_signs(
        convention_id,
        bool(params.get("flipX", False)),
        bool(params.get("flipY", False)),
        bool(params.get("flipZ", False)),
    )

    mode = params.get("mode", "physical")
    if mode == "relative":
        preview_z = _num(params, "previewZ", a.z_per_height_unit or 0.05)
        res = relative_preview(a, b, preview_z, boundary, method, row_fraction)
        n = res["normals"]
        payload = {
            "ok": True,
            "mode": "relative",
            "physical": False,
            "notice": res["notice"],
            "profile": _json_profile(res["profile"]),
        }
    else:
        slope_tol = float(params.get("slopeTol", 0.12))
        expected = params.get("expectedPolarity", "bump")
        res = evaluate_join(
            a, b,
            slope_tol=slope_tol,
            expected_polarity=expected,
            row_fraction=row_fraction,
            boundary=boundary,
            method=method,
        )
        n = res["normals"]
        payload = {
            "ok": True,
            "mode": "physical",
            "physical": True,
            "checks": res["checks"],
            "profile": _json_profile(res["profile"]),
            "fit": res["fit"],
        }

    err = encoding_error_metrics(n, signs)
    normal_png = err.pop("encoded")
    merged = res["merged"]
    light = params.get("light") or {}
    decoded = decode_normal_rgb(normal_png, signs, True)
    shaded = shade(
        decoded,
        azimuth_deg=float(light.get("azimuth", 315.0)),
        elevation_deg=float(light.get("elevation", 45.0)),
        ambient=float(light.get("ambient", 0.22)),
        specular=float(light.get("specular", 0.25)),
    )
    payload.update({
        "width": normal_png.shape[1],
        "height": normal_png.shape[0],
        "normalPng": png_data_url(normal_png),
        "shadedPng": png_data_url(shaded),
        "heightPng": png_data_url(_height_to_gray(merged["heightWorld"])),
        "convention": convention_metadata(
            convention_id,
            bool(params.get("flipX", False)),
            bool(params.get("flipY", False)),
            bool(params.get("flipZ", False)),
        ),
        "scaleInfo": {
            "px": merged["px"], "py": merged["py"],
            "mergedWidth": merged["mergedWidth"], "mergedHeight": merged["mergedHeight"],
            "colsA": merged["colsA"], "colsB": merged["colsB"],
            "zAUsed": merged["zAUsed"], "zBUsed": merged["zBUsed"],
        },
        "encoding": err,
        "stats": slope_strength_stats(
            {"normals": n, "slope_x": res["slope_x"], "slope_y": res["slope_y"]}
        ),
    })
    return payload


def verify_scaling(params: dict) -> dict:
    """对一次真实“高度→法线”转换做独立尺度一致性复核。

    返回内容：
      * measured normal PNG（由内核编码，UI 与真实渲染都用它）；
      * grazingPng：用**从 PN G 读回的法线**参与侧光着色（不是重算法线）；
      * angle / 渲染响应的独立参照结果（reference 来自解析式，非内核中间量）；
      * 两条坡度写法（切线基 vs 真实长度）的角差与渲染差；
      * referenceAvailable=False 时只给出“可证范围”，不宣称绝对正确。

    该入口供纹理制作界面的“尺度一致性验收”按钮调用；命令行仅做辅助。
    """
    from . import normal_validation as nv

    rgba = load_source_rgba(params.get("sampleId"), params.get("image"))
    h = extract_linear_height(
        rgba,
        channel=params.get("channel", "r"),
        transfer=params.get("transfer", "linear"),
        gamma=_num(params, "gamma", 2.2),
        invert=bool(params.get("invert", False)),
    )
    rows, cols = h.shape
    scale = SurfaceScale(
        width_world=_num(params, "widthWorld", 1.0),
        height_world=_num(params, "heightWorld", 1.0),
        z_per_height_unit=_num(params, "zScale", 0.05),
        unit=params.get("unit", "m") or "m",
        stretch_x=_num(params, "stretchX", 1.0),
        stretch_y=_num(params, "stretchY", 1.0),
    )
    boundary = params.get("boundary", "clamp")
    method = params.get("method", "central")
    result = height_to_normals(h, scale, boundary=boundary, method=method)
    n_kernel = result["normals"]

    convention_id = params.get("convention", "opengl")
    fx = bool(params.get("flipX", False))
    fy = bool(params.get("flipY", False))
    fz = bool(params.get("flipZ", False))
    signs = resolve_signs(convention_id, fx, fy, fz)

    # 编码成真实纹素，再从纹素解码 = “测得法线”。UI 与侧光都只用这一份。
    err = encoding_error_metrics(n_kernel, signs)
    normal_png = err.pop("encoded")
    measured = decode_normal_rgb(normal_png, signs, True)

    light = params.get("light") or {}
    grazing = nv.render_lit(
        measured,
        azimuth_deg=float(light.get("azimuth", 150.0)),
        elevation_deg=float(light.get("elevation", 15.0)),
    )
    grazing_png = np.uint8(np.clip(grazing, 0.0, 1.0) * 255.0 + 0.5)
    # 侧光着色预览（真实材质：灰底乘光照响应）
    base = np.array([0.72, 0.74, 0.78])
    lit_rgb = np.uint8(np.clip(grazing[:, :, None] * base.reshape(1, 1, 3), 0, 1) * 255 + 0.5)

    # 两条坡度写法（只在可重复网格上并列，拉伸/单位改变时区分谁动尺度）
    dx, dy = result["dx"], result["dy"]
    sx_tan, sy_tan = nv.slope_in_tangent_basis(h, dx, dy, scale.z_per_height_unit, boundary)
    sx_len, sy_len = nv.slope_by_true_length(h, dx, dy, scale.z_per_height_unit, boundary)
    from .kernel import normals_from_slopes

    n_tan = normals_from_slopes(sx_tan, sy_tan)
    n_len = normals_from_slopes(sx_len, sy_len)
    grazing_tan = nv.render_lit(n_tan, float(light.get("azimuth", 150.0)), float(light.get("elevation", 15.0)))
    grazing_len = nv.render_lit(n_len, float(light.get("azimuth", 150.0)), float(light.get("elevation", 15.0)))
    path_render_delta = float(np.mean(np.abs(grazing_tan - grazing_len)))

    # 独立参照（解析式）；与内核中间量无关。
    # 用调用方声明的世界宽高×真实拉伸，与内核所见的几何一致；
    # 单位/尺度是否误标由 unitConsistency 单独判定（见下）。
    expected = nv.reference_normals_for_sample(
        params.get("sampleId") or "", rows, cols,
        scale.z_per_height_unit,
        scale.width_world, scale.height_world,
        stretch_x=scale.stretch_x, stretch_y=scale.stretch_y,
    )
    reference_available = expected is not None
    unit_consistency = nv.sample_unit_consistency(
        params.get("sampleId") or "",
        scale.width_world, scale.height_world, scale.stretch_x, scale.stretch_y,
    )
    if reference_available:
        verdict = nv.verify_normals_against_reference(
            measured, expected, boundary, quantized=True
        )
        # 与注册表真值的自洽性仅作信息提示：重设覆盖范围/单位缩放在真实几何下
        # 仍然合法，是否“伪凹凸”由法线对真实表面的角差与渲染响应决定。
        verdict["unitMismatch"] = unit_consistency
    else:
        # 旧资源无解析真值：只能给出可证范围（编码往返保真 + 两写法一致），
        # 不宣称绝对法线正确。
        roundtrip = err.get("maxAngularErrorDeg", 0.0)
        verdict = {
            "passed": bool(roundtrip < 1.0 and path_render_delta < 1e-6),
            "angle": None,
            "grazingMeanAbsDelta": None,
            "angleToleranceDeg": None,
            "quantized": True,
            "note": "无解析真值：仅证明编码往返保真与两条坡度写法一致，不能证明绝对法线正确。",
        }

    return {
        "ok": True,
        "width": cols,
        "height": rows,
        "normalPng": png_data_url(normal_png),
        "measuredLitPng": png_data_url(lit_rgb),
        "convention": convention_metadata(convention_id, fx, fy, fz),
        "scale": result["scaleInfo"],
        "stats": slope_strength_stats(result),
        "encoding": err,
        "referenceAvailable": bool(reference_available),
        "verdict": verdict,
        "slopePaths": {
            "tangentVsTrueLengthRenderMeanAbsDelta": path_render_delta,
            "agree": bool(path_render_delta < 1e-6),
        },
        "grazing": light,
    }


def _json_profile(p: dict) -> dict:
    # 全量剖面可能 512 点，直接给前端；保留数值精度
    return {k: v for k, v in p.items()}
