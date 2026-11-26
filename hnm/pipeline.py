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


def _json_profile(p: dict) -> dict:
    # 全量剖面可能 512 点，直接给前端；保留数值精度
    return {k: v for k, v in p.items()}
