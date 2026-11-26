"""生成内置资产、交付贴图/预览与核对报告。

用法：
  python -m hnm.cli assets     # 只生成 assets/source 源贴图
  python -m hnm.cli all        # 资产 + outputs 全部交付物（默认）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .charts import draw_profile_chart
from .conventions import resolve_signs
from .images import load_rgba, save_png
from .islands import IslandSpec, evaluate_join
from .kernel import (
    SurfaceScale,
    angular_error_deg,
    decode_normal_rgb,
    encode_normal_rgb,
    encoding_error_metrics,
    height_to_normals,
)
from .lighting import shade
from .samples import (
    ASSET_DIR,
    OUTPUT_DIR,
    SAMPLE_REGISTRY,
    build_all_assets,
    island_pair_fields,
)
from .transfer import extract_linear_height


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _save_normal_pair(n, stem: str, sub: str = "normals") -> dict:
    out_dir = OUTPUT_DIR / sub
    paths = {}
    for conv in ("opengl", "directx"):
        signs = resolve_signs(conv)
        rgb = encode_normal_rgb(n, signs)
        p = out_dir / f"{stem}_{conv}.png"
        save_png(rgb, p)
        paths[conv] = str(p.relative_to(OUTPUT_DIR.parent))
    return paths


def generate_sample_outputs(report: dict) -> None:
    sample_reports = {}
    for entry in SAMPLE_REGISTRY:
        rgba = load_rgba(ASSET_DIR / entry["file"])
        h = extract_linear_height(
            rgba,
            channel=entry.get("channel", "r"),
            transfer=entry.get("transfer", "linear"),
        )
        scale = SurfaceScale(
            entry["widthWorld"], entry["heightWorld"], entry["zScale"], entry.get("unit", "m")
        )
        res = height_to_normals(h, scale, boundary=entry.get("boundary", "clamp"))
        n = res["normals"]

        paths = _save_normal_pair(n, entry["id"])
        signs = resolve_signs("opengl")
        metrics = encoding_error_metrics(n, signs)
        metrics.pop("encoded")

        decoded = decode_normal_rgb(encode_normal_rgb(n, signs), signs, True)
        lit = shade(decoded, azimuth_deg=315, elevation_deg=42)
        lit_path = OUTPUT_DIR / "previews" / f"{entry['id']}_lit315.png"
        save_png(lit, lit_path)

        # 第二方向光，便于核对光照方向
        lit2 = shade(decoded, azimuth_deg=135, elevation_deg=42)
        save_png(lit2, OUTPUT_DIR / "previews" / f"{entry['id']}_lit135.png")

        sample_reports[entry["id"]] = {
            "label": entry["label"],
            "source": f"assets/source/{entry['file']}",
            "normals": paths,
            "litPreview": str(lit_path.relative_to(OUTPUT_DIR.parent)),
            "scale": res["scaleInfo"],
            "encodingErrorOpenGL": metrics,
            "description": entry["description"],
        }

        if entry["id"] == "boundary_step":
            res_p = height_to_normals(h, scale, boundary="periodic")
            save_png(
                encode_normal_rgb(res_p["normals"], signs),
                OUTPUT_DIR / "normals" / "boundary_step_periodic_opengl.png",
            )
            save_png(
                encode_normal_rgb(n, signs),
                OUTPUT_DIR / "normals" / "boundary_step_clamp_opengl.png",
            )

        if entry["id"] == "srgb_encoded_ramp":
            h_naive = extract_linear_height(rgba, "r", "linear")
            res_naive = height_to_normals(h_naive, scale)
            save_png(
                encode_normal_rgb(res_naive["normals"], signs),
                OUTPUT_DIR / "normals" / "srgb_ramp_naive_no_decode_opengl.png",
            )
            sample_reports[entry["id"]]["transferWarning"] = (
                "已额外输出 srgb_ramp_naive_no_decode_opengl.png 作为错误示范："
                "把 sRGB 显示编码值直接当线性高度，暗部梯度被夸大。"
            )
    report["samples"] = sample_reports


def analytic_self_check(report: dict) -> None:
    checks = {}

    n = 128
    u = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, u)
    h = 0.2 + 0.4 * U - 0.3 * V
    sc = SurfaceScale(2.0, 1.0, 0.25)
    res = height_to_normals(h, sc)
    dx, dy = 2.0 / n, 1.0 / n
    sx = 0.25 * (0.4 / n) / dx
    sy = -0.25 * (-0.3 / n) / dy
    exp = np.array([-sx, -sy, 1.0])
    exp /= np.linalg.norm(exp)
    checks["rampMaxComponentError"] = float(np.abs(res["normals"][4:-4, 4:-4] - exp).max())

    hsin = 0.5 + 0.28 * np.sin(6 * np.pi * U) * np.cos(4 * np.pi * V)
    res64 = height_to_normals(hsin[::2, ::2], SurfaceScale(1, 1, 0.108), boundary="periodic")
    res128 = height_to_normals(hsin, SurfaceScale(1, 1, 0.108), boundary="periodic")

    def sine_expected(nn):
        uu = (np.arange(nn) + 0.5) / nn
        UU, VV = np.meshgrid(uu, uu)
        sx = 0.108 * 0.28 * 6 * np.pi * np.cos(6 * np.pi * UU) * np.cos(4 * np.pi * VV)
        sy = 0.108 * 0.28 * 4 * np.pi * np.sin(6 * np.pi * UU) * np.sin(4 * np.pi * VV)
        e = np.stack([-sx, -sy, np.ones_like(sx)], 2)
        return e / np.linalg.norm(e, axis=2, keepdims=True)

    e64 = float(np.max(angular_error_deg(res64["normals"], sine_expected(64))))
    e128 = float(np.max(angular_error_deg(res128["normals"], sine_expected(128))))
    checks["sineMaxAngularErrorDeg_n64"] = e64
    checks["sineMaxAngularErrorDeg_n128"] = e128
    checks["secondOrderRatio"] = e128 / e64

    m = encoding_error_metrics(res128["normals"], resolve_signs("opengl"))
    checks["encodeMaxAngularErrorDeg"] = m["maxAngularErrorDeg"]
    checks["encodeMeanAngularErrorDeg"] = m["meanAngularErrorDeg"]

    # 分辨率不变性
    strength = {}
    for nn in (64, 128, 256):
        uu = (np.arange(nn) + 0.5) / nn
        UU, _ = np.meshgrid(uu, uu)
        hh = 0.5 + 0.3 * np.sin(4 * np.pi * UU)
        rr = height_to_normals(hh, SurfaceScale(1, 1, 0.1), boundary="periodic")
        strength[nn] = float(np.mean(np.abs(rr["slope_x"])))
    checks["resolutionMeanSlope"] = strength
    checks["resolutionRatio256over64"] = strength[256] / strength[64]

    report["analyticChecks"] = checks


def generate_island_outputs(report: dict) -> None:
    hA, hB, truth = island_pair_fields(256)
    z = truth["zScale"]

    # 相对预览（B 无标尺，借用 A 强度）
    a = IslandSpec(hA, 0.5, 0.5, z, label="A")
    b_rel = IslandSpec(hB, 0.5, 0.5, None, label="B")
    from .islands import relative_preview

    rel = relative_preview(a, b_rel, z)
    signs = resolve_signs("opengl")
    save_png(encode_normal_rgb(rel["normals"], signs), OUTPUT_DIR / "islands" / "relative_merge_normal.png")

    # 缺标尺的结构化失败（界面 422 同样结构）
    failure_record = None
    try:
        evaluate_join(a, b_rel, slope_tol=0.12)
    except Exception as exc:
        failure_record = {
            "error": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": exc.message if hasattr(exc, "message") else str(exc),
            "missing": getattr(exc, "missing", []),
        }

    # 物理合并：正确标尺
    b = IslandSpec(hB, 0.5, 0.5, z, label="B")
    ok = evaluate_join(a, b, slope_tol=0.12)
    merged = ok["merged"]
    save_png(_gray_world(merged["heightWorld"]), OUTPUT_DIR / "islands" / "merged_height_physical.png")
    paths = _save_normal_pair(ok["normals"], "merged_mirrored_islands", sub="islands")
    decoded = decode_normal_rgb(
        encode_normal_rgb(ok["normals"], signs), signs, True
    )
    save_png(shade(decoded, 315, 42), OUTPUT_DIR / "islands" / "merged_lit315.png")
    save_png(shade(decoded, 135, 42), OUTPUT_DIR / "islands" / "merged_lit135.png")
    chart = draw_profile_chart(ok["profile"], ok["checks"])
    save_png(chart, OUTPUT_DIR / "islands" / "seam_profile.png")

    # 错误标尺 / 错误镜像：记录为反例
    bad_scale = evaluate_join(a, IslandSpec(hB, 0.5, 0.5, 2 * z, label="B"), slope_tol=0.12)
    bad_flip = evaluate_join(
        a, IslandSpec(hB, 0.5, 0.5, z, label="B", flip_cols=True), slope_tol=0.12
    )
    save_png(
        draw_profile_chart(bad_scale["profile"], bad_scale["checks"]),
        OUTPUT_DIR / "islands" / "seam_profile_wrong_scale.png",
    )
    save_png(
        draw_profile_chart(bad_flip["profile"], bad_flip["checks"]),
        OUTPUT_DIR / "islands" / "seam_profile_wrong_flip.png",
    )

    report["islands"] = {
        "truth": truth,
        "scaleRequiredFailure": failure_record,
        "correctJoin": {
            "normalMaps": paths,
            "heightMap": "outputs/islands/merged_height_physical.png",
            "litPreviews": [
                "outputs/islands/merged_lit315.png",
                "outputs/islands/merged_lit135.png",
            ],
            "profileChart": "outputs/islands/seam_profile.png",
            "profile": ok["profile"],
            "fit": ok["fit"],
            "checks": ok["checks"],
        },
        "wrongScale2x": {
            "passed": bad_scale["checks"]["passed"],
            "slopeResidual": bad_scale["profile"]["slopeResidual"],
            "chart": "outputs/islands/seam_profile_wrong_scale.png",
        },
        "wrongFlip": {
            "passed": bad_flip["checks"]["passed"],
            "slopeResidual": bad_flip["profile"]["slopeResidual"],
            "chart": "outputs/islands/seam_profile_wrong_flip.png",
        },
        "relativePreview": "outputs/islands/relative_merge_normal.png",
    }


def _gray_world(h: np.ndarray) -> np.ndarray:
    lo, hi = float(h.min()), float(h.max())
    g = (h - lo) / max(hi - lo, 1e-12)
    return np.uint8(g * 255 + 0.5)


def main(argv=None) -> None:
    import sys

    args = list(argv) if argv is not None else sys.argv[1:]
    cmd = args[0] if args else "all"
    if cmd in ("assets", "all"):
        info = build_all_assets()
        print(f"[cli] 源资产 {len(info['written'])} 个 -> {ASSET_DIR}")
    if cmd == "assets":
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"version": 1}
    analytic_self_check(report)
    generate_sample_outputs(report)
    generate_island_outputs(report)

    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"[cli] 报告 -> {report_path}")
    ac = report["analyticChecks"]
    print(
        "[cli] 解析核对: rampErr=%.2e sineErr128=%.4f° encodeMax=%.3f° res256/64=%.5f"
        % (
            ac["rampMaxComponentError"],
            ac["sineMaxAngularErrorDeg_n128"],
            ac["encodeMaxAngularErrorDeg"],
            ac["resolutionRatio256over64"],
        )
    )
    ck = report["islands"]["correctJoin"]["checks"]
    print(f"[cli] 双岛物理合并: passed={ck['passed']}")


if __name__ == "__main__":
    main()
