"""用 Pillow 绘制跨缝高度剖线图，供 CLI 交付与报告使用。

服务器环境通常没有 CJK 字体，因此位图内文字统一用英文；
Web 界面里的同名图表由 Canvas 绘制，可正常显示中文。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def draw_profile_chart(
    profile: dict,
    checks: Optional[dict] = None,
    width: int = 1040,
    height: int = 400,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), (18, 22, 30))
    d = ImageDraw.Draw(img)
    f12 = _font(13)
    f14b = _font(15, bold=True)
    f20b = _font(22, bold=True)

    ml, mr, mt, mb = 64, 286, 52, 48
    pw, ph = width - ml - mr, height - mt - mb

    xs = np.asarray(profile["x"], dtype=float)
    hs = np.asarray(profile["height"], dtype=float)
    seam_x = float(profile["seamX"])

    xmin, xmax = float(xs.min()), float(xs.max())
    hmin, hmax = float(hs.min()), float(hs.max())
    pad_h = max((hmax - hmin) * 0.18, 1e-6)
    hmin -= pad_h
    hmax += pad_h

    def X(x: float) -> int:
        return int(ml + (x - xmin) / max(xmax - xmin, 1e-12) * pw)

    def Y(h: float) -> int:
        return int(mt + (hmax - h) / max(hmax - hmin, 1e-12) * ph)

    for i in range(5):
        gy = mt + int(ph * i / 4)
        d.line([(ml, gy), (ml + pw, gy)], fill=(34, 42, 55), width=1)
        val = hmax - (hmax - hmin) * i / 4
        d.text((8, gy - 7), f"{val:+.4f}", fill=(130, 145, 165), font=f12)

    zero_y = Y(0.0)
    if mt <= zero_y <= mt + ph:
        d.line([(ml, zero_y), (ml + pw, zero_y)], fill=(70, 84, 104), width=1)

    sx = X(seam_x)
    d.line([(sx, mt), (sx, mt + ph)], fill=(230, 180, 70), width=2)
    seam_label = "seam"
    tb = d.textbbox((0, 0), seam_label, font=f14b)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    lx = min(max(sx + 6, ml + 4), ml + pw - tw - 12)
    d.rounded_rectangle(
        [lx - 5, mt + 6, lx + tw + 7, mt + 8 + th + 4], radius=4, fill=(30, 36, 48)
    )
    d.text((lx + 1, mt + 8), seam_label, fill=(230, 190, 90), font=f14b)

    pts = [(X(x), Y(h)) for x, h in zip(xs, hs)]
    if len(pts) >= 2:
        d.polygon([(pts[0][0], mt + ph), *pts, (pts[-1][0], mt + ph)], fill=(28, 66, 92))
        d.line(pts, fill=(90, 200, 255), width=2)

    seam_col = int(profile["seamCol"])
    for idx, color, name, where in (
        (seam_col - 1, (255, 120, 120), "L0", "tl"),
        (seam_col, (120, 255, 160), "R0", "br"),
    ):
        if 0 <= idx < len(xs):
            pxp, pyp = X(xs[idx]), Y(hs[idx])
            d.ellipse([pxp - 5, pyp - 5, pxp + 5, pyp + 5], outline=color, width=2)
            if where == "tl":
                d.text((pxp - 9, pyp - 12), name, fill=color, font=f12, anchor="rs")
            else:
                d.text((pxp + 9, pyp + 6), name, fill=color, font=f12, anchor="la")

    d.text((ml, height - 30), "material X (world units)", fill=(170, 182, 200), font=f12)
    d.text((ml, 16), "Cross-seam height profile H(X)", fill=(225, 232, 242), font=f20b)

    panel_x = ml + pw + 24
    d.line([(panel_x - 12, mt), (panel_x - 12, mt + ph)], fill=(40, 50, 66), width=1)
    lines = [
        ("slope left", f"{profile['slopeLeft']:+.4f}"),
        ("slope cross", f"{profile['slopeCross']:+.4f}"),
        ("slope right", f"{profile['slopeRight']:+.4f}"),
        ("residual", f"{profile['slopeResidual']:.4f}"),
        ("mirror corr", f"{profile['mirrorCorrelation']:+.3f}"),
        ("polarity", profile["polarity"]),
        ("peak", f"{profile['peakRelief']:+.5f}"),
        ("trough", f"{profile['troughRelief']:+.5f}"),
    ]
    y = mt + 4
    for key, val in lines:
        d.text((panel_x, y), key, fill=(130, 145, 165), font=f12)
        d.text((panel_x + 118, y), val, fill=(220, 230, 244), font=f12)
        y += 24

    if checks is not None:
        passed = checks["passed"]
        badge = "PASS" if passed else "FAIL"
        color = (70, 210, 130) if passed else (235, 90, 90)
        d.rounded_rectangle([panel_x, mt + ph - 46, panel_x + 170, mt + ph - 6], radius=8, fill=color)
        d.text((panel_x + 85, mt + ph - 40), badge, fill=(10, 14, 20), font=f20b, anchor="ma")
        tol = checks.get("slopeTolerance")
        if tol is not None:
            d.text((panel_x, mt + ph + 2), f"tol={tol:g}", fill=(130, 145, 165), font=f12)

    return np.asarray(img, dtype=np.uint8)
