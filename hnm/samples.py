"""内置真实纹理资产的程序化生成与样例注册表。

所有几何样例都由已知解析高度场生成，便于用解析法线核对内核；
双岛样例来自同一个解析表面 H(X,Y)=beta*(X-seam)+A*exp(-(r/sigma)^2)，
因此跨缝检查存在客观真值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .images import save_png

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "assets" / "source"
OUTPUT_DIR = REPO_ROOT / "outputs"


def ramp_field(n: int = 256) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, v)
    h = 0.12 + 0.72 * (0.62 * U + 0.38 * V)
    return np.clip(h, 0.0, 1.0)


def sine_field(n: int = 256) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, v)
    return 0.5 + 0.28 * np.sin(2 * np.pi * 3 * U) * np.cos(2 * np.pi * 2 * V)


def step_field(n: int = 128) -> np.ndarray:
    h = np.full((n, n), 0.28, dtype=np.float64)
    h[:, n // 2:] = 0.72
    # 顶/底也放一个台阶，便于演示行方向边界政策
    h[: n // 5, : n // 2] = 0.55
    return h


def _value_noise(rng: np.random.Generator, cells: int, n: int) -> np.ndarray:
    grid = rng.random((cells + 1, cells + 1))
    y = np.linspace(0, cells, n, endpoint=False)
    x = np.linspace(0, cells, n, endpoint=False)
    yi = np.floor(y).astype(int)
    xi = np.floor(x).astype(int)
    yf = y - yi
    xf = x - xi
    yf = yf * yf * (3 - 2 * yf)
    xf = xf * xf * (3 - 2 * xf)
    v00 = grid[yi[:, None], xi[None, :]]
    v01 = grid[yi[:, None], xi[None, :] + 1]
    v10 = grid[yi[:, None] + 1, xi[None, :]]
    v11 = grid[yi[:, None] + 1, xi[None, :] + 1]
    a = v00 * (1 - xf) + v01 * xf
    b = v10 * (1 - xf) + v11 * xf
    return a * (1 - yf[:, None]) + b * yf[:, None]


def fbm_field(n: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n))
    amp = 0.55
    cells = 4
    total = 0.0
    for octave in range(6):
        h += amp * _value_noise(rng, cells, n)
        total += amp
        amp *= 0.5
        cells *= 2
    h /= total
    lo, hi = float(h.min()), float(h.max())
    return np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)


def island_pair_fields(n: int = 256):
    """返回 (hA, hB, truth)。两岛是同一解析表面的左右裁剪，外观互为镜像。"""
    width = 1.0
    height = 0.5
    seam = 0.5
    beta = 0.08
    amp = 0.015
    sigma = 0.07

    x = (np.arange(2 * n) + 0.5) * (width / (2 * n))
    y = (np.arange(n) + 0.5) * (height / n)
    X, Y = np.meshgrid(x, y)
    xc = X - seam
    yc = Y - height / 2
    H = beta * xc + amp * np.exp(-(xc * xc + yc * yc) / (sigma * sigma))

    hmin, hmax = float(H.min()), float(H.max())
    z_truth = hmax - hmin
    h = (H - hmin) / z_truth
    hA = np.ascontiguousarray(h[:, :n])
    hB = np.ascontiguousarray(h[:, n:])
    truth = {
        "widthWorld": 0.5,
        "heightWorld": 0.5,
        "zScale": float(z_truth),
        "unit": "m",
        "beta": beta,
        "amp": amp,
        "sigma": sigma,
        "seam": "vertical @ X=0.5m",
        "note": "zScale 是线性归一化高度 1.0 对应的真实世界高度（米）",
    }
    return hA, hB, truth


def to_u8gray(h: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(h, 0, 1) * 255.0 + 0.5)


SAMPLE_REGISTRY = [
    {
        "id": "fbm_rock",
        "label": "FBM 岩面（真实感噪声高度）",
        "file": "fbm_rock.png",
        "widthWorld": 1.0,
        "heightWorld": 1.0,
        "zScale": 0.028,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
        "boundary": "clamp",
        "description": "6 阶分形噪声生成的岩面高度，线性灰度数据。",
    },
    {
        "id": "sine_relief",
        "label": "正弦浮雕（解析真值）",
        "file": "sine_relief.png",
        "widthWorld": 1.0,
        "heightWorld": 1.0,
        "zScale": 0.108,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
        "boundary": "periodic",
        "description": "h=0.5+0.28·sin(6πU)cos(4πV)，周期边界，可逐像素核对解析法线。",
    },
    {
        "id": "plane_ramp",
        "label": "平面斜坡（解析真值）",
        "file": "plane_ramp.png",
        "widthWorld": 2.0,
        "heightWorld": 1.0,
        "zScale": 0.5,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
        "boundary": "clamp",
        "description": "严格线性斜坡，中心差分内部应与解析法线完全一致；宽高尺度不等距。",
    },
    {
        "id": "boundary_step",
        "label": "边界台阶（周期/非周期）",
        "file": "boundary_step.png",
        "widthWorld": 1.0,
        "heightWorld": 1.0,
        "zScale": 0.2,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
        "boundary": "clamp",
        "description": "含内部台阶与对边台阶，用于检查两种边界政策。",
    },
    {
        "id": "srgb_encoded_ramp",
        "label": "sRGB 编码斜坡（传递函数演示）",
        "file": "srgb_encoded_ramp.png",
        "widthWorld": 1.0,
        "heightWorld": 1.0,
        "zScale": 0.3,
        "unit": "m",
        "channel": "r",
        "transfer": "srgb",
        "boundary": "clamp",
        "description": "同一线性斜坡经 sRGB 显示编码保存；必须用 sRGB EOTF 解码，不能直接当线性高度。",
    },
]

ISLAND_REGISTRY = {
    "islandA": {
        "id": "island_A_calibrated",
        "label": "A 岛（带实际高度标尺）",
        "file": "island_A_calibrated.png",
        "widthWorld": 0.5,
        "heightWorld": 0.5,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
    },
    "islandB": {
        "id": "island_B_relative",
        "label": "B 岛（仅相对起伏，标尺缺失）",
        "file": "island_B_relative.png",
        "widthWorld": 0.5,
        "heightWorld": 0.5,
        "unit": "m",
        "channel": "r",
        "transfer": "linear",
    },
}


def build_all_assets(asset_dir: Path = ASSET_DIR) -> dict:
    asset_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fields = {
        "plane_ramp.png": ramp_field(256),
        "sine_relief.png": sine_field(256),
        "boundary_step.png": step_field(128),
        "fbm_rock.png": fbm_field(512),
    }
    linear_ramp = ramp_field(256)
    # 用 sRGB OETF 编码制造“显示编码后”的斜坡资产
    encoded = np.where(
        linear_ramp <= 0.0031308,
        linear_ramp * 12.92,
        1.055 * np.power(np.clip(linear_ramp, 0, 1), 1 / 2.4) - 0.055,
    )
    fields["srgb_encoded_ramp.png"] = encoded

    for name, h in fields.items():
        p = asset_dir / name
        save_png(to_u8gray(h), p)
        written.append(str(p))

    hA, hB, truth = island_pair_fields(256)
    save_png(to_u8gray(hA), asset_dir / ISLAND_REGISTRY["islandA"]["file"])
    save_png(to_u8gray(hB), asset_dir / ISLAND_REGISTRY["islandB"]["file"])
    atlas = np.concatenate([hA, np.ones((256, 8)) * 0.0, hB], axis=1)
    save_png(to_u8gray(atlas), asset_dir / "island_pair_atlas.png")
    written += [
        str(asset_dir / ISLAND_REGISTRY["islandA"]["file"]),
        str(asset_dir / ISLAND_REGISTRY["islandB"]["file"]),
        str(asset_dir / "island_pair_atlas.png"),
    ]
    return {"written": written, "islandTruth": truth}
