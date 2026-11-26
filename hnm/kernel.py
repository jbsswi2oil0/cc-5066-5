"""高度 -> 法线核心内核。

关键原则
========
1. 梯度在**物理世界尺度**中计算：像素间距 dx = 表面实际宽度 / 列数，
   高度 H = z_scale * 线性高度值。因此
       dH/dX = z_scale * (dh/dcol) / dx
       dH/dY = -z_scale * (dh/drow) / dy      （+Y 朝画面上方）
   改变分辨率但覆盖同一物理表面时，dx 与差分步长同步缩小，法线强度不会翻倍。

2. 边界政策显式：
   - "clamp"    ：非周期贴图，边缘用单侧差分，绝不从对边借像素；
   - "periodic" ：可平铺贴图，跨左右/上下边做周期中心差分。

3. 输出法线始终先在 conventions.CANONICAL_FRAME_DOC 所述规范系中算出，
   编码时才应用坐标约定符号。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .conventions import resolve_signs

BOUNDARIES = ("clamp", "periodic")
METHODS = ("central", "sobel")


class KernelError(ValueError):
    """输入参数或数据不满足转换条件。field 用于把失败反馈到具体控件。"""

    def __init__(self, message: str, field: Optional[str] = None, code: str = "invalid_parameter"):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code


class ScaleRequired(KernelError):
    def __init__(self, message: str, missing: list[str]):
        super().__init__(message, field=missing[0] if missing else None, code="scale_required")
        self.missing = missing


@dataclass
class SurfaceScale:
    """纹理覆盖的物理表面尺度与高度标尺。

    width_world/height_world : 整张纹理覆盖的表面宽/高（世界单位）
    z_per_height_unit        : 线性高度值 1.0 对应的世界高度（实际高度标尺）
    stretch_x/stretch_y      : 模型对 UV 的额外拉伸倍数（不等距纹理坐标）；
                               拉伸后像素世界间距 = 原间距 * stretch
    unit                     : 世界单位名称（m/cm/mm...），仅用于读数
    """

    width_world: float
    height_world: float
    z_per_height_unit: float
    unit: str = "m"
    stretch_x: float = 1.0
    stretch_y: float = 1.0

    def validate(self) -> None:
        for name, value in (
            ("width_world", self.width_world),
            ("height_world", self.height_world),
            ("z_per_height_unit", self.z_per_height_unit),
            ("stretch_x", self.stretch_x),
            ("stretch_y", self.stretch_y),
        ):
            if value is None or not np.isfinite(value) or value <= 0:
                raise KernelError(f"尺度参数 {name} 必须是正数，当前为 {value!r}", field=name)

    def pitches(self, rows: int, cols: int) -> tuple[float, float]:
        self.validate()
        if rows < 2 or cols < 2:
            raise KernelError("图像至少需要 2x2 像素才能计算梯度", field="image")
        dx = self.width_world * self.stretch_x / cols
        dy = self.height_world * self.stretch_y / rows
        return dx, dy

    def describe(self, rows: int, cols: int) -> dict:
        dx, dy = self.pitches(rows, cols)
        return {
            "rows": rows,
            "cols": cols,
            "unit": self.unit,
            "widthWorld": self.width_world,
            "heightWorld": self.height_world,
            "zPerHeightUnit": self.z_per_height_unit,
            "stretchX": self.stretch_x,
            "stretchY": self.stretch_y,
            "pixelPitchX": dx,
            "pixelPitchY": dy,
            "texelAspect": dx / dy,
        }


def index_gradients(
    h: np.ndarray,
    boundary: str = "clamp",
    method: str = "central",
) -> tuple[np.ndarray, np.ndarray]:
    """返回每索引单位的差分 (dh/dcol, dh/drow)。"""
    if boundary not in BOUNDARIES:
        raise KernelError(f"未知边界政策 {boundary!r}，可选 {BOUNDARIES}", field="boundary")
    if method not in METHODS:
        raise KernelError(f"未知梯度算子 {method!r}，可选 {METHODS}", field="method")
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
        raise KernelError("高度场至少需要 2x2 采样", field="image")

    h = np.asarray(h, dtype=np.float64)

    if method == "central":
        dc = np.empty_like(h)
        dr = np.empty_like(h)
        if boundary == "periodic":
            dc[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
            dc[:, 0] = (h[:, 1] - h[:, -1]) * 0.5
            dc[:, -1] = (h[:, 0] - h[:, -2]) * 0.5
            dr[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
            dr[0, :] = (h[1, :] - h[-1, :]) * 0.5
            dr[-1, :] = (h[0, :] - h[-2, :]) * 0.5
        else:
            dc[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
            dc[:, 0] = h[:, 1] - h[:, 0]
            dc[:, -1] = h[:, -1] - h[:, -2]
            dr[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
            dr[0, :] = h[1, :] - h[0, :]
            dr[-1, :] = h[-1, :] - h[-2, :]
        return dc, dr

    # Sobel 3x3（/8 归一化使线性斜坡的响应恰为每索引导数）。
    # 内部用 Sobel；最外圈统一替换为与 central 相同的单侧/周期差分，
    # 保证边界政策语义不随算子改变。
    if boundary == "periodic":
        hp = np.pad(h, 1, mode="wrap")
    else:
        hp = np.pad(h, 1, mode="edge")

    def sobel(axis: int) -> np.ndarray:
        if axis == 1:  # 列方向核 [-1 0 1; -2 0 2; -1 0 1]
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        else:          # 行方向（向下为正）
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        out = np.zeros_like(h)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * hp[i:i + h.shape[0], j:j + h.shape[1]]
        return out / 8.0

    dc = sobel(axis=1)
    dr = sobel(axis=0)

    # 边界一圈改用与 central 一致的明确差分
    if boundary == "periodic":
        dc[:, 0] = (h[:, 1] - h[:, -1]) * 0.5
        dc[:, -1] = (h[:, 0] - h[:, -2]) * 0.5
        dr[0, :] = (h[1, :] - h[-1, :]) * 0.5
        dr[-1, :] = (h[0, :] - h[-2, :]) * 0.5
    else:
        dc[:, 0] = h[:, 1] - h[:, 0]
        dc[:, -1] = h[:, -1] - h[:, -2]
        dr[0, :] = h[1, :] - h[0, :]
        dr[-1, :] = h[-1, :] - h[-2, :]
    return dc, dr


def physical_slopes(
    h: np.ndarray,
    scale: SurfaceScale,
    boundary: str = "clamp",
    method: str = "central",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """返回世界坡度 (dH/dX, dH/dY) 以及实际像素间距 dx, dy。"""
    rows, cols = h.shape
    dx, dy = scale.pitches(rows, cols)
    dc, dr = index_gradients(h, boundary, method)
    d_h_dx = scale.z_per_height_unit * dc / dx
    d_h_dy = -scale.z_per_height_unit * dr / dy
    return d_h_dx, d_h_dy, dx, dy


def normals_from_slopes(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """HxW 坡度 -> HxWx3 单位法线（规范系）。n = normalize(-sx, -sy, 1)。"""
    nx = -sx
    ny = -sy
    nz = np.ones_like(sx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=2)


def height_to_normals(
    h: np.ndarray,
    scale: SurfaceScale,
    boundary: str = "clamp",
    method: str = "central",
) -> dict:
    """完整转换，返回规范法线、坡度与尺度读数。"""
    sx, sy, dx, dy = physical_slopes(h, scale, boundary, method)
    n = normals_from_slopes(sx, sy)
    return {
        "normals": n,
        "slope_x": sx,
        "slope_y": sy,
        "dx": dx,
        "dy": dy,
        "scaleInfo": scale.describe(*h.shape),
    }


def normals_from_physical_height(
    height_world: np.ndarray,
    dx: float,
    dy: float,
    boundary: str = "clamp",
    method: str = "central",
) -> dict:
    """高度已经是世界单位时使用（双岛合并后的物理高度场）。"""
    if dx <= 0 or dy <= 0:
        raise KernelError("像素间距必须为正", code="invalid_scale")
    dc_idx, dr_idx = index_gradients(height_world, boundary, method)
    sx = dc_idx / dx
    sy = -dr_idx / dy
    n = normals_from_slopes(sx, sy)
    return {"normals": n, "slope_x": sx, "slope_y": sy, "dx": dx, "dy": dy}


def encode_normal_rgb(n_canonical: np.ndarray, signs: tuple[int, int, int]) -> np.ndarray:
    """规范法线 -> 约定编码的 uint8 RGB。"""
    s = np.asarray(signs, dtype=np.float64).reshape(1, 1, 3)
    encoded01 = np.clip(n_canonical * s * 0.5 + 0.5, 0.0, 1.0)
    return np.rint(encoded01 * 255.0).astype(np.uint8)


def decode_normal_rgb(
    rgb_u8: np.ndarray,
    signs: tuple[int, int, int],
    renormalize: bool = True,
) -> np.ndarray:
    """编码 RGB -> 规范系法线。默认重新归一化（真实渲染器通常也会归一化）。"""
    q = rgb_u8[:, :, :3].astype(np.float64) / 255.0
    n = q * 2.0 - 1.0
    s = np.asarray(signs, dtype=np.float64).reshape(1, 1, 3)
    n = n * s
    if renormalize:
        norm = np.sqrt(np.sum(n * n, axis=2, keepdims=True))
        norm = np.maximum(norm, 1e-12)
        n = n / norm
    return n


def angular_error_deg(n_a: np.ndarray, n_b: np.ndarray) -> np.ndarray:
    dot = np.sum(n_a * n_b, axis=2) if n_a.ndim == 3 else float(np.dot(n_a, n_b))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def encoding_error_metrics(
    n_canonical: np.ndarray,
    signs: tuple[int, int, int],
) -> dict:
    """编码->解码往返误差（8-bit 量化），含最大/平均角度误差。"""
    rgb = encode_normal_rgb(n_canonical, signs)
    decoded_raw = decode_normal_rgb(rgb, signs, renormalize=False)
    decoded = decode_normal_rgb(rgb, signs, renormalize=True)
    err = angular_error_deg(n_canonical, decoded)
    component_err = np.abs(decoded_raw - n_canonical)
    return {
        "maxAngularErrorDeg": float(np.max(err)),
        "meanAngularErrorDeg": float(np.mean(err)),
        "rmseAngularErrorDeg": float(np.sqrt(np.mean(err * err))),
        "maxComponentQuantization": float(np.max(component_err)),
        "encoded": rgb,
    }


def slope_strength_stats(result: dict) -> dict:
    sx, sy = result["slope_x"], result["slope_y"]
    n = result["normals"]
    return {
        "slopeXMin": float(np.min(sx)),
        "slopeXMax": float(np.max(sx)),
        "slopeYMin": float(np.min(sy)),
        "slopeYMax": float(np.max(sy)),
        "meanAbsSlope": float(np.mean(np.sqrt(sx * sx + sy * sy))),
        "maxSlope": float(np.max(np.sqrt(sx * sx + sy * sy))),
        "normalZMin": float(np.min(n[:, :, 2])),
        "normalZMean": float(np.mean(n[:, :, 2])),
    }
