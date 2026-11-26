"""输入通道选择与传递函数（Transfer Function / EOTF）。

显示编码（sRGB / gamma）后的亮度不能直接当成线性高度使用：灰度值 0.5
在线性光中约为 0.214，直接求梯度会让暗部坡度被系统性夸大。因此通道取值后
必须显式选择传递函数。高度数据若本就是线性数值，选择 linear（保持原样）。
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("r", "g", "b", "a", "luma")
TRANSFERS = ("linear", "srgb", "gamma")


def srgb_decode(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def transfer_apply(values01: np.ndarray, transfer: str, gamma: float = 2.2) -> np.ndarray:
    if transfer == "linear":
        return values01.astype(np.float64)
    if transfer == "srgb":
        return srgb_decode(values01)
    if transfer == "gamma":
        if gamma is None or gamma <= 0:
            raise ValueError("gamma 必须为正数")
        return np.power(np.clip(values01, 0.0, 1.0), float(gamma))
    raise ValueError(f"未知传递函数: {transfer!r}，可选 {TRANSFERS}")


def extract_linear_height(
    rgba_u8: np.ndarray,
    channel: str = "luma",
    transfer: str = "linear",
    gamma: float = 2.2,
    invert: bool = False,
) -> np.ndarray:
    """HxWx4 uint8 -> HxW float64 线性高度代理值（约 0..1，未乘物理标尺）。"""
    if rgba_u8.ndim != 3 or rgba_u8.shape[2] != 4:
        raise ValueError("输入必须是 HxWx4 的 RGBA uint8 数组")
    if channel not in CHANNELS:
        raise ValueError(f"未知通道 {channel!r}，可选 {CHANNELS}")

    arr = rgba_u8[:, :, :3].astype(np.float64) / 255.0
    alpha = rgba_u8[:, :, 3].astype(np.float64) / 255.0

    if channel == "r":
        raw = arr[:, :, 0]
        h = transfer_apply(raw, transfer, gamma)
    elif channel == "g":
        raw = arr[:, :, 1]
        h = transfer_apply(raw, transfer, gamma)
    elif channel == "b":
        raw = arr[:, :, 2]
        h = transfer_apply(raw, transfer, gamma)
    elif channel == "a":
        h = transfer_apply(alpha, transfer, gamma)
    else:  # luma：先对每个通道线性化，再按 Rec.709 系数求和
        lin = transfer_apply(arr, transfer, gamma)
        h = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

    if invert:
        h = 1.0 - h
    return np.ascontiguousarray(h, dtype=np.float64)
