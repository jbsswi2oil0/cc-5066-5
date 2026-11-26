"""图像读写与 base64 data URL 辅助。"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def load_rgba_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def encode_png(arr_u8: np.ndarray) -> bytes:
    if arr_u8.dtype != np.uint8:
        arr_u8 = np.clip(arr_u8, 0, 255).astype(np.uint8)
    if arr_u8.ndim == 2:
        mode = "L"
    elif arr_u8.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode=mode).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def save_png(arr_u8: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(arr_u8))
    return path


def png_data_url(arr_u8: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(arr_u8)).decode("ascii")


def bytes_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def decode_data_url(data_url: str) -> bytes:
    if "," in data_url and data_url.startswith("data:"):
        header, b64 = data_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("仅支持 base64 编码的 data URL")
        return base64.b64decode(b64)
    return base64.b64decode(data_url)
