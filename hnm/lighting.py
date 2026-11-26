"""简单可解析的方向光表面着色，用于渲染预览与光照方向核对。

所有计算都在规范切线系（+X 右 / +Y 上 / +Z 外）中进行，
因此可以用解析坡度直接断言“光从右侧来时右坡更亮”。
"""
from __future__ import annotations

import numpy as np


def light_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """方位角从 +X 轴向 +Y 轴逆时针计；仰角为光线高出 XY 平面的角度。

    返回指向光源的单位向量 L。仰角 90 时 L=+Z。
    """
    az = np.radians(azimuth_deg)
    el = np.radians(np.clip(elevation_deg, -90.0, 90.0))
    return np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )


def shade(
    normals: np.ndarray,
    azimuth_deg: float = 315.0,
    elevation_deg: float = 45.0,
    ambient: float = 0.22,
    base_color: tuple[float, float, float] = (0.72, 0.74, 0.78),
    specular: float = 0.25,
    shininess: float = 24.0,
) -> np.ndarray:
    """法线 HxWx3 -> uint8 RGB 光照预览。"""
    n = normals
    l = light_vector(azimuth_deg, elevation_deg)
    ndotl = np.clip(np.sum(n * l.reshape(1, 1, 3), axis=2), 0.0, 1.0)
    diffuse = ambient + (1.0 - ambient) * ndotl

    # Blinn-Phong 半程向量
    v = np.array([0.0, 0.0, 1.0])
    h = l + v
    h /= np.linalg.norm(h)
    ndoth = np.clip(np.sum(n * h.reshape(1, 1, 3), axis=2), 0.0, 1.0)
    spec = specular * np.power(ndoth, shininess)

    color = np.asarray(base_color).reshape(1, 1, 3)
    out = color * diffuse[:, :, None] + spec[:, :, None]
    return np.uint8(np.clip(out, 0.0, 1.0) * 255.0 + 0.5)
