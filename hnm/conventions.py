"""切线空间坐标约定与法线编码符号。

内部规范坐标系（canonical frame）是显式且唯一的数学参考：

* +X = 沿存储图像的列递增，即画面右方；
* +Y = 画面上方，即行号递减方向；
* +Z = 表面朝向观察者（高度场 H(X, Y) 的外侧）。

该坐标系为右手系。内核求出的解析法线 n_c = normalize(-dH/dX, -dH/dY, 1)
始终处于此规范系。编码到 RGB 时，再按目标渲染器/引擎的约定对三个分量
分别乘 +/-1。任何“翻轴”造成的颜色与光照差异都必须能由这里的符号解释。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Convention:
    id: str
    label: str
    sign_x: int
    sign_y: int
    sign_z: int
    green: str
    note: str

    @property
    def signs(self) -> tuple[int, int, int]:
        return self.sign_x, self.sign_y, self.sign_z


CANONICAL_FRAME_DOC = (
    "规范切线系：+X 朝画面右（列+），+Y 朝画面上（行-），+Z 朝表面外；右手系。"
)

CONVENTIONS: Dict[str, Convention] = {
    "opengl": Convention(
        "opengl",
        "OpenGL / Blender / Unity（绿通道 +Y 朝上）",
        1, 1, 1,
        "+Y（绿=上方）",
        "标准 OpenGL 风格法线贴图，Blender 与 Unity 通常使用此约定。",
    ),
    "directx": Convention(
        "directx",
        "DirectX / Unreal（绿通道 -Y 朝下）",
        1, -1, 1,
        "-Y（绿=下方）",
        "仅翻转绿通道，Unreal Engine 与 DirectX 管线通常使用此约定。",
    ),
    "opengl_invert_x": Convention(
        "opengl_invert_x",
        "OpenGL 系但翻转红通道 (-X)",
        -1, 1, 1,
        "+Y",
        "用于切线基手性相反或 U 轴被镜像的网格。",
    ),
    "directx_invert_x": Convention(
        "directx_invert_x",
        "DirectX 系且翻转红通道 (-X, -Y)",
        -1, -1, 1,
        "-Y",
        "U 轴镜像叠加 DirectX 绿通道翻转时使用。",
    ),
}

PRESET_ORDER = ["opengl", "directx", "opengl_invert_x", "directx_invert_x"]


def resolve_signs(
    convention_id: str,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> tuple[int, int, int]:
    """预设符号上再叠加显式单轴翻转。"""
    conv = CONVENTIONS.get(convention_id)
    if conv is None:
        raise ValueError(f"未知坐标约定: {convention_id!r}，可选 {sorted(CONVENTIONS)}")
    sx, sy, sz = conv.signs
    if flip_x:
        sx = -sx
    if flip_y:
        sy = -sy
    if flip_z:
        sz = -sz
    return sx, sy, sz


def convention_metadata(
    convention_id: str,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> dict:
    conv = CONVENTIONS[convention_id]
    sx, sy, sz = resolve_signs(convention_id, flip_x, flip_y, flip_z)
    return {
        "preset": conv.id,
        "presetLabel": conv.label,
        "greenChannel": conv.green,
        "presetNote": conv.note,
        "canonicalFrame": CANONICAL_FRAME_DOC,
        "encoding": "RGB8: channel = round(255 * clamp(sign * n_canonical * 0.5 + 0.5))",
        "signs": {"x": sx, "y": sy, "z": sz},
        "explicitFlips": {"x": bool(flip_x), "y": bool(flip_y), "z": bool(flip_z)},
    }
