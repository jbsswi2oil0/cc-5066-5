"""高度纹理 -> 物理正确的法线贴图内核与服务。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_LIBS = _ROOT / ".pylibs"
if _LIBS.is_dir() and str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))

__version__ = "1.0.0"
