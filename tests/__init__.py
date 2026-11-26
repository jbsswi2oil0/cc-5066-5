import sys
from pathlib import Path

_LIBS = Path(__file__).resolve().parent.parent / ".pylibs"
if _LIBS.is_dir() and str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))
