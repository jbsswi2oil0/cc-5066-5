#!/usr/bin/env bash
# 高度 -> 法线纹理工作台启动脚本
# 用法: ./start.sh [--host 0.0.0.0] [--port 8765]
set -euo pipefail
cd "$(dirname "$0")"

# 1) 依赖检查：优先系统 numpy/Pillow；缺失时装到仓库内 .pylibs（无需 root）
export PYTHONPATH="$PWD/.pylibs${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 -c "import numpy, PIL" >/dev/null 2>&1; then
  echo "[start.sh] Python 环境缺少 numpy/Pillow，准备安装到 ./.pylibs ..."
  if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "[start.sh] 未发现 pip，使用 get-pip.py 引导到 .pylibs ..."
    if [ -f .tools/get-pip.py ]; then
      python3 .tools/get-pip.py --target="$PWD/.pylibs" -q
    else
      curl -sS https://bootstrap.pypa.io/get-pip.py -o .tools-get-pip.py
      python3 .tools-get-pip.py --target="$PWD/.pylibs" -q
      rm -f .tools-get-pip.py
    fi
  fi
  python3 -m pip install -q --target="$PWD/.pylibs" numpy pillow
fi

# 2) 首次运行生成内置真实纹理资产
if [ ! -f assets/source/fbm_rock.png ]; then
  echo "[start.sh] 生成内置高度纹理资产 ..."
  python3 -m hnm.cli assets
fi

# 3) 启动服务（hnm 包也会自动把 .pylibs 加入 sys.path）
exec python3 -m hnm.server "$@"
