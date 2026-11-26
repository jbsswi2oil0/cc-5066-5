"""零框架 HTTP 服务：托管 web 界面并把请求交给 hnm.pipeline 内核。

启动：python -m hnm.server [--host 127.0.0.1] [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __version__, pipeline
from .kernel import KernelError, ScaleRequired
from .samples import ASSET_DIR, OUTPUT_DIR, REPO_ROOT

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY = 32 * 1024 * 1024

_STATIC = {
    "/": (WEB_DIR / "index.html", "text/html; charset=utf-8"),
    "/index.html": (WEB_DIR / "index.html", "text/html; charset=utf-8"),
    "/app.js": (WEB_DIR / "app.js", "application/javascript; charset=utf-8"),
    "/styles.css": (WEB_DIR / "styles.css", "text/css; charset=utf-8"),
}


def _safe_under(root: Path, rel: str) -> Path | None:
    root = root.resolve()
    target = (root / rel).resolve()
    if target == root or root in target.parents:
        return target
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = f"HeightNormalStudio/{__version__}"

    def log_message(self, fmt: str, *args) -> None:
        # 简洁日志
        print(f"[http] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json({"ok": False, "error": {"message": f"文件不存在: {path.name}"}}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        if path in _STATIC:
            f, ct = _STATIC[path]
            return self._send_file(f, ct)
        if path == "/api/samples":
            try:
                return self._send_json({"ok": True, **pipeline.list_samples()})
            except Exception as exc:
                traceback.print_exc()
                return self._send_json(self._error_body(exc), 500)
        if path.startswith("/assets/"):
            target = _safe_under(ASSET_DIR, path[len("/assets/"):])
            if target and target.is_file():
                return self._send_file(target, _ctype(target.suffix))
            return self._send_json({"ok": False, "error": {"message": "asset not found"}}, 404)
        if path.startswith("/outputs/"):
            target = _safe_under(OUTPUT_DIR, path[len("/outputs/"):])
            if target and target.is_file():
                return self._send_file(target, _ctype(target.suffix))
            return self._send_json({"ok": False, "error": {"message": "output not found"}}, 404)
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send_json({"ok": False, "error": {"message": f"未知路径 {path}"}}, 404)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path not in ("/api/convert", "/api/join", "/api/verify", "/api/verification"):
            return self._send_json({"ok": False, "error": {"message": "未知 API"}}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise KernelError("请求体为空或超过 32MB 限制", code="body_too_large")
            raw = self.rfile.read(length)
            params = json.loads(raw.decode("utf-8"))
            if url.path == "/api/convert":
                result = pipeline.convert_single(params)
            elif url.path == "/api/verify":
                result = pipeline.verify_scaling(params)
            elif url.path == "/api/verification":
                from .normal_validation import run_all_verification
                result = {"ok": True, **run_all_verification()}
            else:
                result = pipeline.join_islands(params)
            return self._send_json(result)
        except ScaleRequired as exc:
            return self._send_json(
                {"ok": False, "error": {
                    "code": exc.code, "message": exc.message,
                    "field": exc.field, "missing": exc.missing,
                }},
                422,
            )
        except KernelError as exc:
            return self._send_json(
                {"ok": False, "error": {"code": exc.code, "message": exc.message, "field": exc.field}},
                400,
            )
        except json.JSONDecodeError as exc:
            return self._send_json(
                {"ok": False, "error": {"code": "bad_json", "message": f"JSON 解析失败: {exc}"}}, 400
            )
        except Exception as exc:
            traceback.print_exc()
            return self._send_json(
                {"ok": False, "error": {"code": "internal", "message": f"内核内部错误: {exc}"}}, 500
            )

    @staticmethod
    def _error_body(exc: Exception) -> dict:
        return {"ok": False, "error": {"code": "internal", "message": str(exc)}}


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _ctype(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
    }.get(suffix.lower(), "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="高度->法线纹理工作台")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(__import__("os").environ.get("PORT", "8765")))
    args = parser.parse_args()

    if not ASSET_DIR.exists():
        print("[startup] 未发现样例资产，先生成 ...")
        from .samples import build_all_assets
        info = build_all_assets()
        print(f"[startup] 已生成 {len(info['written'])} 个资产文件")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 64)
    print(" 高度纹理 -> 法线贴图工作台 已启动")
    print(f" 界面地址: {url}")
    print(" 按 Ctrl+C 停止服务")
    print("=" * 64)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] 正在停止 ...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
