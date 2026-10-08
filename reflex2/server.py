"""Reflex-2 HTTP endpoint (same request shape as Reflex-1 / Jev, with video, image and audio inputs).

POST /v1/decisions
  {"video": "<path or base64>",            # or "image" / "audio" / "state" (text)
   "questions": {"fall": {"type": "choice", "head": "fall"},                       # trained head
                 "scene": {"type": "choice", "instructions": "What is shown?",     # untrained: similarity
                           "criteria": {"kitchen": "a kitchen", "street": "a street"}}}}
-> {"answers": {"fall": {"choice": "fall", "confidence": 0.98, "probs": {...}}, ...}, "latency_ms": 290}
GET /health

Usage:
  python -m reflex2.server --heads heads --port 8098
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Reflex2


def _input(req: dict):
    """Return (value, kind). Files may be given as a local path or base64 (optionally a data URL)."""
    for kind in ("video", "image", "audio"):
        v = req.get(kind)
        if v is None:
            continue
        if os.path.exists(v):
            return v, kind
        raw = base64.b64decode(v.split(",", 1)[1] if v.startswith("data:") else v)
        if kind == "image":
            from PIL import Image

            return Image.open(io.BytesIO(raw)).convert("RGB"), kind
        suffix = {"video": ".mp4", "audio": ".wav"}[kind]
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(raw)
        f.close()
        return f.name, kind
    s = req.get("state", "")
    return (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)), "text"


def decide(rf: Reflex2, req: dict, lock: threading.Lock) -> dict:
    x, kind = _input(req)
    out = {}
    with lock:
        v = rf.embed(x, kind)  # one embedding serves every question
        for name, q in req["questions"].items():
            if q.get("head"):
                if q["head"] not in rf.heads:
                    raise ValueError(f"unknown head: {q['head']} (available: {list(rf.heads)})")
                out[name] = rf.decide_vec(v, q["head"])
            else:
                out[name] = rf.zero_shot_vec(v, q["criteria"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--heads", default="heads")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8098)
    a = ap.parse_args()
    rf = Reflex2(a.model).load_heads(a.heads)
    lock = threading.Lock()
    print(f"READY heads={list(rf.heads)} device={rf.device} port={a.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "heads": list(rf.heads), "device": rf.device})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decisions":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.time()
                ans = decide(rf, req, lock)
                self._send(200, {"answers": ans, "latency_ms": int((time.time() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001 report the error to the caller
                self._send(400, {"error": str(e)})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
