"""Tiny static server so the 3D viewer is a real webpage, not file://."""
from __future__ import annotations

import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import OUTPUT

PORT = 8765
_started = False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUTPUT), **k)

    def log_message(self, fmt, *args):
        return


def start_viewer_server():
    global _started
    if _started:
        return PORT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    _started = True
    return PORT
