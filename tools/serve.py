#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Static dev server for web/ with HTTP Range support (tiles are streamed by range).

    python3 tools/serve.py [--port 8080]
"""
import argparse
import functools
import http.server
import os
import re

MIME = {".wasm": "application/wasm", ".js": "text/javascript", ".json": "application/json"}


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        return MIME.get(os.path.splitext(path)[1], super().guess_type(path))

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(65536, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)
        self._remaining = None

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--root", help="directory to serve (default: web/)")
    a = ap.parse_args()
    root = a.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
    handler = functools.partial(RangeHandler, directory=os.path.abspath(root))
    print(f"latent-fovea: http://127.0.0.1:{a.port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
