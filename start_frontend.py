#!/usr/bin/env python
"""Start simple HTTP server for static files"""

import http.server
import socketserver
import os
from pathlib import Path

os.chdir(str(Path(__file__).parent / "static"))
PORT = 5000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"[OK] Frontend server running on http://localhost:{PORT}")
    print(f"[OK] Open http://localhost:{PORT} in your browser")
    httpd.serve_forever()
