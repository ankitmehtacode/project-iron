import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class SecurityHeadersHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    ui_dir = Path(__file__).resolve().parent
    handler = functools.partial(SecurityHeadersHandler, directory=str(ui_dir))
    print("Server running at http://localhost:8000")
    HTTPServer(("localhost", 8000), handler).serve_forever()

