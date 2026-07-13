from http.server import HTTPServer, SimpleHTTPRequestHandler


class SecurityHeadersHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


print("Server running at http://localhost:8000")
HTTPServer(("localhost", 8000), SecurityHeadersHandler).serve_forever()
