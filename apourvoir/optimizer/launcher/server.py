"""
HTTP servers for the launcher (standard library only).

- API server: /health, /status, /v1/models, and /v1/completions.
  No inference engine exists in Apourvoir yet, so /v1/completions answers
  501 Not Implemented instead of returning fabricated text.
- Web UI server: a small status page that reads /status.

Each server runs in a daemon thread; the launcher keeps the process alive.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED = {
    "error": {
        "type": "not_implemented",
        "message": "Apourvoir has no inference engine yet: completions are not available.",
    }
}


class _JSONHandler(BaseHTTPRequestHandler):
    status_provider: Callable[[], Dict[str, Any]] = staticmethod(lambda: {})
    model_name: str = ""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"), "application/json")

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


class APIHandler(_JSONHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/status":
            self._json(200, self.status_provider())
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model_name, "object": "model"}]})
        else:
            self._json(404, {"error": {"type": "not_found", "message": self.path}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.path in ("/v1/completions", "/v1/chat/completions"):
            self._json(501, NOT_IMPLEMENTED)
        else:
            self._json(404, {"error": {"type": "not_found", "message": self.path}})


UI_PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Apourvoir</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}pre{background:#f4f4f4;padding:1rem;overflow:auto}</style>
</head><body><h1>Apourvoir</h1>
<p>Statut du launcher (rafraîchi toutes les 5 s). Aucun moteur d'inférence n'est implémenté.</p>
<pre id="s">chargement…</pre>
<script>
async function r(){try{const x=await fetch('/status');document.getElementById('s').textContent=JSON.stringify(await x.json(),null,2)}catch(e){document.getElementById('s').textContent=String(e)}}
r();setInterval(r,5000);
</script></body></html>"""


class UIHandler(_JSONHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, UI_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/status":
            self._json(200, self.status_provider())
        else:
            self._json(404, {"error": {"type": "not_found", "message": self.path}})


class BackgroundServer:
    """A ThreadingHTTPServer running in a daemon thread."""

    def __init__(self, name: str, handler: type, host: str, port: int,
                 status_provider: Callable[[], Dict[str, Any]], model_name: str = ""):
        handler_cls = type(handler.__name__, (handler,), {
            "status_provider": staticmethod(status_provider),
            "model_name": model_name,
        })
        self.name = name
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)
        self.thread: Optional[threading.Thread] = None

    @property
    def address(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.thread = threading.Thread(target=self.httpd.serve_forever, name=self.name, daemon=True)
        self.thread.start()
        logger.info(f"{self.name} listening on {self.address}")

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"{self.name} stopped")
