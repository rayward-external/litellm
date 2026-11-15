"""An upstream the router tests can mock on either transport.

Anthropic `/v1/messages` runs on the Rust path only, and the Rust HTTP client never goes through httpx, so
respx cannot see its calls. `LoopbackUpstream` serves them from a real socket on 127.0.0.1 instead, answering
each request with the same `httpx.Request -> httpx.Response` callable a respx route takes. `RespxUpstream`
wraps a respx route behind the same surface, for the Python routes respx still intercepts.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, Protocol

import httpx
import respx

Answer = Callable[[httpx.Request], httpx.Response]

RESPX_API_BASE: Final = "https://anthropic.example.invalid"


@dataclass(frozen=True, slots=True)
class RecordedCall:
    request: httpx.Request


class Upstream(Protocol):
    @property
    def api_base(self) -> str: ...

    @property
    def calls(self) -> list[RecordedCall] | respx.models.CallList: ...

    @property
    def call_count(self) -> int: ...

    def mock(self, *, side_effect: Answer) -> object: ...


class RespxUpstream:
    def __init__(self, respx_mock: respx.MockRouter, path: str) -> None:
        self.api_base: Final = RESPX_API_BASE
        self._route: Final = respx_mock.post(f"{RESPX_API_BASE}{path}")

    @property
    def calls(self) -> respx.models.CallList:
        return self._route.calls

    @property
    def call_count(self) -> int:
        return self._route.call_count

    def mock(self, *, side_effect: Answer) -> object:
        return self._route.mock(side_effect=side_effect)


class LoopbackUpstream:
    def __init__(self, api_base: str) -> None:
        self.api_base: Final = api_base
        self.calls: Final[list[RecordedCall]] = []
        self._answer: Answer | None = None
        self._lock: Final = threading.Lock()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def mock(self, *, side_effect: Answer) -> object:
        self._answer = side_effect
        return self

    def answer(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.calls.append(RecordedCall(request))
            answer: Final = self._answer
        if answer is None:
            return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "unmocked"}})
        return answer(request)


def _handler(upstream: LoopbackUpstream) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # the base class's name
            del format, args

        def do_POST(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("content-length") or 0))
            request: Final = httpx.Request(
                "POST", f"{upstream.api_base}{self.path}", headers=list(self.headers.items()), content=body
            )
            response: Final = upstream.answer(request)
            content: Final = response.content
            self.send_response(response.status_code)
            for name, value in response.headers.items():
                if name.lower() not in {"content-length", "transfer-encoding", "connection"}:
                    self.send_header(name, value)
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            _ = self.wfile.write(content)

    return Handler


@contextmanager
def loopback_upstream() -> Iterator[LoopbackUpstream]:
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    upstream: Final = LoopbackUpstream(f"http://127.0.0.1:{server.server_address[1]}")
    server.RequestHandlerClass = _handler(upstream)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield upstream
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
