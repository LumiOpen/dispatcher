import asyncio
from binascii import hexlify
from urllib.parse import unquote

import h11
from uvicorn.protocols.http.h11_impl import (
    HIGH_WATER_LIMIT,
    H11Protocol,
    RequestResponseCycle,
    service_unavailable,
)


class InvalidRequestLoggingH11Protocol(H11Protocol):
    """Uvicorn h11 protocol with extra diagnostics for malformed requests.

    FastAPI middleware cannot log these errors because h11 rejects the bytes
    before an ASGI request exists. This protocol is intentionally opt-in.
    """

    invalid_http_preview_bytes = 64

    def data_received(self, data: bytes) -> None:
        preview_size = max(0, self.invalid_http_preview_bytes)
        if preview_size:
            current = getattr(self, "_invalid_http_preview", b"")
            self._invalid_http_preview = (current + data)[:preview_size]
        super().data_received(data)

    def _format_invalid_http_preview(self) -> tuple[str, str]:
        data = getattr(self, "_invalid_http_preview", b"")
        hex_preview = hexlify(data).decode("ascii")
        ascii_preview = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in data)
        return hex_preview, ascii_preview

    def handle_events(self) -> None:
        while True:
            try:
                event = self.conn.next_event()
            except h11.RemoteProtocolError as exc:
                msg = "Invalid HTTP request received."
                hex_preview, ascii_preview = self._format_invalid_http_preview()
                self.logger.warning(
                    "%s client=%s server=%s error=%r first_bytes_hex=%s first_bytes_ascii=%r",
                    msg,
                    self.client,
                    self.server,
                    str(exc),
                    hex_preview,
                    ascii_preview,
                )
                self.send_400_response(msg)
                return

            if event is h11.NEED_DATA:
                break

            elif event is h11.PAUSED:
                self.flow.pause_reading()
                break

            elif isinstance(event, h11.Request):
                self.headers = [(key.lower(), value) for key, value in event.headers]
                raw_path, _, query_string = event.target.partition(b"?")
                path = unquote(raw_path.decode("ascii"))
                full_path = self.root_path + path
                full_raw_path = self.root_path.encode("ascii") + raw_path
                self.scope = {
                    "type": "http",
                    "asgi": {"version": self.config.asgi_version, "spec_version": "2.3"},
                    "http_version": event.http_version.decode("ascii"),
                    "server": self.server,
                    "client": self.client,
                    "scheme": self.scheme,
                    "method": event.method.decode("ascii"),
                    "root_path": self.root_path,
                    "path": full_path,
                    "raw_path": full_raw_path,
                    "query_string": query_string,
                    "headers": self.headers,
                    "state": self.app_state.copy(),
                }
                if self._should_upgrade():
                    self.handle_websocket_upgrade(event)
                    return

                if self.limit_concurrency is not None and (
                    len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency
                ):
                    app = service_unavailable
                    message = "Exceeded concurrency limit."
                    self.logger.warning(message)
                else:
                    app = self.app

                self._unset_keepalive_if_required()

                self.cycle = RequestResponseCycle(
                    scope=self.scope,
                    conn=self.conn,
                    transport=self.transport,
                    flow=self.flow,
                    logger=self.logger,
                    access_logger=self.access_logger,
                    access_log=self.access_log,
                    default_headers=self.server_state.default_headers,
                    message_event=asyncio.Event(),
                    on_response=self.on_response_complete,
                )
                task = self.loop.create_task(self.cycle.run_asgi(app))
                task.add_done_callback(self.tasks.discard)
                self.tasks.add(task)

            elif isinstance(event, h11.Data):
                if self.conn.our_state is h11.DONE:
                    continue
                self.cycle.body += event.data
                if len(self.cycle.body) > HIGH_WATER_LIMIT:
                    self.flow.pause_reading()
                self.cycle.message_event.set()

            elif isinstance(event, h11.EndOfMessage):
                if self.conn.our_state is h11.DONE:
                    self.transport.resume_reading()
                    self.conn.start_next_cycle()
                    continue
                self.cycle.more_body = False
                self.cycle.message_event.set()
                if self.conn.their_state == h11.MUST_CLOSE:
                    break
