"""Receive Alertmanager notifications in the local Compose stack."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Webhook(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 64 * 1024:
                raise ValueError("invalid payload size")
            payload = json.loads(self.rfile.read(size))
            status, alerts = payload["status"], payload["alerts"]
            if status not in ("firing", "resolved") or not isinstance(alerts, list):
                raise ValueError("invalid alert payload")
            for alert in alerts:
                print(
                    json.dumps(
                        {
                            "status": alert["status"],
                            "alert": alert["labels"]["alertname"],
                            "kind": alert["labels"].get("kind"),
                        }
                    )
                )
        except (ValueError, TypeError, KeyError):
            self.send_error(400)
            return
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    with HTTPServer(("0.0.0.0", 8080), Webhook) as server:
        server.serve_forever()
