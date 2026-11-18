"""Python-3.9-compatible LEAD RPC server; the model stays in UniDriveVLA's navsim env."""

import argparse
import ipaddress
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from unidrivevla.closedloop.lead_protocol import (
    LEAD_INPUT_HZ,
    LEAD_POSE_INTERVAL_S,
    LEAD_TEMPORAL_RING_SIZE,
    MAX_REQUEST_BYTES,
    decode_request,
    encode_trajectory,
)
from unidrivevla.serve import load_runtime

LOG = logging.getLogger("unidrivevla.lead_server")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _handler(runtime):
    class LeadHandler(BaseHTTPRequestHandler):
        server_version = "UniDriveVLA-LEAD/1"

        def log_message(self, format, *args):
            LOG.info("%s - %s", self.client_address[0], format % args)

        def _json(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            return self._json(status, json.dumps({"error": message}, separators=(",", ":")).encode("utf-8"))

        def do_GET(self):
            if self.path != "/alive":
                return self._error(404, "not found")
            return self._json(200, b"true")

        def do_POST(self):
            if self.path == "/reset":
                runtime.reset()
                return self._json(200, b"true")
            if self.path != "/infer":
                return self._error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Content-Length must be 1..128 MiB")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete LEAD request body")
                request = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
                inputs = decode_request(request)
                trajectory = runtime.predict(**inputs)
                return self._json(200, encode_trajectory(trajectory))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self._error(400, str(exc))
            except Exception:
                LOG.exception("LEAD inference failed")
                runtime.reset()
                return self._error(500, "inference failed; model state was reset")

    return LeadHandler


def serve(runtime, host="127.0.0.1", port=9001):
    if (runtime.profile.name != "lead" or runtime.profile.horizon != 8
            or runtime.profile.pose_interval != LEAD_POSE_INTERVAL_S):
        raise ValueError("LEAD RPC requires the LEAD 4 Hz profile with an 8-step, 2-second plan")
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError as exc:
            raise ValueError("LEAD RPC host must be localhost or a loopback IP") from exc
        if not loopback:
            raise ValueError("LEAD RPC binds to loopback only; use an authenticated HTTPS proxy for remote access")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("LEAD RPC port must be in 1..65535")
    runtime.enable_temporal_ring(LEAD_TEMPORAL_RING_SIZE)
    server = HTTPServer((host, int(port)), _handler(runtime))
    LOG.info("LEAD UniDriveVLA server listening on %s:%d", host, port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        runtime.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve UniDriveVLA for the official LEAD CARLA agent")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vlm-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args(argv)
    runtime = load_runtime("lead", args.checkpoint, args.device, vlm_path=args.vlm_path)
    serve(runtime, args.host, args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
