"""NeuroNCAP's HTTP Model Node and shared inference-only policy loader."""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf

from unidrivevla.adapters import initialize_for_inference, load_checkpoint_agent_config
from unidrivevla.closedloop.neuro_ncap import canonical_input, trajectory_to_ncap
from unidrivevla.closedloop.runtime import PolicyRuntime

LOG = logging.getLogger("unidrivevla.serve")
PACKAGE = Path(__file__).resolve().parent


def load_runtime(profile="nuscenes", checkpoint=None, device="cuda", with_motion=None, vlm_path=None,
                 checkpoint_frame=None):
    if profile not in {"nuscenes", "b2d", "navsim"}:
        raise ValueError(f"unknown profile {profile!r}; choose nuscenes, b2d, or navsim")
    if checkpoint_frame not in (None, "ego", "lidar"):
        raise ValueError("checkpoint_frame must be null, 'ego', or 'lidar'")
    if profile == "navsim" and checkpoint_frame == "lidar":
        raise ValueError("NAVSIM checkpoints use the ego frame")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"inference requires an existing checkpoint file: {checkpoint!r}")
    metadata = load_checkpoint_agent_config(checkpoint)
    if metadata is not None:
        agent_cfg = OmegaConf.create(metadata)
        # The saved frame in each profile is the checkpoint's provenance. A setting kept from
        # the training config may have described its input checkpoint, so only honor this call's
        # explicit compatibility choice during inference.
        agent_cfg.checkpoint_frame = None
        if "profiles" in agent_cfg:
            if profile not in agent_cfg.profiles:
                raise ValueError(f"checkpoint does not contain profile {profile!r}")
            saved_frame = OmegaConf.select(agent_cfg, f"profiles.{profile}.frame")
            if profile == "navsim" and saved_frame not in (None, "ego"):
                raise ValueError("NAVSIM checkpoints must use the ego frame")
            if checkpoint_frame is not None and saved_frame is not None and checkpoint_frame != saved_frame:
                raise ValueError("requested checkpoint_frame disagrees with the selected checkpoint profile metadata")
            # A CLI choice describes only the selected source. Passing it as a scalar to a multi-profile
            # initializer would incorrectly compare it with every other source's legitimate frame.
            if checkpoint_frame is not None:
                agent_cfg.checkpoint_frame = {profile: checkpoint_frame}
            agent_cfg.exact_resume = False
        elif str(agent_cfg.profile.name) != profile:
            raise ValueError(f"checkpoint profile is {agent_cfg.profile.name!r}, not {profile!r}")
        else:
            saved_frame = OmegaConf.select(agent_cfg, "profile.frame")
            if profile == "navsim" and saved_frame not in (None, "ego"):
                raise ValueError("NAVSIM checkpoints must use the ego frame")
            if checkpoint_frame is not None and saved_frame is not None and checkpoint_frame != saved_frame:
                raise ValueError("requested checkpoint_frame disagrees with checkpoint profile metadata")
            if checkpoint_frame is not None:
                agent_cfg.checkpoint_frame = checkpoint_frame
        if with_motion is not None and bool(with_motion) != bool(agent_cfg.with_motion):
            raise ValueError("requested motion architecture differs from the checkpoint agent_config")
    else:
        if profile in {"nuscenes", "b2d"} and checkpoint_frame is None:
            raise ValueError(f"legacy {profile} checkpoints have no frame metadata; pass a verified --checkpoint-frame")
        profile_name = f"{profile}_ego" if profile in {"nuscenes", "b2d"} and checkpoint_frame == "ego" else profile
        profile_cfg = OmegaConf.load(PACKAGE / "config" / "profile" / f"{profile_name}.yaml")
        agent_cfg = OmegaConf.load(PACKAGE / "config" / "agent" / "unidrivevla.yaml")
        agent_cfg.profile = profile_cfg
        agent_cfg.with_motion = bool(with_motion)
        if checkpoint_frame is not None:
            agent_cfg.checkpoint_frame = checkpoint_frame
    agent_cfg.checkpoint_path = str(checkpoint)
    agent_cfg.train_vlm = False
    agent_cfg.gradient_checkpointing = False
    if vlm_path:
        agent_cfg.vlm_path = str(vlm_path)
    OmegaConf.resolve(agent_cfg)
    if not Path(agent_cfg.vlm_path).is_dir():
        raise FileNotFoundError(f"Qwen3-VL model directory does not exist: {agent_cfg.vlm_path}")
    # Checkpoint metadata is data: never recursively instantiate a nested _target_ from it.
    policy = instantiate(agent_cfg, _recursive_=False)
    initialize_for_inference(policy)
    if callable(getattr(policy, "policy_for_source", None)):
        policy = policy.policy_for_source(profile)
    return PolicyRuntime(policy, device=device)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _handler(runtime):
    class NeuroNCAPHandler(BaseHTTPRequestHandler):
        server_version = "UniDriveVLA/1"

        def log_message(self, format, *args):
            LOG.info("%s - %s", self.client_address[0], format % args)

        def _json(self, status, value):
            payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/alive":
                return self._json(404, {"error": "not found"})
            self._json(200, True)

        def do_POST(self):
            if self.path == "/reset":
                runtime.reset()
                return self._json(200, True)
            if self.path != "/infer":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= 128 * 1024 * 1024:
                    raise ValueError("Content-Length must be 1..128 MiB")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete HTTP request body")
                request = json.loads(raw, object_pairs_hook=_unique_object)
                inputs = canonical_input(request)
                trajectory = runtime.predict(**inputs)
                return self._json(200, {"trajectory": trajectory_to_ncap(trajectory), "aux_outputs": {}})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception:
                LOG.exception("NeuroNCAP inference failed")
                runtime.reset()
                return self._json(500, {"error": "inference failed; reset the model node before retrying"})

    return NeuroNCAPHandler


def serve(runtime, host="0.0.0.0", port=9000):
    if runtime.profile.name != "nuscenes":
        raise ValueError("NeuroNCAP's calibrated six-camera protocol requires the nuscenes profile")
    server = HTTPServer((host, int(port)), _handler(runtime))
    LOG.info("NeuroNCAP Model Node listening on %s:%d", host, port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        runtime.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve UniDriveVLA through the NeuroNCAP Model Node API")
    parser.add_argument("--config", default=str(PACKAGE / "config" / "serve.yaml"))
    parser.add_argument("--profile")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--vlm-path")
    parser.add_argument("--checkpoint-frame", choices=("ego", "lidar"),
                        help="verified frame for legacy checkpoints without frame metadata")
    parser.add_argument("--with-motion", action="store_true", default=None,
                        help="use the stage-2 architecture")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    profile = args.profile or cfg.profile
    if profile != "nuscenes":
        parser.error("NeuroNCAP's calibrated six-camera protocol requires --profile nuscenes")
    checkpoint = args.checkpoint or cfg.checkpoint
    device = args.device or cfg.device
    with_motion = cfg.with_motion if args.with_motion is None else args.with_motion
    checkpoint_frame = args.checkpoint_frame or OmegaConf.select(cfg, "checkpoint_frame")
    runtime = load_runtime(profile, checkpoint, device, with_motion, args.vlm_path, checkpoint_frame)
    serve(runtime, args.host or cfg.host, args.port or cfg.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
