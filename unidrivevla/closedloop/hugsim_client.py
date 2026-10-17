"""Trusted local FIFO client for the HUGSIM closed-loop simulator."""

import argparse
import errno
import io
import json
import logging
import os
import pickle
import select
import stat
import sys
import time
from pathlib import Path

import numpy as np

from unidrivevla.closedloop.hugsim import HUGSIMInputAdapter, load_calibration, trajectory_to_hugsim
from unidrivevla.serve import load_runtime

LOG = logging.getLogger("unidrivevla.hugsim")
MAX_FIFO_MESSAGE = 128 * 1024 * 1024


def _fifo_mode(path):
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise
    if not stat.S_ISFIFO(mode):
        raise ValueError(f"refusing non-FIFO simulator path: {path}")
    if hasattr(os, "geteuid") and os.stat(path, follow_symlinks=False).st_uid != os.geteuid():
        raise PermissionError(f"simulator FIFO is not owned by this user: {path}")


class PickleFIFOReader:
    """Read trusted simulator messages with an idle timeout and a size bound."""

    def __init__(self, path, timeout):
        _fifo_mode(path)
        flags = os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(path, flags)
        if not stat.S_ISFIFO(os.fstat(self.fd).st_mode):
            os.close(self.fd)
            raise ValueError(f"simulator path stopped being a FIFO: {path}")
        self.timeout = float(timeout)
        self.buffer = bytearray()

    def read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            if self.buffer:
                stream = io.BytesIO(self.buffer)
                try:
                    value = pickle.Unpickler(stream).load()
                except EOFError:
                    value = None
                except pickle.UnpicklingError as exc:
                    if "truncated" not in str(exc).lower() and "ran out of input" not in str(exc).lower():
                        raise
                    value = None
                if value is not None or stream.tell() > 0:
                    consumed = stream.tell()
                    if consumed <= 0:
                        raise pickle.UnpicklingError("empty pickle payload")
                    del self.buffer[:consumed]
                    return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for HUGSIM obs_pipe data")
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                raise TimeoutError("timed out waiting for HUGSIM obs_pipe data")
            data = os.read(self.fd, 1024 * 1024)
            if not data:
                raise EOFError("HUGSIM obs_pipe closed")
            self.buffer.extend(data)
            if len(self.buffer) > MAX_FIFO_MESSAGE:
                raise ValueError("HUGSIM FIFO message exceeds 128 MiB")

    def close(self):
        os.close(self.fd)


def write_pickle_fifo(path, value, timeout=30):
    _fifo_mode(path)
    payload = pickle.dumps(value, protocol=4)
    if len(payload) > 4096:
        raise ValueError("HUGSIM plan pickle unexpectedly exceeds PIPE_BUF")
    deadline = time.monotonic() + float(timeout)
    fd = None
    while fd is None:
        flags = os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)  # without O_TRUNC; type is checked again after opening
        except OSError as exc:
            if exc.errno != errno.ENXIO or time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for HUGSIM plan_pipe reader") from exc
            time.sleep(0.02)
    try:
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise ValueError(f"simulator path stopped being a FIFO: {path}")
        if os.write(fd, payload) != len(payload):
            raise OSError("short HUGSIM plan FIFO write")
    finally:
        os.close(fd)


def wait_for_fifos(obs_path, plan_path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exists = []
        for path in (obs_path, plan_path):
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                exists.append(False)
                continue
            if not stat.S_ISFIFO(mode):
                raise ValueError(f"refusing non-FIFO simulator path: {path}")
            exists.append(True)
        if all(exists):
            return
        time.sleep(0.1)
    raise TimeoutError("HUGSIM did not create obs_pipe and plan_pipe before timeout")


def _failure_marker(output, count, failures):
    root = Path(output)
    for index in range(10000):
        path = root / ("unidrivevla_failure.json" if index == 0 else f"unidrivevla_failure_{index}.json")
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"valid_model_run": False, "failure_count": count,
                           "failures": failures[:10]}, stream, indent=2)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("could not create a unique HUGSIM failure marker")


def run_fifo(output, runtime, adapter, timeout=3600):
    """Consume simulator messages until 'Done'; every received frame gets a waypoint reply."""
    output = Path(output)
    if not output.is_dir():
        raise NotADirectoryError(f"HUGSIM output directory does not exist: {output}")
    runtime.reset()
    obs_path, plan_path = output / "obs_pipe", output / "plan_pipe"
    wait_for_fifos(obs_path, plan_path, timeout)
    failed, failure_count = [], 0
    fatal = False
    saw_done = False
    reader = PickleFIFOReader(obs_path, timeout)
    try:
        while True:
            try:
                message = reader.read()  # simulator-only pickle IPC; never accept network-supplied pickle
            except TimeoutError as exc:
                LOG.error("HUGSIM observation stream timed out")
                failure_count += 1
                failed.append(str(exc))
                break
            except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
                LOG.exception("HUGSIM observation FIFO failed")
                failure_count += 1
                failed.append(f"{type(exc).__name__}: {exc}")
                break

            if isinstance(message, str) and message == "Done":
                saw_done = True
                break

            if fatal:
                plan = np.zeros((runtime.profile.horizon, 2), dtype=np.float64)
            else:
                try:
                    if not isinstance(message, tuple) or len(message) != 2:
                        raise ValueError("HUGSIM frame must be the official (obs, info) tuple")
                    inputs = adapter.convert(*message)
                    plan = trajectory_to_hugsim(runtime.predict(**inputs))
                except Exception as exc:
                    LOG.exception("HUGSIM inference failed; sending marked braking plans until episode end")
                    failure_count += 1
                    if len(failed) < 10:
                        failed.append(f"{type(exc).__name__}: {exc}")
                    fatal = True
                    runtime.reset()
                    plan = np.zeros((runtime.profile.horizon, 2), dtype=np.float64)

            try:
                write_pickle_fifo(plan_path, plan)
            except Exception as exc:
                LOG.exception("HUGSIM plan FIFO failed")
                failure_count += 1
                if len(failed) < 10:
                    failed.append(f"{type(exc).__name__}: {exc}")
                break
    finally:
        reader.close()

    runtime.reset()
    if not saw_done:
        failure_count += 1
        if len(failed) < 10:
            failed.append("simulator ended without Done")
    if failure_count or not saw_done:
        marker = _failure_marker(output, failure_count, failed)
        LOG.error("HUGSIM result is invalid; failure details: %s", marker)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="UniDriveVLA HUGSIM FIFO client")
    parser.add_argument("--output", default=os.environ.get("HUGSIM_OUTPUT"), required="HUGSIM_OUTPUT" not in os.environ)
    parser.add_argument("--checkpoint", default=os.environ.get("UNIDRIVE_CHECKPOINT"))
    parser.add_argument("--profile", default=os.environ.get("UNIDRIVE_PROFILE", "nuscenes"))
    parser.add_argument("--device", default=os.environ.get("UNIDRIVE_DEVICE", "cuda"))
    parser.add_argument("--checkpoint-frame", choices=("ego", "lidar"),
                        default=os.environ.get("UNIDRIVE_CHECKPOINT_FRAME"))
    parser.add_argument("--with-motion", action="store_true",
                        default=True if os.environ.get("UNIDRIVE_WITH_MOTION") == "1" else None)
    parser.add_argument("--camera-config", default=os.environ.get("HUGSIM_CAMERA_CONFIG"))
    parser.add_argument("--kinematic-config", default=os.environ.get("HUGSIM_KINEMATIC_CONFIG"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("HUGSIM_FIFO_TIMEOUT", "3600")))
    args = parser.parse_args(argv)
    if args.checkpoint_frame not in (None, "ego", "lidar"):
        parser.error("UNIDRIVE_CHECKPOINT_FRAME must be 'ego' or 'lidar'")
    if not args.camera_config or not args.kinematic_config:
        parser.error("--camera-config and --kinematic-config must point to the configs used by HUGSIM")
    if args.profile != "nuscenes":
        parser.error("HUGSIM's six nuScenes cameras require --profile nuscenes")
    cam_rect, wheelbase, expected_cameras = load_calibration(args.camera_config, args.kinematic_config)
    adapter = HUGSIMInputAdapter(cam_rect, wheelbase, expected_cameras)
    runtime = load_runtime(args.profile, args.checkpoint, args.device, args.with_motion,
                           checkpoint_frame=args.checkpoint_frame)
    return run_fifo(args.output, runtime, adapter, args.timeout)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
