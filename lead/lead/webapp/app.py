#!/usr/bin/env python3
"""Flask webapp for visualizing driving infractions from CARLA evaluations."""

import argparse
import json
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Default evaluation output directory
DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent.parent.parent / "outputs" / "local_evaluation"
)

# Read-only mode (disable file operations like open folder, cut video)
READ_ONLY_MODE = False


def load_infractions_data(infractions_file: Path) -> dict:
    """Load infractions data from JSON file.

    Handles both old format (list) and new format (object with 'infractions' and 'video_fps').

    Args:
        infractions_file: Path to infractions.json file

    Returns:
        Dictionary with 'infractions' (list), 'video_fps' (float), and 'is_legacy_format' (bool)
    """
    with open(infractions_file) as f:
        data = json.load(f)

    # Handle legacy format (just a list)
    if isinstance(data, list):
        return {
            "infractions": data,
            "video_fps": None,  # FPS not available in legacy format
            "is_legacy_format": True,
        }

    # Handle new format (object with infractions and video_fps)
    return {
        "infractions": data.get("infractions", []),
        "video_fps": data.get("video_fps"),
        "is_legacy_format": False,
    }


@app.route("/")
@app.route("/<path:output_dir>")
def index(output_dir=None):
    """Render main dashboard page."""
    return render_template("index.html", read_only_mode=READ_ONLY_MODE)


@app.route("/api/output_directories")
def list_output_directories():
    """List all output directories from outputs/evaluation."""
    base_path = Path(__file__).parent.parent.parent / "outputs" / "evaluation"

    if not base_path.exists():
        return jsonify(
            {"error": "Evaluation directory not found", "directories": []},
        ), 404

    directories = []

    # Iterate through experiment_name/benchmark_seed/timestamp structure
    for experiment_dir in sorted(base_path.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name

        for benchmark_dir in sorted(experiment_dir.iterdir()):
            if not benchmark_dir.is_dir():
                continue

            benchmark_seed = benchmark_dir.name

            for timestamp_dir in sorted(benchmark_dir.iterdir()):
                if not timestamp_dir.is_dir():
                    continue

                timestamp = timestamp_dir.name
                relative_path = (
                    f"outputs/evaluation/{experiment_name}/{benchmark_seed}/{timestamp}"
                )

                # Count total infractions in all routes within this directory
                total_infractions = 0
                has_data = False

                for route_dir in timestamp_dir.iterdir():
                    if not route_dir.is_dir():
                        continue

                    infractions_file = route_dir / "infractions.json"
                    if infractions_file.exists():
                        has_data = True
                        try:
                            infraction_data = load_infractions_data(infractions_file)
                            infractions = infraction_data["infractions"]
                            filtered_infractions = [
                                inf
                                for inf in infractions
                                if "minspeed" not in inf.get("infraction", "").lower()
                                and "completion"
                                not in inf.get("infraction", "").lower()
                            ]
                            total_infractions += len(filtered_infractions)
                        except Exception:
                            pass

                if has_data:
                    directories.append(
                        {
                            "path": relative_path,
                            "full_path": str(timestamp_dir),
                            "experiment": experiment_name,
                            "benchmark": benchmark_seed,
                            "timestamp": timestamp,
                            "infraction_count": total_infractions,
                        },
                    )

    return jsonify({"directories": directories, "base_path": str(base_path)})


@app.route("/api/routes")
def list_routes():
    """List all available evaluation routes."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    output_path = Path(output_dir)

    if not output_path.exists():
        return jsonify({"error": "Directory not found", "routes": []}), 404

    routes = []
    for route_dir in sorted(output_path.iterdir()):
        if route_dir.is_dir():
            route_name = route_dir.name
            # Check if it has the expected files (with route name prefix)
            has_infractions = (route_dir / "infractions.json").exists()
            has_debug = (route_dir / f"{route_name}_debug.mp4").exists()
            has_demo = (route_dir / f"{route_name}_demo.mp4").exists()
            has_grid = (route_dir / f"{route_name}_grid.mp4").exists()
            has_checkpoint = (route_dir / "checkpoint_endpoint.json").exists()

            if has_infractions or has_debug or has_demo or has_grid:
                route_info = {
                    "name": route_name,
                    "path": str(route_dir),
                    "has_infractions": has_infractions,
                    "has_debug_video": has_debug,
                    "has_demo_video": has_demo,
                    "has_grid_video": has_grid,
                    "has_checkpoint": has_checkpoint,
                }

                # Load infraction count (excluding min speed and completion)
                if has_infractions:
                    try:
                        infraction_data = load_infractions_data(
                            route_dir / "infractions.json",
                        )
                        infractions = infraction_data["infractions"]
                        # Filter out min speed and completion infractions
                        filtered_infractions = [
                            inf
                            for inf in infractions
                            if "minspeed" not in inf.get("infraction", "").lower()
                            and "completion" not in inf.get("infraction", "").lower()
                        ]
                        route_info["infraction_count"] = len(filtered_infractions)
                        route_info["video_fps"] = infraction_data["video_fps"]
                    except Exception:
                        route_info["infraction_count"] = 0
                        route_info["video_fps"] = None

                routes.append(route_info)

    return jsonify({"routes": routes, "output_dir": str(output_path)})


@app.route("/api/route/<path:route_name>/infractions")
def get_infractions(route_name):
    """Get infractions for a specific route."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    infractions_file = route_path / "infractions.json"

    if not infractions_file.exists():
        return jsonify(
            {
                "error": "Infractions file not found",
                "infractions": [],
                "video_fps": None,
            },
        ), 404

    try:
        infraction_data = load_infractions_data(infractions_file)
        return jsonify(
            {
                "infractions": infraction_data["infractions"],
                "video_fps": infraction_data["video_fps"],
                "is_legacy_format": infraction_data["is_legacy_format"],
            },
        )
    except Exception as e:
        return jsonify({"error": str(e), "infractions": [], "video_fps": None}), 500


@app.route("/api/route/<path:route_name>/checkpoint")
def get_checkpoint(route_name):
    """Get checkpoint data for a specific route."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    checkpoint_file = route_path / "checkpoint_endpoint.json"

    if not checkpoint_file.exists():
        return jsonify({"error": "Checkpoint file not found"}), 404

    try:
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        return jsonify(checkpoint)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/route/<path:route_name>/video_info")
def get_video_info(route_name):
    """Get video metadata including FPS from infractions.json."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    infractions_file = route_path / "infractions.json"

    if not infractions_file.exists():
        return jsonify({"error": "Infractions file not found", "video_fps": None}), 404

    try:
        infraction_data = load_infractions_data(infractions_file)
        return jsonify(
            {
                "video_fps": infraction_data["video_fps"],
                "is_legacy_format": infraction_data["is_legacy_format"],
            },
        )
    except Exception as e:
        return jsonify({"error": str(e), "video_fps": None}), 500


@app.route("/video/<path:route_name>/<video_type>")
def serve_video(route_name, video_type):
    """Serve video file for a specific route with range request support."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name

    # Video files are named with route prefix, e.g., 23687_debug.mp4
    video_files = {
        "debug": f"{route_name}_debug.mp4",
        "demo": f"{route_name}_demo.mp4",
        "grid": f"{route_name}_grid.mp4",
    }

    if video_type not in video_files:
        return "Invalid video type", 404

    video_file = video_files[video_type]
    video_path = route_path / video_file

    if not video_path.exists():
        return f"Video file {video_file} not found", 404

    # Support range requests for video seeking

    from flask import Response

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range", None)

    if range_header:
        byte_start, byte_end = 0, None
        match = range_header.replace("bytes=", "").split("-")
        byte_start = int(match[0])
        byte_end = int(match[1]) if match[1] else file_size - 1

        length = byte_end - byte_start + 1

        with open(video_path, "rb") as f:
            f.seek(byte_start)
            data = f.read(length)

        response = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        response.headers.add(
            "Content-Range",
            f"bytes {byte_start}-{byte_end}/{file_size}",
        )
        response.headers.add("Accept-Ranges", "bytes")
        response.headers.add("Content-Length", str(length))
        return response

    # Return full video if no range requested
    return send_from_directory(route_path, video_file, mimetype="video/mp4")


@app.route("/api/open_directory", methods=["POST"])
def open_directory():
    """Open the directory containing the route's videos in the file manager."""
    if READ_ONLY_MODE:
        return jsonify({"error": "File operations disabled in read-only mode"}), 403

    data = request.json
    route_name = data.get("route_name")
    output_dir = data.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not route_name:
        return jsonify({"error": "Missing route_name parameter"}), 400

    route_path = Path(output_dir) / route_name

    if not route_path.exists():
        return jsonify({"error": f"Directory not found: {route_path}"}), 404

    try:
        # Open directory in file manager (works on Linux, Mac, Windows)
        import platform

        system = platform.system()

        if system == "Linux":
            subprocess.run(["xdg-open", str(route_path)], check=True)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(route_path)], check=True)
        elif system == "Windows":
            subprocess.run(["explorer", str(route_path)], check=True)
        else:
            return jsonify({"error": f"Unsupported operating system: {system}"}), 400

        return jsonify({"success": True, "path": str(route_path)})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to open directory: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cut_video", methods=["POST"])
def cut_video():
    """Cut a video segment around an infraction timestamp."""
    if READ_ONLY_MODE:
        return jsonify({"error": "File operations disabled in read-only mode"}), 403

    data = request.json
    route_name = data.get("route_name")
    video_type = data.get("video_type")
    timestamp = data.get("timestamp")
    buffer_seconds = data.get("buffer", 3)
    infraction_number = data.get("infraction_number")
    output_dir = data.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not all([route_name, video_type, timestamp is not None, infraction_number]):
        return jsonify({"error": "Missing required parameters"}), 400

    route_path = Path(output_dir) / route_name

    # Get input video file
    video_files = {
        "debug": f"{route_name}_debug.mp4",
        "demo": f"{route_name}_demo.mp4",
        "grid": f"{route_name}_grid.mp4",
    }

    if video_type not in video_files:
        return jsonify({"error": "Invalid video type"}), 400

    input_video = route_path / video_files[video_type]

    if not input_video.exists():
        return jsonify(
            {"error": f"Video file {video_files[video_type]} not found"},
        ), 404

    # Calculate start and duration
    start_time = max(0, timestamp - buffer_seconds)
    duration = buffer_seconds * 2  # buffer before + buffer after

    # Output file name - save to desktop as infraction.mp4
    desktop_path = Path.home() / "Desktop"
    output_path = desktop_path / "infraction.mp4"

    try:
        # Use ffmpeg to cut and re-encode video for Notion compatibility
        # Comprehensive settings for maximum web/Notion compatibility
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-ss",
            str(start_time),  # Start time
            "-i",
            str(input_video),  # Input file
            "-t",
            str(duration),  # Duration
            "-vcodec",
            "libx264",  # H.264 video codec
            "-pix_fmt",
            "yuv420p",  # Pixel format (critical for web compatibility)
            "-profile:v",
            "baseline",  # Use baseline profile for maximum compatibility
            "-level",
            "3.0",  # H.264 level
            "-movflags",
            "+faststart",  # Move moov atom to beginning for streaming
            "-acodec",
            "aac",  # AAC audio codec
            "-ar",
            "44100",  # Audio sample rate
            "-strict",
            "experimental",  # Allow experimental encoders
            str(output_path),  # Output file
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        return jsonify(
            {
                "success": True,
                "output_path": str(output_path),
                "start_time": start_time,
                "duration": duration,
                "buffer": buffer_seconds,
            },
        )

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg error: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cut_custom_video", methods=["POST"])
def cut_custom_video():
    """Cut a custom video segment between start and stop times."""
    data = request.json
    route_name = data.get("route_name")
    video_type = data.get("video_type")
    start_time = data.get("start_time")
    stop_time = data.get("stop_time")
    output_dir = data.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not all([route_name, video_type, start_time is not None, stop_time is not None]):
        return jsonify({"error": "Missing required parameters"}), 400

    route_path = Path(output_dir) / route_name

    # Get input video file
    video_files = {
        "debug": f"{route_name}_debug.mp4",
        "demo": f"{route_name}_demo.mp4",
        "grid": f"{route_name}_grid.mp4",
    }

    if video_type not in video_files:
        return jsonify({"error": "Invalid video type"}), 400

    input_video = route_path / video_files[video_type]

    if not input_video.exists():
        return jsonify({"error": "Video file not found"}), 404

    # Find next available number for output file
    desktop_path = Path.home() / "Desktop"
    desktop_path.mkdir(exist_ok=True)

    counter = 1
    while True:
        output_filename = f"{counter:03d}.mp4"
        output_path = desktop_path / output_filename
        if not output_path.exists():
            break
        counter += 1

    duration = stop_time - start_time

    try:
        import subprocess

        # Use ffmpeg with re-encoding for precise cuts and consistent quality
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-ss",
            str(start_time),
            "-i",
            str(input_video),
            "-t",
            str(duration),
            "-c:v",
            "libx264",  # H.264 video codec
            "-crf",
            "18",  # High quality (0-51, lower is better, 18 is visually lossless)
            "-preset",
            "fast",  # Faster preset (fast, medium, slow)
            "-c:a",
            "aac",  # AAC audio codec
            "-b:a",
            "192k",  # Audio bitrate
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)

        return jsonify(
            {
                "success": True,
                "output_path": str(output_path),
                "filename": output_filename,
                "duration": duration,
            },
        )

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"ffmpeg failed: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infraction Dashboard - CARLA Evaluation Viewer",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Enable read-only mode (disable file operations like open folder, cut video)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    READ_ONLY_MODE = args.read_only

    print("Starting Infraction Dashboard...")
    print(f"Default output directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Read-only mode: {'ENABLED' if READ_ONLY_MODE else 'DISABLED'}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(debug=True, port=args.port, host=args.host)
