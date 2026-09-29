"""Video recording and processing utilities for CARLA agent evaluation."""

import copy
import logging
import os

import carla
import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import PIL.Image
import torch
from beartype import beartype

from lead.common import common_utils
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)

DEMO_CAMERAS = [
    {
        "name": "cinematic_camera",
        "draw_target_points": False,
        "draw_planning": False,
        "image_size_x": "960",
        "image_size_y": "1080",
        "fov": "100",
        "x": -6.5,
        "y": -0.0,
        "z": 6.0,
        "pitch": -30.0,
        "yaw": 0.0,
    },
    {
        "name": "bev_camera",
        "draw_target_points": True,
        "draw_planning": True,
        "image_size_x": "960",
        "image_size_y": "1080",
        "fov": "100",
        "x": 0.0,
        "y": 0.0,
        "z": 22.0,
        "pitch": -90.0,
        "yaw": 0.0,
    },
]


class VideoRecorder:
    """Handles all video recording and processing for agent evaluation.

    This class manages:
    - Demo camera setup and positioning
    - Video writer initialization
    - Frame capture and processing
    - Video compression using ffmpeg
    - Waypoint and target point visualization
    """

    @beartype
    def __init__(
        self,
        config_closed_loop: ClosedLoopConfig,
        vehicle: carla.Actor,
        world: carla.World,
        step_counter: int = -1,
        training_config: TrainingConfig | None = None,
    ):
        """Initialize video recorder.

        Args:
            config_closed_loop: Configuration for closed-loop inference
            vehicle: CARLA ego vehicle actor
            world: CARLA world instance
            step_counter: Initial step counter value
            training_config: Training configuration containing camera calibration
        """
        self.config = config_closed_loop
        self.vehicle = vehicle
        self.world = world
        self.step = step_counter
        self.training_config = training_config

        # Video writers
        self.debug_video_writer = None
        self.input_video_writer = None
        self.demo_video_writer = None
        self.grid_video_writer = None

        # Demo cameras
        self._demo_cameras = []
        self._demo_camera_images = {}

        # Store last images for grid creation
        self._last_demo_image = None
        self._last_input_image = None

        # Initialize demo cameras if needed
        if self.config.save_path is not None and (
            self.config.produce_demo_video or self.config.produce_demo_image
        ):
            self._setup_demo_cameras()

    @beartype
    def _setup_demo_cameras(self) -> None:
        """Set up demo cameras for cinematic and BEV views."""
        bp_lib = self.world.get_blueprint_library()
        for idx, camera_config in enumerate(DEMO_CAMERAS, start=1):
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", camera_config["image_size_x"])
            camera_bp.set_attribute("image_size_y", camera_config["image_size_y"])
            camera_bp.set_attribute("fov", camera_config["fov"])
            camera_bp.set_attribute("motion_blur_intensity", "0.0")

            # Create transform for this demo camera
            demo_camera_location = carla.Location(
                x=camera_config["x"],
                y=camera_config["y"],
                z=camera_config["z"],
            )
            world_camera_location = common_utils.get_world_coordinate_2d(
                self.vehicle.get_transform(),
                demo_camera_location,
            )
            demo_camera_transform = carla.Transform(
                world_camera_location,
                carla.Rotation(
                    pitch=camera_config["pitch"],
                    yaw=self.vehicle.get_transform().rotation.yaw
                    + camera_config["yaw"],
                ),
            )

            demo_camera = self.world.spawn_actor(camera_bp, demo_camera_transform)

            # Create callback to store image in buffer
            def _make_image_callback(camera_idx):
                def _store_image(image):
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = copy.deepcopy(array)
                    array = np.reshape(array, (image.height, image.width, 4))
                    bgr = array[:, :, :3]
                    self._demo_camera_images[camera_idx] = bgr

                return _store_image

            demo_camera.listen(_make_image_callback(idx))
            self._demo_cameras.append(
                {
                    "camera": demo_camera,
                    "config": camera_config,
                    "index": idx,
                },
            )
        LOG.info(f"[VideoRecorder] Initialized {len(self._demo_cameras)} demo cameras")

    @beartype
    def update_step(self, step: int) -> None:
        """Update the current step counter.

        Args:
            step: Current step number
        """
        self.step = step

    @beartype
    def move_demo_cameras_with_ego(self) -> None:
        """Update demo camera transforms to follow ego vehicle position and orientation."""
        if self.config.save_path is None or not (
            self.config.produce_demo_video or self.config.produce_demo_image
        ):
            return

        for demo_cam_info in self._demo_cameras:
            if demo_cam_info["camera"].is_alive:
                camera_config = demo_cam_info["config"]
                demo_camera_location = carla.Location(
                    x=camera_config["x"],
                    y=camera_config["y"],
                    z=camera_config["z"],
                )
                world_camera_location = common_utils.get_world_coordinate_2d(
                    self.vehicle.get_transform(),
                    demo_camera_location,
                )
                demo_camera_transform = carla.Transform(
                    world_camera_location,
                    carla.Rotation(
                        pitch=camera_config["pitch"],
                        yaw=self.vehicle.get_transform().rotation.yaw
                        + camera_config["yaw"],
                    ),
                )
                demo_cam_info["camera"].set_transform(demo_camera_transform)

    @beartype
    def save_demo_cameras(
        self,
        pred_waypoints: jt.Float[torch.Tensor, "n_waypoints 2"] | None = None,
        target_points: dict[str, jt.Float[npt.NDArray, " 2"] | None] | None = None,
    ) -> None:
        """Save concatenated demo cameras (cinematic + BEV) as single JPG/video.

        Args:
            pred_waypoints: Waypoints in vehicle coords, shape (n_waypoints, 2) with (x, y) in meters.
            target_points: Route targets {'previous': (x,y), 'current': (x,y), 'next': (x,y)}.
        """
        if self.config.save_path is None or not (
            self.config.produce_demo_video or self.config.produce_demo_image
        ):
            return

        processed_images = []
        for camera_idx in sorted(self._demo_camera_images.keys()):
            image = self._demo_camera_images[camera_idx]
            camera_config = DEMO_CAMERAS[camera_idx - 1]  # camera_idx is 1-based
            camera_name = camera_config.get("name", f"demo_{camera_idx}")
            draw_target_points = camera_config.get("draw_target_points", False)
            draw_planning = camera_config.get("draw_planning", False)

            processed_image = image.copy()

            # Add visualizations if enabled
            if draw_planning and pred_waypoints is not None:
                processed_image = self.draw_waypoints(
                    processed_image,
                    pred_waypoints,
                    camera_config,
                )
            if (
                draw_target_points
                and target_points is not None
                and camera_name == "bev_camera"
            ):
                processed_image = self.draw_target_points(
                    processed_image,
                    target_points,
                    camera_config,
                    is_bev=True,
                )

            processed_images.append(processed_image)

        # Concatenate horizontally: [cinematic | BEV]
        if not processed_images:
            LOG.warning(
                "[VideoRecorder] No demo camera images available yet, skipping frame %d",
                self.step,
            )
            return
        concatenated = np.hstack(processed_images)

        # Store for grid creation (only if grid features are enabled)
        if self.config.produce_grid_image or self.config.produce_grid_video:
            self._last_demo_image = concatenated

        # Save as PNG for demo (higher quality for presentation)
        if self.config.produce_demo_image:
            save_path_demo = str(self.config.save_path / "demo_images")
            os.makedirs(save_path_demo, exist_ok=True)
            PIL.Image.fromarray(cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)).save(
                f"{save_path_demo}/{str(self.step).zfill(5)}.png",
                optimize=False,
                compress_level=0,  # Really space expensive, do this local only.
            )

        # Add to demo video if enabled
        if (
            self.config.produce_demo_video
            and self.step % self.config.produce_frame_frequency == 0
        ):
            if self.demo_video_writer is None:
                os.makedirs(os.path.dirname(self.config.demo_video_path), exist_ok=True)
                self.demo_video_writer = cv2.VideoWriter(
                    self.config.demo_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.config.video_fps,
                    (concatenated.shape[1], concatenated.shape[0]),
                )
            self.demo_video_writer.write(concatenated)

    @beartype
    def draw_waypoints(
        self,
        image: jt.UInt8[npt.NDArray, "height width 3"],
        pred_waypoints: jt.Float[torch.Tensor, "n_waypoints 2"],
        camera_config: dict[str, str | float | bool],
    ) -> jt.UInt8[npt.NDArray, "height width 3"]:
        """Project and draw waypoints from vehicle to image coordinates.

        Args:
            image: BGR image, shape (height, width, 3).
            pred_waypoints: Waypoints in vehicle coords, shape (n_waypoints, 2) with (x, y) in meters.
            camera_config: Camera params {'x','y','z','pitch','yaw','fov'}.

        Returns:
            Image copy with yellow waypoints and connecting lines.
        """
        img_with_viz = image.copy()
        camera_height = image.shape[0]
        camera_width = image.shape[1]

        # Extract camera parameters from config
        camera_fov = float(camera_config["fov"])
        camera_pos = [camera_config["x"], camera_config["y"], camera_config["z"]]
        camera_rot = [
            camera_config.get("roll", 0.0),
            camera_config["pitch"],
            camera_config["yaw"],
        ]  # roll, pitch, yaw

        # Draw route in blue
        if pred_waypoints is not None and len(pred_waypoints) > 0:
            route_points = pred_waypoints.detach().cpu().float().numpy()
            projected_route, points_inside_image = common_utils.project_points_to_image(
                camera_rot,
                camera_pos,
                camera_fov,
                camera_width,
                camera_height,
                route_points,
            )

            # Draw circles for waypoints
            for pt, inside in zip(projected_route, points_inside_image, strict=True):
                if inside:
                    cv2.circle(
                        img_with_viz,
                        (int(pt[0]), int(pt[1])),
                        radius=3,
                        color=(255, 255, 0),
                        thickness=-1,  # Red in BGR
                        lineType=cv2.LINE_AA,
                    )
            # # Draw connected line for route
            for i in range(len(projected_route) - 1):
                pt1, inside1 = projected_route[i], points_inside_image[i]
                pt2, inside2 = projected_route[i + 1], points_inside_image[i + 1]
                if inside1 and inside2:
                    cv2.line(
                        img_with_viz,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (255, 255, 0),  # Blue in BGR
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

        return img_with_viz

    @beartype
    def draw_target_points(
        self,
        image: jt.UInt8[npt.NDArray, "height width 3"],
        target_points: dict[str, jt.Float[npt.NDArray, " 2"] | None],
        camera_config: dict[str, str | float | bool],
        is_bev: bool = False,
    ) -> jt.UInt8[npt.NDArray, "height width 3"]:
        """Project and draw route target points (previous/current/next) as red circles.

        Args:
            image: BGR image, shape (height, width, 3).
            target_points: Route targets {'previous': (x,y), 'current': (x,y), 'next': (x,y)} in vehicle coords (meters).
            camera_config: Camera params {'x','y','z','pitch','yaw','fov'}.
            is_bev: If True, use fixed pixel size. If False, calculate radius from 10cm physical size.

        Returns:
            Image copy with red target point circles.
        """
        img_with_targets = image.copy()
        camera_height = image.shape[0]
        camera_width = image.shape[1]

        # Extract camera parameters from config
        camera_fov = float(camera_config["fov"])
        camera_pos = [camera_config["x"], camera_config["y"], camera_config["z"]]
        camera_rot = [
            camera_config.get("roll", 0.0),
            camera_config["pitch"],
            camera_config["yaw"],
        ]

        # Define colors and sizes for each target point (BGR format)
        targets_config = [
            ("previous", (0, 0, 255)),
            ("current", (0, 0, 255)),
            ("next", (0, 0, 255)),
        ]

        for key, color in targets_config:
            if key in target_points and target_points[key] is not None:
                # Get target point in vehicle coordinates
                target_point = np.array(
                    [[target_points[key][0], target_points[key][1]]],
                )

                # Project center point to image
                projected, points_inside_image = common_utils.project_points_to_image(
                    camera_rot,
                    camera_pos,
                    camera_fov,
                    camera_width,
                    camera_height,
                    target_point,
                )

                if len(projected) > 0:
                    pt, inside = projected[0], points_inside_image[0]
                    if inside:
                        x, y = int(pt[0]), int(pt[1])

                        if is_bev:
                            # For BEV camera: use fixed pixel size
                            pixel_radius = 3  # Fixed size for BEV view
                        else:
                            # For input camera: calculate radius based on physical size using perspective projection
                            sphere_radius_meters = 0.05

                            # Calculate 3D distance from camera to target point
                            target_3d = np.array(
                                [target_points[key][0], target_points[key][1], 0.0],
                            )
                            camera_3d = np.array(camera_pos)
                            distance = np.linalg.norm(target_3d - camera_3d)

                            # Calculate pixel radius using perspective projection
                            # pixel_size = (object_size * focal_length) / distance
                            # focal_length ≈ (image_width / 2) / tan(fov/2)
                            focal_length = (camera_width / 2.0) / np.tan(
                                np.radians(camera_fov / 2.0),
                            )
                            pixel_radius = int(
                                (sphere_radius_meters * focal_length) / distance,
                            )
                            pixel_radius = max(
                                1,
                                min(pixel_radius, 50),
                            )  # Clamp between 1 and 50 pixels

                        if pixel_radius > 0:
                            # Draw white border
                            cv2.circle(
                                img_with_targets,
                                (x, y),
                                pixel_radius + 1,
                                (255, 255, 255),
                                thickness=-1,
                                lineType=cv2.LINE_AA,
                            )

                            # Filled colored circle
                            cv2.circle(
                                img_with_targets,
                                (x, y),
                                pixel_radius,
                                color,
                                thickness=-1,
                                lineType=cv2.LINE_AA,
                            )

        return img_with_targets

    @beartype
    def save_input_image(self, input_image: npt.NDArray) -> None:
        """Save input image as PNG.

        Args:
            input_image: BGR input image from camera
        """
        # Store for grid creation (only if grid features are enabled)
        if self.config.produce_grid_image or self.config.produce_grid_video:
            self._last_input_image = input_image

        if (
            self.config.save_path is None
            or not self.config.produce_input_image
            or self.step % self.config.produce_frame_frequency != 0
        ):
            return

        save_path_input = str(self.config.save_path / "input_images")
        os.makedirs(save_path_input, exist_ok=True)
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        PIL.Image.fromarray(input_image_rgb).save(
            f"{save_path_input}/{str(self.step).zfill(5)}.png",
            optimize=True,
            compress_level=0,
        )

    @beartype
    def save_input_video_frame(self, input_image: npt.NDArray) -> None:
        """Add frame to input video.

        Args:
            input_image: BGR input image from camera
        """
        if (
            self.config.save_path is None
            or not self.config.produce_input_video
            or self.step % self.config.produce_frame_frequency != 0
        ):
            return

        if self.input_video_writer is None:
            os.makedirs(os.path.dirname(self.config.input_video_path), exist_ok=True)
            self.input_video_writer = cv2.VideoWriter(
                self.config.input_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.video_fps,
                (input_image.shape[1], input_image.shape[0]),
            )
        self.input_video_writer.write(input_image)

    @beartype
    def save_debug_image(self, image: npt.NDArray) -> None:
        """Save debug visualization image as PNG.

        Args:
            image: RGB debug visualization image
        """
        if (
            self.config.save_path is None
            or not self.config.produce_debug_image
            or self.step % self.config.produce_frame_frequency != 0
        ):
            return

        save_dir = self.config.save_path / "debug_images"
        os.makedirs(save_dir, exist_ok=True)
        PIL.Image.fromarray(image).save(
            f"{save_dir}/{str(self.step).zfill(5)}.png",
            optimize=False,
            compress_level=0,
        )

    @beartype
    def save_debug_video_frame(self, image: npt.NDArray) -> None:
        """Add frame to debug video.

        Args:
            image: RGB debug visualization image
        """
        if (
            self.config.save_path is None
            or not self.config.produce_debug_video
            or self.step % self.config.produce_frame_frequency != 0
        ):
            return

        if self.debug_video_writer is None:
            os.makedirs(os.path.dirname(self.config.debug_video_path), exist_ok=True)
            self.debug_video_writer = cv2.VideoWriter(
                str(self.config.debug_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.video_fps,
                (image.shape[1], image.shape[0]),
            )

        self.debug_video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @beartype
    def save_grid_image_and_video(
        self,
        demo_image: npt.NDArray | None = None,
        input_image: npt.NDArray | None = None,
        pred_waypoints: jt.Float[torch.Tensor, "n_waypoints 2"] | None = None,
        target_points: dict[str, jt.Float[npt.NDArray, " 2"] | None] | None = None,
    ) -> None:
        """Save grid layout with demo and input images stacked vertically.

        Creates a grid by:
        - Cropping x% from top and bottom of demo image
        - Cropping x% from top and bottom of input image
        - Resizing input image width to match demo image width
        - Optionally drawing waypoints and target points on input image
        - Stacking them vertically

        Args:
            demo_image: BGR demo image (concatenated cinematic + BEV). If None, uses last stored demo image.
            input_image: BGR input image from camera. If None, uses last stored input image.
            pred_waypoints: Waypoints in vehicle coords, shape (n_waypoints, 2) with (x, y) in meters.
            target_points: Route targets {'previous': (x,y), 'current': (x,y), 'next': (x,y)}.
        """
        if (
            self.config.save_path is None
            or (
                not self.config.produce_grid_image
                and not self.config.produce_grid_video
            )
            or self.step % self.config.produce_frame_frequency != 0
        ):
            return

        # Use stored images if not provided
        if demo_image is None:
            demo_image = self._last_demo_image
        if input_image is None:
            input_image = self._last_input_image

        # Check if we have both images
        if demo_image is None or input_image is None:
            return

        # Draw waypoints and target points on input image if provided
        input_with_viz = input_image.copy()
        if (
            pred_waypoints is not None or target_points is not None
        ) and self.training_config is not None:
            # The input image is stitched from multiple cameras horizontally
            # We need to draw on each camera section separately with correct calibration
            input_height, input_width = input_image.shape[:2]
            num_cameras = self.training_config.num_cameras
            camera_width = input_width // num_cameras

            assert self.training_config.carla_leaderboard_mode, (
                "Grid video is currently supported for CARLA leaderboard mode."
            )
            # Mapping from image section index to calibration index
            # Image order: [LEFT_FRONT, CENTER_FRONT, RIGHT_FRONT, RIGHT_REAR, CENTER_REAR, LEFT_REAR]
            # Calib order: [RIGHT_FRONT, CENTER_FRONT, LEFT_FRONT, LEFT_REAR, CENTER_REAR, RIGHT_REAR]
            if num_cameras == 6:
                section_to_calib = [
                    3,
                    2,
                    1,
                    6,
                    5,
                    4,
                ]  # Map section index to calibration index
            elif num_cameras == 3:
                section_to_calib = [3, 2, 1]  # [LEFT_FRONT, CENTER_FRONT, RIGHT_FRONT]
            else:
                section_to_calib = list(
                    range(1, num_cameras + 1),
                )  # Fallback: assume they match

            for section_idx in range(num_cameras):
                # Get calibration index for this image section
                calib_idx = section_to_calib[section_idx]

                # Get exact camera config for this camera
                cam_config = self.training_config.camera_calibration[calib_idx]
                camera_config = {
                    "x": cam_config["pos"][0],
                    "y": cam_config["pos"][1],
                    "z": cam_config["pos"][2],
                    "roll": cam_config["rot"][0],
                    "pitch": cam_config["rot"][1],
                    "yaw": cam_config["rot"][2],
                    "fov": str(cam_config["fov"]),
                }

                # Extract this camera's section from the stitched image
                x_start = section_idx * camera_width
                x_end = (section_idx + 1) * camera_width
                camera_section = input_with_viz[:, x_start:x_end, :].copy()

                # Draw visualizations on this camera section
                if pred_waypoints is not None:
                    camera_section = self.draw_waypoints(
                        camera_section,
                        pred_waypoints,
                        camera_config,
                    )
                if target_points is not None:
                    camera_section = self.draw_target_points(
                        camera_section,
                        target_points,
                        camera_config,
                        is_bev=False,
                    )

                # Put the modified section back into the stitched image
                input_with_viz[:, x_start:x_end, :] = camera_section

        # Process demo image: crop 20% from top and bottom
        demo_h, demo_w = demo_image.shape[:2]
        crop_demo_top = int(demo_h * 0.25)
        crop_demo_bottom = int(demo_h * 0.25)
        demo_cropped = demo_image[crop_demo_top : demo_h - crop_demo_bottom, :]

        # Process input image: crop 10% from top and bottom (use visualized version)
        input_h, input_w = input_with_viz.shape[:2]
        crop_input_top = int(input_h * 0.078125)
        crop_input_bottom = int(input_h * 0.078125)
        input_cropped = input_with_viz[crop_input_top : input_h - crop_input_bottom, :]

        # Resize demo image down to match input image width while maintaining aspect ratio
        demo_cropped_h, demo_cropped_w = demo_cropped.shape[:2]
        input_cropped_w = input_cropped.shape[1]
        target_width = input_cropped_w
        aspect_ratio = demo_cropped_h / demo_cropped_w
        target_height = int(target_width * aspect_ratio)
        demo_resized = cv2.resize(
            demo_cropped,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,  # Better for downscaling
        )

        # Stack vertically: demo on top, input on bottom
        grid_image = np.vstack([demo_resized, input_cropped])

        # Save as PNG for grid image
        if self.config.produce_grid_image:
            save_path_grid = str(self.config.save_path / "grid_images")
            os.makedirs(save_path_grid, exist_ok=True)
            PIL.Image.fromarray(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)).save(
                f"{save_path_grid}/{str(self.step).zfill(5)}.png",
                optimize=False,
                compress_level=0,
            )

        # Add to grid video
        if self.config.produce_grid_video:
            if self.grid_video_writer is None:
                os.makedirs(os.path.dirname(self.config.grid_video_path), exist_ok=True)
                self.grid_video_writer = cv2.VideoWriter(
                    self.config.grid_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.config.video_fps,
                    (grid_image.shape[1], grid_image.shape[0]),
                )
            self.grid_video_writer.write(grid_image)

    @beartype
    def compress_video(
        self,
        temp_path: str,
        final_path: str,
        crf: int,
        preset: str,
    ) -> None:
        """Compress a video using ffmpeg.

        Args:
            temp_path: Path to the uncompressed video.
            final_path: Path to save the compressed video.
            crf: Constant Rate Factor for ffmpeg compression (lower is better quality).
            preset: Preset for ffmpeg compression speed/quality trade-off.
        """
        # Check if ffmpeg is installed
        command = f"ffmpeg -i {final_path} -c:v libx264 -crf {crf} -preset {preset} -an {temp_path} -y"
        os.system(command)
        os.replace(temp_path, final_path)
        LOG.info(f"Compressed video: {final_path}")

    @beartype
    def cleanup_and_compress(self) -> None:
        """Release all video writers and compress videos."""
        # Clean up demo cameras
        if hasattr(self, "_demo_cameras"):
            for demo_cam_info in self._demo_cameras:
                if demo_cam_info["camera"].is_alive:
                    demo_cam_info["camera"].stop()
                    demo_cam_info["camera"].destroy()
                    LOG.info(f"Destroyed demo camera {demo_cam_info['index']}")

        # Clean up video writers
        if self.config.save_path is not None:
            # Input video - high quality for presentation
            if self.config.produce_input_video and self.input_video_writer is not None:
                self.input_video_writer.release()
                self.compress_video(
                    temp_path=os.path.splitext(self.config.input_video_path)[0]
                    + "_temp.mp4",
                    final_path=self.config.input_video_path,
                    crf=18,
                    preset="slow",
                )

            if self.config.produce_debug_video and self.debug_video_writer is not None:
                # Debug video - low quality for disk space
                self.debug_video_writer.release()
                self.compress_video(
                    temp_path=os.path.splitext(self.config.debug_video_path)[0]
                    + "_temp.mp4",
                    final_path=self.config.debug_video_path,
                    crf=28,
                    preset="slower",
                )

            # Demo video - high quality for presentation
            if self.config.produce_demo_video and self.demo_video_writer is not None:
                self.demo_video_writer.release()
                self.compress_video(
                    temp_path=os.path.splitext(self.config.demo_video_path)[0]
                    + "_temp.mp4",
                    final_path=self.config.demo_video_path,
                    crf=18,
                    preset="slow",
                )

            # Grid video - high quality for presentation
            if self.config.produce_grid_video and self.grid_video_writer is not None:
                self.grid_video_writer.release()
                self.compress_video(
                    temp_path=os.path.splitext(self.config.grid_video_path)[0]
                    + "_temp.mp4",
                    final_path=self.config.grid_video_path,
                    crf=18,
                    preset="slow",
                )
