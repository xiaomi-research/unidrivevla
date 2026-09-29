from __future__ import annotations

import io
import lzma
import os
import pickle
from dataclasses import dataclass

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype

from lead.common import common_utils
from lead.training.config_training import TrainingConfig


@jt.jaxtyped(typechecker=beartype)
@dataclass
class SensorData:
    image: jt.UInt8[npt.NDArray, "img_height img_width 3"] | None
    rasterized_lidar: jt.Float32[npt.NDArray, "bev_height bev_width"] | None
    semantic: jt.UInt8[npt.NDArray, "img_height img_width"] | None
    hdmap: jt.UInt8[npt.NDArray, "bev_semantic_height bev_semantic_width"] | None
    depth: jt.Float32[npt.NDArray, "depth_img_height depth_img_width"] | None
    boxes: jt.Float32[npt.NDArray, "num_boxes features"] | None
    boxes_waypoints: jt.Float32[npt.NDArray, "num_boxes timesteps 2"] | None
    boxes_num_waypoints: jt.Int32[npt.NDArray, " num_boxes"] | None
    bev_occupancy: (
        jt.UInt8[npt.NDArray, "bev_occupancy_height bev_occupancy_width"] | None
    )
    bev_3rd_person_image: jt.UInt8[npt.NDArray, "bev_img_height bev_img_width 3"] | None
    radars: (
        tuple[
            jt.Float16[npt.NDArray, "num_points_1 4"],
            jt.Float16[npt.NDArray, "num_points_2 4"],
            jt.Float16[npt.NDArray, "num_points_3 4"],
            jt.Float16[npt.NDArray, "num_points_4 4"],
        ]
        | None
    )
    radar_detections: jt.Float32[npt.NDArray, "num_radar_queries radar_features"] | None

    @beartype
    def compress(
        self,
        raw_image_bytes,
        config: TrainingConfig,
        current_measurement: dict,
    ) -> CompressedSensorData:
        # LiDAR BEV
        compressed_lidar_bev = None
        if self.rasterized_lidar is not None:
            compressed_lidar_bev = compress_float_image(self.rasterized_lidar, config)

        # Semantic
        compressed_semantic = None
        if self.semantic is not None:
            compressed_semantic = compress_integer_image_lossless(self.semantic, config)

        # BEV semantic
        compressed_bev_semantic = None
        if self.hdmap is not None:
            compressed_bev_semantic = compress_integer_image_lossless(
                self.hdmap,
                config,
            )

        # Depth
        compressed_depth = None
        if self.depth is not None:
            encoded_depth = common_utils.encode_depth_8bit(self.depth)
            compressed_depth = compress_integer_image_lossless(encoded_depth, config)

        # BEV occupancy
        compressed_bev_occupancy = None
        if self.bev_occupancy is not None:
            compressed_bev_occupancy = compress_integer_image_lossless(
                self.bev_occupancy,
                config,
            )

        # BEV 3rd person image
        compressed_bev_3rd_person_image = None
        if self.bev_3rd_person_image is not None:
            compressed_bev_3rd_person_image = compress_integer_image_lossy(
                self.bev_3rd_person_image,
                current_measurement["jpeg_storage_quality"],
            )

        # Radars
        compressed_radars = None
        if self.radars is not None:
            compressed_radars_list = []
            for radar in self.radars:
                compressed_radars_list.append(compress_radar_lossless(radar))
            compressed_radars = tuple(compressed_radars_list)

        return CompressedSensorData(
            image=raw_image_bytes,
            lidar_bev=compressed_lidar_bev,
            semantic=compressed_semantic,
            bev_semantic=compressed_bev_semantic,
            depth=compressed_depth,
            bboxes=self.boxes,  # Boxes are not compressed
            bboxes_waypoints=self.boxes_waypoints,  # Boxes waypoints are not compressed
            bboxes_num_waypoints=self.boxes_num_waypoints,  # Boxes num waypoints are not compressed
            bev_occupancy=compressed_bev_occupancy,
            bev_3rd_person_image=compressed_bev_3rd_person_image,
            radars=compressed_radars,
            radar_detections=self.radar_detections,  # Radar detections are not compressed
        )


@jt.jaxtyped(typechecker=beartype)
@dataclass
class CompressedSensorData:
    """Compressed version of SensorData for efficient storage."""

    image: bytes | None  # JPEG compressed RGB image
    lidar_bev: (
        jt.UInt8[npt.NDArray, " lidar_bytes"] | None
    )  # PNG compressed float image
    semantic: (
        jt.UInt8[npt.NDArray, " semantic_bytes"] | None
    )  # PNG compressed integer image
    bev_semantic: (
        jt.UInt8[npt.NDArray, " bev_semantic_bytes"] | None
    )  # PNG compressed integer image
    depth: (
        jt.UInt8[npt.NDArray, " depth_bytes"] | None
    )  # PNG compressed 8-bit encoded depth
    bboxes: jt.Float32[npt.NDArray, "num_boxes features"] | None  # Uncompressed boxes
    bboxes_waypoints: (
        jt.Float32[npt.NDArray, "num_boxes timesteps 2"] | None
    )  # Uncompressed boxes waypoints
    bboxes_num_waypoints: (
        jt.Int32[npt.NDArray, " num_boxes"] | None
    )  # Uncompressed boxes num waypoints
    bev_occupancy: (
        jt.UInt8[npt.NDArray, " bev_occupancy_bytes"] | None
    )  # PNG compressed integer image
    bev_3rd_person_image: (
        jt.UInt8[npt.NDArray, " bev_3rd_person_bytes"] | None
    )  # JPEG compressed BEV image (as numpy array)
    radars: tuple[bytes, ...] | None  # Tuple of compressed radar data
    radar_detections: (
        jt.Float32[npt.NDArray, "num_radar_queries radar_features"] | None
    )  # Uncompressed radar detections

    @beartype
    def decompress(self) -> SensorData:
        """Decompress all sensor data back to original format."""
        # RGB image
        decompressed_image = None
        if self.image is not None:
            decompressed_image = cv2.imdecode(
                np.frombuffer(self.image, np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB)

        # LiDAR BEV
        decompressed_lidar_bev = None
        if self.lidar_bev is not None:
            decompressed_lidar_bev = decompress_float_image(self.lidar_bev)

        # Semantic
        decompressed_semantic = None
        if self.semantic is not None:
            decompressed_semantic = cv2.imdecode(self.semantic, cv2.IMREAD_UNCHANGED)

        # BEV semantic
        decompressed_bev_semantic = None
        if self.bev_semantic is not None:
            decompressed_bev_semantic = cv2.imdecode(
                self.bev_semantic,
                cv2.IMREAD_UNCHANGED,
            )

        # Depth
        decompressed_depth = None
        if self.depth is not None:
            encoded_depth = cv2.imdecode(self.depth, cv2.IMREAD_UNCHANGED)
            decompressed_depth = common_utils.decode_depth(encoded_depth)

        # BEV occupancy
        decompressed_bev_occupancy = None
        if self.bev_occupancy is not None:
            decompressed_bev_occupancy = cv2.imdecode(
                self.bev_occupancy,
                cv2.IMREAD_UNCHANGED,
            )

        # BEV 3rd person image
        decompressed_bev_3rd_person_image = None
        if self.bev_3rd_person_image is not None:
            decompressed_bev_3rd_person_image = cv2.imdecode(
                self.bev_3rd_person_image,
                cv2.IMREAD_UNCHANGED,
            )

        # Radars
        decompressed_radars = None
        if self.radars is not None:
            decompressed_radars_list = []
            for compressed_radar in self.radars:
                decompressed_radars_list.append(
                    decompress_radar_lossless(compressed_radar),
                )
            decompressed_radars = tuple(decompressed_radars_list)

        return SensorData(
            image=decompressed_image,
            rasterized_lidar=decompressed_lidar_bev,
            semantic=decompressed_semantic,
            hdmap=decompressed_bev_semantic,
            depth=decompressed_depth,
            boxes=self.bboxes,  # Boxes are not compressed
            boxes_waypoints=self.bboxes_waypoints,  # Boxes waypoints are not compressed
            boxes_num_waypoints=self.bboxes_num_waypoints,  # Boxes num waypoints are not compressed
            bev_occupancy=decompressed_bev_occupancy,
            bev_3rd_person_image=decompressed_bev_3rd_person_image,
            radars=decompressed_radars,
            radar_detections=self.radar_detections,  # Radar detections are not compressed
        )


@dataclass(frozen=True)
class CacheKey:
    scenario: str
    route: str
    frame: str
    perturbated: bool
    config: TrainingConfig

    @property
    def persistent_cache_full_path(self):
        perturbation_path = "normal"
        if self.perturbated:
            perturbation_path = "perturbated"
        return os.path.join(
            self.config.carla_root,
            "cache",
            self.scenario,
            self.route,
            *self.config.carla_cache_path,
            perturbation_path,
            f"{self.frame}.pkl",
        )

    def __str__(self):
        return f"{self.scenario}_{self.route}_{self.frame}_{self.perturbated})"


class PersistentCache:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.existence_cache = {}

    def __contains__(self, key: CacheKey):
        if key in self.existence_cache and self.existence_cache[key]:
            return True
        exists = os.path.exists(key.persistent_cache_full_path)
        self.existence_cache[key] = exists
        return exists

    def __setitem__(self, key: CacheKey, value):
        path = key.persistent_cache_full_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with lzma.open(path, "wb") as f:
            pickle.dump(value, f)
        self.existence_cache[key] = True

    def __getitem__(self, key: CacheKey):
        if key not in self:
            raise KeyError(f"CacheKey {key} does not exist.")
        with lzma.open(key.persistent_cache_full_path, "rb") as f:
            return pickle.load(f)


@beartype
def compress_float_image(
    image: npt.NDArray,
    config: TrainingConfig,
) -> jt.UInt8[npt.NDArray, " N"]:
    """Compress a float image to PNG format for storage efficiency.

    Args:
        image: Float image array with values in range [0, 1].
        config: Training configuration object containing compression settings.

    Returns:
        Compressed image as bytes in PNG format.
    """
    assert 0 <= image.min() <= image.max() <= 1.0, (
        f"Image values should be in range [0, 1]. Found {image.min()} to {image.max()}"
    )
    scaled_image = (image * 65535.0).astype(np.uint16)  # Scale to 16-bit range
    success, compressed = cv2.imencode(
        ".png",
        scaled_image,
        [int(cv2.IMWRITE_PNG_COMPRESSION), config.training_png_compression_level],
    )
    if not success:
        raise RuntimeError("Failed to compress float image")
    return compressed


@beartype
def decompress_float_image(
    compressed_image: jt.UInt8[npt.NDArray, " N"],
) -> npt.NDArray:
    """Decompress a PNG-compressed float image back to original format.

    Args:
        compressed_image: Compressed image bytes in PNG format.

    Returns:
        Decompressed float image array with values in range [0, 1].
    """
    decoded_image = (
        cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED).astype(np.float32)
        / 65535.0
    )
    assert 0 <= decoded_image.min() <= decoded_image.max() <= 1.0, (
        f"Decoded image values should be in range [0, 1]. Found {decoded_image.min()} to {decoded_image.max()}"
    )
    return decoded_image


@beartype
def compress_integer_image_lossless(
    image: npt.NDArray,
    config: TrainingConfig,
) -> jt.UInt8[npt.NDArray, " N"]:
    """Compress an integer image to PNG format using lossless compression.

    Args:
        image: Integer image array of type np.uint8.
        config: Training configuration object containing compression settings.

    Returns:
        Compressed image as bytes in PNG format.
    """
    assert image.dtype == np.uint8, "Image must be of type np.uint8"
    success, compressed = cv2.imencode(
        ".png",
        image,
        [int(cv2.IMWRITE_PNG_COMPRESSION), config.training_png_compression_level],
    )
    if not success:
        raise RuntimeError("Failed to compress integer image")
    return compressed


@beartype
def compress_integer_image_lossy(image: npt.NDArray, jpeg_quality: int) -> bytes:
    """Compress an integer image to JPEG format using lossy compression.

    Args:
        image: Integer image array of type np.uint8.
        jpeg_quality: JPEG compression quality (0-100).

    Returns:
        Compressed image as bytes in JPEG format.
    """
    assert image.dtype == np.uint8, "Image must be of type np.uint8"
    success, compressed = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError("Failed to compress integer image lossy")
    return compressed


@beartype
def compress_radar_lossless(radar: npt.NDArray) -> bytes:
    """Compress radar data using lossless compression with numpy.

    Args:
        radar: Radar data array to compress.

    Returns:
        Compressed radar data as bytes.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=radar.astype(np.float16))
    return buf.getvalue()


@beartype
def decompress_radar_lossless(buf: bytes) -> npt.NDArray:
    """Decompress radar data from lossless compressed format.

    Args:
        buf: Compressed radar data as bytes.

    Returns:
        Decompressed radar data as numpy array.
    """
    with np.load(io.BytesIO(buf)) as data:
        return data["arr"]
