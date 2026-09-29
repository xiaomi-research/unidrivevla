import itertools
from copy import deepcopy

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import shapely
from beartype import beartype
from shapely.geometry import Polygon


def rect_polygon(
    x: float,
    y: float,
    width: float,
    height: float,
    angle: float,
) -> Polygon:
    """Create a shapely Polygon representing a rotated rectangle.

    Args:
        x: Center x-coordinate of the rectangle.
        y: Center y-coordinate of the rectangle.
        width: Width of the rectangle.
        height: Height of the rectangle.
        angle: Rotation angle in radians.

    Returns:
        Shapely Polygon representing the rotated rectangle.
    """
    p = Polygon(
        [(-width, -height), (width, -height), (width, height), (-width, height)],
    )
    # Shapely is very inefficient at these operations, worth rewriting
    return shapely.affinity.translate(
        shapely.affinity.rotate(p, angle, use_radians=True),
        x,
        y,
    )


def iou_bbs(bb1: jt.Float[npt.NDArray, "5"], bb2: jt.Float[npt.NDArray, "5"]) -> float:
    """Calculate Intersection over Union (IoU) between two oriented bounding boxes.

    Args:
        bb1: First bounding box as [x, y, width, height, angle].
        bb2: Second bounding box as [x, y, width, height, angle].

    Returns:
        IoU value between 0 and 1.
    """
    a = rect_polygon(bb1[0], bb1[1], bb1[2], bb1[3], bb1[4])
    b = rect_polygon(bb2[0], bb2[1], bb2[2], bb2[3], bb2[4])
    intersection_area = a.intersection(b).area
    union_area = a.union(b).area
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou


@beartype
def non_maximum_suppression(
    bounding_boxes: list[jt.Float[npt.NDArray, "num_boxes D"]],
    iou_threshold: float,
) -> jt.Float[npt.NDArray, "num_filtered_boxes D"]:
    """
    Basic Non-Maximum Suppression (NMS) implementation for oriented bounding boxes.

    Args:
        bounding_boxes: List of bounding boxes produced by an ensemble of detectors.
        iou_threshold: IoU threshold for NMS.

    Returns:
        List of filtered bounding boxes after NMS.
    """
    filtered_boxes = []
    bounding_boxes = np.array(
        list(itertools.chain.from_iterable(bounding_boxes)),
        dtype=object,
    )

    if bounding_boxes.size == 0:  # If no bounding boxes are detected can't do NMS
        return np.array(filtered_boxes)
    bounding_boxes = bounding_boxes.reshape(-1, 9)
    confidences_indices = np.argsort(bounding_boxes[:, -1])
    while len(confidences_indices) > 0:
        idx = confidences_indices[-1]
        current_bb = bounding_boxes[idx]
        filtered_boxes.append(current_bb.copy())
        # Remove last element from the list
        confidences_indices = confidences_indices[:-1]

        if len(confidences_indices) == 0:
            break

        for idx2 in deepcopy(confidences_indices):
            if (
                iou_bbs(current_bb, bounding_boxes[idx2]) > iou_threshold
            ):  # Remove BB from list
                confidences_indices = confidences_indices[confidences_indices != idx2]

    return np.array(filtered_boxes).astype(np.float32)
