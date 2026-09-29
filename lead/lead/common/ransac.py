#!/usr/bin/env python
"""
Ground plane removal from a point cloud. What this algorithm does:
0. Choose a set of centers.
1. Divide the point cloud into radial segments around each center.
2. For each radial segment, fit a ground plane iteratively.
3. Remove points that are close to the ground plane.
4. Return the union of all ground masks of every radial segment.
5. Return the final ground mask which is the union of all ground masks of every center.
"""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from lead.expert.config_expert import ExpertConfig


@njit(cache=True)
def extract_initial_seeds(P_segment, center, Th_center, N_LPR, Th_seeds):
    """Extract initial seed points for ground plane estimation. Choose the lowest points near center."""
    dist = np.abs(P_segment[:, 0] - center[0]) + np.abs(
        P_segment[:, 1] - center[1],
    )  # L1 distance
    P_near_center = P_segment[dist <= Th_center]
    P_sorted = P_near_center[np.argsort(P_near_center[:, 2])]
    if len(P_sorted) < N_LPR:
        return np.empty((0, 3))  # Return empty array if not enough points
    LPR_height = np.median(P_sorted[:N_LPR, 2])
    return P_sorted[np.abs(P_sorted[:, 2] - LPR_height) < Th_seeds]


@njit(cache=True)
def estimate_plane(P_g):
    """Estimate plane coefficients with least square."""
    x_coords = P_g[:, 0]
    y_coords = P_g[:, 1]
    ones = np.ones(P_g.shape[0], dtype=np.float64)
    A = np.column_stack((x_coords, y_coords, ones))
    b = P_g[:, 2]
    A = np.ascontiguousarray(A)
    b = np.ascontiguousarray(b)
    coeffs = np.linalg.solve(A.T @ A, A.T @ b)  # Solve (A^T A) x = A^T b
    return coeffs


@njit(cache=True)
def height_difference_to_plane(plane, P):
    """Calculate height differences to the estimated plane."""
    a, b, d = plane
    plane_heights = a * P[:, 0] + b * P[:, 1] + d
    return (P[:, 2] - plane_heights) / np.sqrt(a**2 + b**2 + 1)


@njit(cache=True)
def fit_plane_on_radial_segment(
    P_segment,
    center,
    N_iter,
    N_LPR,
    Th_seeds,
    Th_dist,
    Th_center,
):
    """Fit ground plane for a radial_segment iteratively."""
    if len(P_segment) < 3:
        return np.zeros(P_segment.shape[0], dtype=np.bool_)
    P_g = extract_initial_seeds(P_segment, center, Th_center, N_LPR, Th_seeds)
    if len(P_g) < 3:
        return np.zeros(P_segment.shape[0], dtype=np.bool_)
    for _ in range(N_iter):
        try:
            plane_model = estimate_plane(P_g)
        except:  #
            return np.zeros(P_segment.shape[0], dtype=np.bool_)
        distances = height_difference_to_plane(plane_model, P_segment)
        is_ground = distances < Th_dist
        P_g = P_segment[is_ground]
    return is_ground | (
        P_segment[:, 2] < np.percentile(P_segment[:, 2], q=30)
    )  # Discard points below 30th percentile. TODO: parametrize this.


@njit(cache=True)
def divide_plane_into_radial_segments(P, center, n_segments):
    """Divide points into angular radial_segments."""
    shifted_points = P[:, :2] - center[:2]
    angles = np.arctan2(shifted_points[:, 1], shifted_points[:, 0])
    angle_step = 2 * np.pi / n_segments

    radial_segments = []
    segments_point_indices = []
    loop_range = range(n_segments)
    for i in loop_range:
        min_angle = -np.pi + i * angle_step
        max_angle = min_angle + angle_step
        segment_mask = (angles >= min_angle) & (angles < max_angle)
        radial_segments.append(P[segment_mask])
        segments_point_indices.append(np.where(segment_mask)[0])
    return radial_segments, segments_point_indices


@njit(cache=True)
def remove_ground_around_a_center(
    P,
    center,
    n_segments,
    N_iter,
    N_LPR,
    Th_seeds,
    Th_dist,
    Th_center,
):
    """Fit a ground plane for each radial segment around a center."""
    radial_segments, segments_point_indices = divide_plane_into_radial_segments(
        P,
        center,
        n_segments,
    )
    plane_ground_mask = np.zeros(P.shape[0], dtype=np.bool_)
    loop_range = range(len(radial_segments))
    for i in loop_range:
        radial_segment, segment_point_indices = (
            radial_segments[i],
            segments_point_indices[i],
        )
        radial_segment_ground_mask = fit_plane_on_radial_segment(
            radial_segment,
            center,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
        )
        plane_ground_mask[segment_point_indices] = radial_segment_ground_mask
    return plane_ground_mask


@njit(cache=True)
def fit_ground(
    P,
    centers,
    n_segments,
    N_iter,
    N_LPR,
    Th_seeds,
    Th_dist,
    Th_center,
    max_r,
    min_r,
):
    """Fit ground plane for each center, return the conjunction of ground masks.."""
    n_points = P.shape[0]
    union_ground_mask = np.zeros(n_points, dtype=np.bool_)
    n_centers = centers.shape[0]
    loop_range = range(n_centers)
    for i in loop_range:
        center = centers[i]
        distances_to_center = np.sum(
            np.abs(P[:, :2] - center[:2]),
            axis=1,
        )  # L1 distance
        if (
            np.sum(distances_to_center < 20) < 3
        ):  # If there are less than 5 points within 20m, skip. TODO: parametrize this.
            continue
        proximity_mask = (distances_to_center < max_r) & (
            distances_to_center > min_r
        )  # Only consider points within a certain radius
        points_within_radius = P[proximity_mask]
        center_ground_mask = remove_ground_around_a_center(
            points_within_radius,
            center,
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
        )
        union_ground_mask[proximity_mask] |= center_ground_mask
    return union_ground_mask


@njit(parallel=True, cache=True)
def fit_ground_parallel(
    P,
    centers,
    n_segments,
    N_iter,
    N_LPR,
    Th_seeds,
    Th_dist,
    Th_center,
    max_r,
    min_r,
):
    """Fit ground plane for each center, return the conjunction of ground masks.."""
    n_points = P.shape[0]
    union_ground_mask = np.zeros(n_points, dtype=np.bool_)
    n_centers = centers.shape[0]
    loop_range = prange(n_centers)
    for i in loop_range:
        center = centers[i]
        distances_to_center = np.sum(
            np.abs(P[:, :2] - center[:2]),
            axis=1,
        )  # L1 distance
        if (
            np.sum(distances_to_center < 20) < 3
        ):  # If there are less than 5 points within 20m, skip. TODO: parametrize this.
            continue
        proximity_mask = (distances_to_center < max_r) & (
            distances_to_center > min_r
        )  # Only consider points within a certain radius
        points_within_radius = P[proximity_mask]
        center_ground_mask = remove_ground_around_a_center(
            points_within_radius,
            center,
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
        )
        union_ground_mask[proximity_mask] |= center_ground_mask
    return union_ground_mask


def generate_center_grid(center_resolution, min_x, max_x, min_y, max_y):
    """Generate a grid of centers for ground plane fitting.."""
    x_values = np.arange(
        min_x - center_resolution // 2,
        max_x + center_resolution,
        center_resolution,
    )
    y_values = np.arange(
        min_y - center_resolution // 2,
        max_y + center_resolution,
        center_resolution,
    )
    grid_points = np.array([(x, y) for x in x_values for y in y_values])
    grid_points = [center for center in grid_points] + [[0, 0]]  # Add the origin
    return np.array(grid_points).reshape(-1, 2)


def remove_ground(
    P,
    config,
    n_segments: int = 8,
    N_iter: int = 2,
    N_LPR: int = 64,
    Th_seeds: float = 0.2,
    Th_dist: float = 0.15,
    Th_center: float = 50.0,
    min_r: float = 1.0,
    max_r: float = 32.0,
    center_resolution: float = 28.0,
    parallel: bool = False,
):
    """Remove ground points from a point cloud.

    Args:
        P: shape=(N, 3): Point cloud to remove ground points from.
        config: Configuration object containing dataset parameters.
        n_segments: Number of radial segments to divide around each center.
        N_iter: Number of iterations for ground plane fitting.
        N_LPR: Number of lowest points to consider for initial seed points.
        Th_seeds: Threshold for selecting initial seed points.
        Th_dist: Threshold for inlier points to the fitted plane.
        Th_center: Threshold for selecting points near the center.
        min_r: Minimum radius for selecting points around a center.
        max_r: Maximum radius for selecting points around a center.
        center_resolution: Resolution for generating centers.
        parallel: Whether to use parallel processing.
    Returns:
        Boolean mask shape (n,) indicating whether a point is on a ground or not.
            True indicates ground point.
            False indicates non-ground point.
    """
    if P.dtype != np.float64:
        P = P.astype(np.float64)
    centers = generate_center_grid(
        center_resolution,
        config.min_x_meter,
        config.max_x_meter,
        config.min_y_meter,
        config.max_y_meter,
    )
    if parallel:
        mask = fit_ground_parallel(
            P,
            centers,
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
            max_r,
            min_r,
        )
    else:
        mask = fit_ground(
            P,
            centers,
            n_segments,
            N_iter,
            N_LPR,
            Th_seeds,
            Th_dist,
            Th_center,
            max_r,
            min_r,
        )
    # mask = mask | ((P[:, 0] < 2.45) & (-2.45 < P[:, 0]) & (P[:, 1] < 1.06) & (-1.06 < P[:, 1]))  # Remove points inside ego
    return mask
