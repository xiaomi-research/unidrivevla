# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Transformation utility functions.
Code adapted from https://github.com/zhejz/carla-roach
"""

import carla
import numpy as np


def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_global = carla.Vector3D(x=x, y=y, z=z)
    vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)
    target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
    return target_loc_in_ref


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    rotation = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array(
        [[target_vec_in_global.x], [target_vec_in_global.y], [target_vec_in_global.z]],
    )
    np_vec_in_ref = rotation.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(
        x=np_vec_in_ref[0, 0],
        y=np_vec_in_ref[1, 0],
        z=np_vec_in_ref[2, 0],
    )
    return target_vec_in_ref


def carla_rot_to_mat(carla_rotation):
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ],
    )
    roll_matrix = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]],
    )

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix


def cast_angle(x):
    return (x + 180.0) % 360.0 - 180.0
