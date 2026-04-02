import os
import json
import torch
import numpy as np
import cv2, mmcv, glob
import matplotlib.pyplot as plt


def draw_bboxes2d(img, bboxes, labels=None, centers=None, format='xywh',
                  box_color=(0, 0, 255), box_thickness=2,
                  font_color=None, font_size=None, font_thickness=None,
                  center_color=None, center_radius=None, center_thickness=None):
    # font
    font_color = box_color if font_color is None else font_color
    font_size = box_thickness/2 if font_size is None else font_size
    font_thickness = box_thickness if font_thickness is None else font_thickness
    # center
    center_color = box_color if center_color is None else center_color
    center_radius = box_thickness * 2 if center_radius is None else center_radius
    center_thickness = int(box_thickness * 1.5) if center_thickness is None else center_thickness

    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox)
        if format=='xywh':
            bbox[-2:] = bbox[:2] + bbox[-2:]
        if bboxes is not None:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, box_thickness)
        if labels is not None:
            cv2.putText(img, str(labels[i]), (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        if centers is not None:
            cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), center_radius, center_color, center_thickness)
    return img

def draw_bboxes3d(img, bboxes3d, labels=None, intrinsic=None, extrinsic=None, lidar2img=None,
                  color=(0, 0, 255), back_color=None, thickness=2):
    if labels is None:
        labels = [None] * len(bboxes3d)
    if back_color is None:
        back_color = color

    if lidar2img is not None:
        for box3d, label in zip(bboxes3d, labels):
            corners_global = get_box3d_corner(box3d)
            corners_image = lidar2img @ corners_global
            corners_image[2, :] = np.clip(corners_image[2, :], a_min=1e-5, a_max=1e5)
            corners_image[:2, :] = corners_image[:2, :] / corners_image[2:3, :]

            img = draw_box3d_line(img, corners_image, corners_image, label, color=color, back_color=back_color, thickness=thickness)
    else:
        for box3d, label in zip(bboxes3d, labels):
            corners_global = get_box3d_corner(box3d)
            corners_image = intrinsic @ extrinsic @ corners_global
            corners_image[2, :] = np.clip(corners_image[2, :], a_min=1e-5, a_max=1e5)
            corners_image[:2, :] = corners_image[:2, :] / corners_image[2:3, :]

            draw_corners_camera = extrinsic @ corners_global
            draw_corners_camera[2, :] = np.abs(draw_corners_camera[2, :])
            draw_corners_image = intrinsic @ draw_corners_camera
            draw_corners_image[:2, :] = draw_corners_image[:2, :] / draw_corners_image[2:3, :]

            img = draw_box3d_line(img, corners_image, draw_corners_image, label, color=color, back_color=back_color, thickness=thickness)

    return img

def get_box3d_corner(box3d):
    posx, posy, posz, l, w, h = box3d[:6]
    theta = box3d[6]

    rotation_matrix = np.array([[+np.cos(theta), -np.sin(theta),0],
                                [+np.sin(theta), +np.cos(theta),0],
                                [0,              0,             1]])

    coord_x = [l/2., l/2., l/2., l/2., -l/2., -l/2., -l/2., -l/2.]
    coord_y = [-w/2., -w/2., w/2., w/2., w/2., w/2., -w/2., -w/2.]
    coord_z = [h/2., -h/2., h/2., -h/2., h/2., -h/2., h/2., -h/2.]

    corners = rotation_matrix @ np.array([coord_x, coord_y, coord_z])
    center = np.array([posx, posy, posz]).reshape(3, 1)
    global_corners = corners + center

    global_corners = np.concatenate([global_corners, np.ones((1, 8))], axis=0)
    return global_corners

def draw_box3d_line(img, uvs, draw_uvs, label=None, color=(0, 0, 255), back_color=(255, 0, 0),
                    thickness=2, draw_bbox=False):
    line_indices = (
        (0, 1), (2, 3), (0, 2), (1, 3),   # front
        (4, 5), (6, 7), (4, 6), (5, 7),   # back
        (0, 6), (1, 7), (2, 4), (3, 5),   # link line between front and back
    )
    is_draw = False
    h, w = img.shape[:2]
    uvs = np.clip(uvs, -1e4, 1e5).astype(np.int32)
    draw_uvs = np.clip(draw_uvs, -1e4, 1e5).astype(np.int32)
    for start, end in line_indices:
        if ((uvs[0, start]>=w or uvs[0, start] < 0) or (uvs[1, start]>=h or uvs[1, start] < 0)) and \
            ((uvs[0, end]>=w or uvs[0, end] < 0) or (uvs[1, end]>=h or uvs[1, end] < 0)):
            continue
        is_draw = True
        if (start, end) in [(4, 5), (6, 7), (4, 6), (5, 7)]:
            cv2.line(img,
                     ((draw_uvs[0, start]), (draw_uvs[1, start])),
                     ((draw_uvs[0, end]), (draw_uvs[1, end])),
                     back_color,
                     thickness)
        else:
            cv2.line(img,
                     ((draw_uvs[0, start]), (draw_uvs[1, start])),
                     ((draw_uvs[0, end]), (draw_uvs[1, end])),
                     color,
                     thickness)

    if is_draw and label is not None:
        cv2.putText(img, str(label), (int((draw_uvs[0, 0]+draw_uvs[0, 1]+draw_uvs[0, 2]+draw_uvs[0, 3])*0.25),
                                      int((draw_uvs[1, 0]+draw_uvs[1, 1]+draw_uvs[1, 2]+draw_uvs[1, 3])*0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, thickness)

    if is_draw and draw_bbox:
        img_h, img_w = img.shape[0], img.shape[1]
        x_min, y_min, x_max, y_max = draw_uvs[0].min(), draw_uvs[1].min(), draw_uvs[0].max(), draw_uvs[1].max()

        if (0 < x_min < img_w and 0 < y_min < img_h ) or \
            (0 < x_max < img_w and 0 < y_max < img_h):
            x_min, y_min, x_max, y_max = max(x_min, 0), max(y_min, 0), min(x_max, img_w), min(y_max, img_h)

            cx, cy = (x_min + x_max)/ 2, (y_min + y_max)/ 2
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), thickness)
            cv2.circle(img, (int(cx), int(cy)), 4, (255, 0, 0), 3)

    return img
