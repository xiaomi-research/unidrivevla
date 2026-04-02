
import os
import sys
import argparse
import math
import colorsys
import time
import pickle
from typing import List, Optional, Tuple

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

OCC_COLOR_MAP = np.array([
    [255, 120,  50],
    [255, 192, 203],
    [255, 255,   0],
    [  0, 150, 245],
    [  0, 255, 255],
    [255, 127,   0],
    [255,   0,   0],
    [255, 240, 150],
    [135,  60,   0],
    [160,  32, 240],
    [255,   0, 255],
    [139, 137, 137],
    [ 75,   0,  75],
    [150, 240,  80],
    [230, 230, 250],
    [  0, 175,   0],
], dtype=np.uint8)

OCC_CLASS_NAMES = [
    'others',
    'barrier', 'bicycle', 'bus', 'car', 'construction',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free', 'unobserved'
]

def vis_occ_open3d(sem, out_path, voxelSize=(0.4, 0.4, 0.4), vox_range=(-40., -40., -1., 40., 40., 5.4), img_w=2560, img_h=1440):
    """Render OCC using VTK cube glyphs, replicating visualize_nuscenes_occupancy.py VIEW_1."""
    try:
        import vtk

        H, W, Z = sem.shape
        vx, vy, vz = voxelSize

        mask = (sem >= 1) & (sem <= 16)
        xi, yi, zi = np.where(mask)
        if len(xi) == 0:
            _vis_occ_matplotlib_bev(sem, out_path, voxelSize, vox_range)
            return

        lbls = sem[xi, yi, zi].astype(np.float32)
        cx = (vox_range[0] + (xi + 0.5) * vx).astype(np.float32)
        cy = (vox_range[1] + (yi + 0.5) * vy).astype(np.float32)
        cz = (vox_range[2] + (zi + 0.5) * vz).astype(np.float32)

        # Build VTK point set with scalar labels
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetNumberOfPoints(len(cx))
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfValues(len(cx))
        for i in range(len(cx)):
            vtk_pts.SetPoint(i, float(cx[i]), float(cy[i]), float(cz[i]))
            scalars.SetValue(i, float(lbls[i]))

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_pts)
        pd.GetPointData().SetScalars(scalars)

        # Cube glyph (same as mlab.points3d mode='cube')
        cube = vtk.vtkCubeSource()
        side = vx * 0.95
        cube.SetXLength(side); cube.SetYLength(side); cube.SetZLength(side)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(pd)
        glyph.SetSourceConnection(cube.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetColorModeToColorByScalar()
        glyph.Update()

        # Build LUT from OCC_COLOR_MAP (matches colors array in reference script)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(20)
        lut.SetTableRange(1, 19)
        lut.Build()
        for i, rgb in enumerate(OCC_COLOR_MAP):
            lut.SetTableValue(i + 1, rgb[0]/255., rgb[1]/255., rgb[2]/255., 1.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarRange(1, 19)
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().SetSpecular(0.0)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.AddActor(actor)

        renwin = vtk.vtkRenderWindow()
        renwin.SetOffScreenRendering(1)
        renwin.SetSize(img_w, img_h)
        renwin.AddRenderer(ren)

        # VIEW_1 camera from visualize_nuscenes_occupancy.py
        cam = ren.GetActiveCamera()
        cam.SetPosition(-70, -70, 60)
        cam.SetFocalPoint(0.0, 0.0, -3.0)
        cam.SetViewAngle(40.0)
        cam.SetViewUp(0, 0, 1)
        cam.SetClippingRange(0.01, 300.0)
        ren.ResetCameraClippingRange()

        renwin.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(renwin)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.SetFileName(str(out_path))
        writer.Write()
    except Exception:
        _vis_occ_matplotlib_bev(sem, out_path, voxelSize, vox_range)


def _vis_occ_matplotlib_bev(sem, out_path, voxelSize=(0.4, 0.4, 0.4), vox_range=(-40., -40., -1., 40., 40., 5.4)):
    H, W, Z = sem.shape
    vx, vy = voxelSize[0], voxelSize[1]
    x_min, y_min = vox_range[0], vox_range[1]
    x_max, y_max = vox_range[3], vox_range[4]

    # Top-down projection: take topmost occupied voxel per column (z from top to bottom)
    top = np.full((H, W), 17, dtype=np.int32)  # default: free
    for z in range(Z - 1, -1, -1):
        layer = sem[:, :, z].astype(np.int32)
        occupied = (layer >= 0) & (layer <= 16)
        top[occupied] = layer[occupied]

    # Build full palette: index = class label (0..17=free, 255->18=unknown)
    palette = np.zeros((19, 3), dtype=np.uint8)
    palette[0] = [80, 80, 80]          # 0: others
    palette[1:17] = OCC_COLOR_MAP[:, :3]  # 1..16: barrier..vegetation
    palette[17] = [20, 20, 20]         # 17: free (dark bg)
    palette[18] = [50, 50, 50]         # 18: unknown

    top_idx = top.copy()
    top_idx[top_idx == 255] = 18
    top_idx = np.clip(top_idx, 0, 18)

    # Render: H=x-axis (front=high x), W=y-axis; flip H so forward is up in image
    rgb = palette[top_idx][::-1, :, :]  # (H, W, 3), front of vehicle at top

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.imshow(rgb, interpolation='nearest',
              extent=[y_min, y_max, x_min, x_max],  # (left, right, bottom, top)
              origin='upper')
    ax.set_facecolor('#0d1117')

    # Axis ticks and grid
    tick_spacing = 10  # meters
    ax.set_xticks(np.arange(int(y_min), int(y_max) + 1, tick_spacing))
    ax.set_yticks(np.arange(int(x_min), int(x_max) + 1, tick_spacing))
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_xlabel('y (m)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('x (m, forward)', color='#aaaaaa', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, color='#333333', linewidth=0.5, alpha=0.6)

    # Ego vehicle marker
    ax.plot(0, 0, '^', color='#ff3333', markersize=12, zorder=10, label='ego')

    # Legend: only show classes that appear
    present = np.unique(top_idx)
    patches = []
    for cid in present:
        if 0 <= cid <= 16:
            patches.append(mpatches.Patch(color=palette[cid] / 255.0, label=OCC_CLASS_NAMES[cid]))
    if patches:
        leg = ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=2,
                        facecolor='#1a1a2e', labelcolor='white', edgecolor='#444444',
                        framealpha=0.9, title='Classes', title_fontsize=8)
        leg.get_title().set_color('white')

    ax.set_title('OCC BEV (top-down)', color='white', fontsize=11, pad=6)
    fig.tight_layout()
    savefig(fig, out_path, dpi=200)

CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
BOX_COLOR_BGR = (0, 255, 0)     # bright green, highly visible on camera images
BOX_THICKNESS = 3
MAP_CLASSES = ['ped_crossing', 'divider', 'boundary']
MAP_BGR = [(0, 165, 255), (50, 230, 50), (255, 100, 50)]
MAP_RGB = [(b/255, g/255, r/255) for r, g, b in MAP_BGR]
EGO_BGR = (0, 255, 255)         # yellow in BGR for ego trajectory on camera
EGO_COLOR = '#ffff00'           # bright yellow for ego trajectory in BEV
BEV_RANGE = 51.2
BG_COLOR = '#0d1117'

NUSC_MAP_LAYERS = {'road_divider': 1, 'lane_divider': 1, 'ped_crossing': 0, 'road_segment': 2, 'lane': 2}

def savefig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)

def bev_axis(ax):
    ax.set_xlim([-BEV_RANGE, BEV_RANGE])
    ax.set_ylim([-BEV_RANGE, BEV_RANGE])
    ax.set_aspect('equal')
    ax.set_facecolor(BG_COLOR)
    ax.grid(False)
    ax.axis('off')

def get_map_annos_from_api(nusc, nusc_maps, sample_token, patch_radius=60.0):
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']

    if map_name not in nusc_maps:
        return {}

    nmap = nusc_maps[map_name]
    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
    ego_xy = np.array(ego_pose['translation'][:2])
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
    patch_box = (ego_xy[0], ego_xy[1], patch_radius * 2, patch_radius * 2)

    result = {0: [], 1: [], 2: []}
    layer_map = {'road_divider': 1, 'lane_divider': 1, 'ped_crossing': 0}

    for layer_name, label_int in layer_map.items():
        try:
            records = nmap.get_records_in_patch(patch_box, [layer_name], mode='intersect')
        except Exception:
            continue
        for token in records.get(layer_name, []):
            rec = nmap.get(layer_name, token)
            node_tokens = rec.get('exterior_node_tokens', rec.get('node_tokens', []))
            if len(node_tokens) < 2:
                continue
            pts_global = np.array([[nmap.get('node', t)['x'], nmap.get('node', t)['y']] for t in node_tokens])
            c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
            R = np.array([[c, -s], [s, c]])
            pts_ego = (pts_global - ego_xy) @ R.T
            within = (np.abs(pts_ego[:, 0]) < patch_radius) & (np.abs(pts_ego[:, 1]) < patch_radius)
            if within.sum() < 2:
                continue
            result[label_int].append(pts_ego)

    try:
        records = nmap.get_records_in_patch(patch_box, ['road_segment'], mode='intersect')
        for token in records.get('road_segment', []):
            rec = nmap.get('road_segment', token)
            node_tokens = rec.get('exterior_node_tokens', [])
            if len(node_tokens) < 2:
                continue
            pts_global = np.array([[nmap.get('node', t)['x'], nmap.get('node', t)['y']] for t in node_tokens])
            c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
            R = np.array([[c, -s], [s, c]])
            pts_ego = (pts_global - ego_xy) @ R.T
            within = (np.abs(pts_ego[:, 0]) < patch_radius) & (np.abs(pts_ego[:, 1]) < patch_radius)
            if within.sum() < 2:
                continue
            result[2].append(pts_ego)
    except Exception:
        pass

    return result

def lidar_pts_to_cam(pts_lidar, nusc, sample_token, cam):
    if pts_lidar.shape[1] == 2:
        pts_lidar = np.hstack([pts_lidar, np.zeros((len(pts_lidar), 1))])

    sample = nusc.get('sample', sample_token)
    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sd_cam = nusc.get('sample_data', sample['data'][cam])
    cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    pose_l = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
    pose_c = nusc.get('ego_pose', sd_cam['ego_pose_token'])

    p = pts_lidar.T.copy()
    p = Quaternion(cs_lidar['rotation']).rotation_matrix @ p + np.array(cs_lidar['translation'])[:, None]
    p = Quaternion(pose_l['rotation']).rotation_matrix @ p + np.array(pose_l['translation'])[:, None]
    p = Quaternion(pose_c['rotation']).rotation_matrix.T @ (p - np.array(pose_c['translation'])[:, None])
    p = Quaternion(cs_cam['rotation']).rotation_matrix.T @ (p - np.array(cs_cam['translation'])[:, None])

    depth = p[2]
    K = np.array(cs_cam['camera_intrinsic'])
    pixels = view_points(p, K, normalize=True)[:2]
    W, H = sd_cam['width'], sd_cam['height']
    valid = (depth > 0.1) & (pixels[0] >= 0) & (pixels[0] < W) & (pixels[1] >= 0) & (pixels[1] < H)
    return pixels, valid


def ego_pts_to_cam(pts_ego, nusc, sample_token, cam):
    """Project points from ego frame (output of get_map_annos_from_api) to camera pixels.

    get_map_annos_from_api returns points in ego body frame (centered at ego pos, rotated by
    ego yaw), NOT lidar frame. lidar_pts_to_cam would apply lidar calibration incorrectly.
    This function skips the lidar calibration step and goes ego→global→camera directly.
    """
    if pts_ego.shape[1] == 2:
        # In nuScenes, ego_pose z ≈ 0 (ground level). Road surface = z=0 in ego frame.
        pts_ego = np.hstack([pts_ego, np.zeros((len(pts_ego), 1))])

    sample = nusc.get('sample', sample_token)
    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sd_cam = nusc.get('sample_data', sample['data'][cam])
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    pose_l = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
    pose_c = nusc.get('ego_pose', sd_cam['ego_pose_token'])

    p = pts_ego.T.copy()
    # ego → global (use lidar ego_pose since map coords are relative to it)
    p = Quaternion(pose_l['rotation']).rotation_matrix @ p + np.array(pose_l['translation'])[:, None]
    # global → camera ego
    p = Quaternion(pose_c['rotation']).rotation_matrix.T @ (p - np.array(pose_c['translation'])[:, None])
    # camera ego → camera sensor
    p = Quaternion(cs_cam['rotation']).rotation_matrix.T @ (p - np.array(cs_cam['translation'])[:, None])

    depth = p[2]
    K = np.array(cs_cam['camera_intrinsic'])
    pixels = view_points(p, K, normalize=True)[:2]
    W, H = sd_cam['width'], sd_cam['height']
    valid = (depth > 0.1) & (pixels[0] >= 0) & (pixels[0] < W) & (pixels[1] >= 0) & (pixels[1] < H)
    return pixels, valid

def ego_traj_pts(fut_trajs, fut_masks, z=0.0):
    # fut_trajs: (N,2) stored as (lateral_right, forward) in ego-centric frame
    # convert to nuScenes ego frame: x=forward, y=left
    traj = np.vstack([np.zeros((1, 2)), np.cumsum(fut_trajs, axis=0)])
    valid = np.concatenate([[True], fut_masks.astype(bool)])
    pts = np.zeros((len(traj), 3))
    pts[:, 0] = traj[:, 1]   # x_ego (forward)  = col1
    pts[:, 1] = -traj[:, 0]  # y_ego (left)      = -col0
    pts[:, 2] = z
    return pts[valid]

def draw_box3d_on_img(img, box, intrinsic, color=BOX_COLOR_BGR, thickness=BOX_THICKNESS):
    c3d = view_points(box.corners(), intrinsic, normalize=True)[:2]
    
    def line(i, j, t=thickness):
        cv2.line(img, (int(c3d[0, i]), int(c3d[1, i])), (int(c3d[0, j]), int(c3d[1, j])), color, t, cv2.LINE_AA)

    for i, j in [(4,5),(5,6),(6,7),(7,4)]: line(i, j)
    for i, j in [(0,1),(1,2),(2,3),(3,0)]: line(i, j)
    for i, j in [(0,4),(1,5),(2,6),(3,7)]: line(i, j)
    for i, j in [(0,1),(4,5)]: line(i, j, thickness+1)

def draw_map_on_img(img, map_annos, nusc, sample_token, cam):
    if not map_annos:
        return
    for label_int, polys in map_annos.items():
        if label_int >= len(MAP_CLASSES):
            continue
        # road_segment (label 2) are large closed polygons — skip for camera view,
        # only project open polylines (dividers=1) and ped_crossings=0
        if label_int == 2:
            continue
        color = MAP_BGR[label_int]
        for pts in polys:
            pts = np.array(pts)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            # densify polyline for smoother projection
            dense = []
            for k in range(len(pts) - 1):
                seg = np.linspace(pts[k], pts[k + 1], num=8, endpoint=False)
                dense.append(seg)
            dense.append(pts[-1:])
            pts_dense = np.vstack(dense)
            # map points are in ego frame → use ego_pts_to_cam
            pixels, valid = ego_pts_to_cam(pts_dense, nusc, sample_token, cam)
            if valid.sum() < 2:
                continue
            px = pixels[:, valid].T.astype(int)
            for k in range(len(px) - 1):
                cv2.line(img, tuple(px[k]), tuple(px[k + 1]), color, 3, cv2.LINE_AA)

def _traj_grad_color(t):
    """Orange (near, t=0) → yellow (far, t=1) gradient in BGR, matching reference style."""
    # near: (0, 140, 255) BGR = orange;  far: (30, 255, 220) BGR = yellow-green
    b = int(0   + t * 30)
    g = int(140 + t * 115)
    r = int(255 - t * 35)
    return (b, g, r)

def draw_ego_traj_on_img(img, fut_trajs, fut_masks, nusc, sample_token, cam, color=EGO_BGR, radius=6):
    pts = ego_traj_pts(fut_trajs, fut_masks)
    if len(pts) < 1:
        return
    pixels, valid = ego_pts_to_cam(pts, nusc, sample_token, cam)
    H, W = img.shape[:2]

    # Recompute depth-only validity (in front of camera, ignore image bounds)
    sample = nusc.get('sample', sample_token)
    sd_lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sd_cam = nusc.get('sample_data', sample['data'][cam])
    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
    pose_l = nusc.get('ego_pose', sd_lidar['ego_pose_token'])
    pose_c = nusc.get('ego_pose', sd_cam['ego_pose_token'])
    p = pts.T.copy()
    if p.shape[0] == 2:
        p = np.vstack([p, np.zeros((1, p.shape[1]))])
    p = Quaternion(pose_l['rotation']).rotation_matrix @ p + np.array(pose_l['translation'])[:, None]
    p = Quaternion(pose_c['rotation']).rotation_matrix.T @ (p - np.array(pose_c['translation'])[:, None])
    p = Quaternion(cs_cam['rotation']).rotation_matrix.T @ (p - np.array(cs_cam['translation'])[:, None])
    depth = p[2]
    depth_valid = depth > 0.1  # in front of camera

    if depth_valid.sum() == 0:
        return

    # pixels already computed above; clamp to image bounds for drawing
    px_all = pixels.T.copy()  # shape (N, 2)
    px_clamped = np.stack([
        np.clip(px_all[:, 0], 0, W - 1),
        np.clip(px_all[:, 1], 0, H - 1),
    ], axis=1).astype(int)

    valid_idxs = np.where(depth_valid)[0]
    N = len(valid_idxs)

    def pt_color(t):
        """t in [0,1]: 0=nearest (orange), 1=farthest (yellow-green)"""
        return _traj_grad_color(float(t))

    # Only draw points that are actually within image bounds (skip clamped-to-edge points)
    inview_idxs = [k for k in valid_idxs if 0 <= px_all[k, 1] < H and 0 <= px_all[k, 0] < W]
    M = len(inview_idxs)
    if M < 2:
        return

    # Draw smooth gradient lines: densify each segment into sub-segments for continuous color transition
    SUBSTEPS = 10
    for i in range(M - 1):
        p_start = px_clamped[inview_idxs[i]].astype(float)
        p_end   = px_clamped[inview_idxs[i + 1]].astype(float)
        t_start = i / max(M - 1, 1)
        t_end   = (i + 1) / max(M - 1, 1)
        for s in range(SUBSTEPS):
            a0 = s / SUBSTEPS
            a1 = (s + 1) / SUBSTEPS
            sub_p0 = tuple((p_start + a0 * (p_end - p_start)).astype(int))
            sub_p1 = tuple((p_start + a1 * (p_end - p_start)).astype(int))
            sub_t  = t_start + (a0 + a1) / 2 * (t_end - t_start)
            cv2.line(img, sub_p0, sub_p1, pt_color(sub_t), 16, cv2.LINE_AA)

def box_corners_bev(cx, cy, w, l, yaw):
    dx = np.array([ l/2,  l/2, -l/2, -l/2])
    dy = np.array([ w/2, -w/2, -w/2,  w/2])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([cx + c*dx - s*dy, cy + s*dx + c*dy], axis=1)

def draw_boxes_bev(ax, boxes, color='#00ff00'):  # bright green
    for box in boxes:
        cx, cy = box.center[0], box.center[1]
        w, l = box.wlh[0], box.wlh[1]
        yaw = box.orientation.yaw_pitch_roll[0]
        corners = box_corners_bev(cx, cy, w, l, yaw)
        poly = plt.Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=2.5, zorder=4)
        ax.add_patch(poly)
        alen = max(w, l) * 0.45
        ax.annotate('', xy=(cx + alen*np.cos(yaw), cy + alen*np.sin(yaw)), xytext=(cx, cy), arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=5)

def draw_map_bev(ax, map_annos):
    if not map_annos:
        return
    for label_int, polys in map_annos.items():
        if label_int >= len(MAP_CLASSES):
            continue
        color = MAP_RGB[label_int]
        for pts in polys:
            pts = np.array(pts)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2.0, alpha=0.9, zorder=3)

def draw_ego_traj_bev(ax, fut_trajs, fut_masks):
    pts = ego_traj_pts(fut_trajs, fut_masks)
    N = len(pts)
    cmap = matplotlib.cm.get_cmap('YlOrRd_r')
    SUBSTEPS = 10
    for i in range(N - 1):
        p0, p1 = pts[i], pts[i + 1]
        for s in range(SUBSTEPS):
            a0, a1 = s / SUBSTEPS, (s + 1) / SUBSTEPS
            sub_t = (i + (a0 + a1) / 2) / max(N - 1, 1)
            c = cmap(0.1 + sub_t * 0.8)
            seg = np.array([p0 + a0 * (p1 - p0), p0 + a1 * (p1 - p0)])
            ax.plot(seg[:, 0], seg[:, 1], '-', color=c, linewidth=9.0, zorder=10, solid_capstyle='round')

def draw_agent_trajs_bev(ax, boxes, predict_helper, sample_token, nusc):
    for box in boxes:
        inst = nusc.get('sample_annotation', box.token)['instance_token']
        fut = predict_helper.get_future_for_agent(inst, sample_token, seconds=3, in_agent_frame=True)
        if fut.shape[0] == 0:
            continue
        fut = convert_local_coords_to_global(fut, box.center, Quaternion(matrix=box.rotation_matrix))
        fut = np.vstack([box.center[:2][None], fut])
        ax.plot(fut[:, 0], fut[:, 1], '--', color='#00ff44', linewidth=2.0, alpha=0.9, zorder=6)

def vis_cameras_raw(sample_token, nusc, out_dir):
    sample = nusc.get('sample', sample_token)
    imgs = []
    for cam in CAM_ORDER:
        path = nusc.get_sample_data_path(sample['data'][cam])
        img = np.array(Image.open(path))
        imgs.append(img)
        fname = cam.lower().replace('cam_', '') + '.jpg'
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('black')
        ax.imshow(img)
        ax.axis('off')
        savefig(fig, os.path.join(out_dir, fname))

    fig, axes = plt.subplots(2, 3, figsize=(24, 9))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.01, wspace=0.01)
    for ax, img, cam in zip(axes.flat, imgs, CAM_ORDER):
        ax.imshow(img)
        ax.axis('off')
        ax.text(12, 38, cam.replace('CAM_', ''), color='white', fontsize=11, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=2))
    savefig(fig, os.path.join(out_dir, 'cameras.jpg'))

def vis_cameras_box_map(sample_token, nusc, map_annos, out_dir):
    sample = nusc.get('sample', sample_token)
    annotated = []
    
    for cam in CAM_ORDER:
        cam_tok = sample['data'][cam]
        sd_rec = nusc.get('sample_data', cam_tok)
        cs_cam = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        K = np.array(cs_cam['camera_intrinsic'])
        imsize = (sd_rec['width'], sd_rec['height'])
        data_path, boxes, _ = nusc.get_sample_data(cam_tok, box_vis_level=BoxVisibility.ANY, selected_anntokens=sample['anns'])

        img = cv2.imread(data_path)

        for box in boxes:
            if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
                draw_box3d_on_img(img, box, K, color=BOX_COLOR_BGR)

        if map_annos:
            draw_map_on_img(img, map_annos, nusc, sample_token, cam)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated.append(img_rgb)

        fname = cam.lower().replace('cam_', '') + '_box_map.jpg'
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('black')
        ax.imshow(img_rgb)
        ax.axis('off')
        savefig(fig, os.path.join(out_dir, fname))

    fig, axes = plt.subplots(2, 3, figsize=(24, 9))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.01, wspace=0.01)
    for ax, img, cam in zip(axes.flat, annotated, CAM_ORDER):
        ax.imshow(img)
        ax.axis('off')
        ax.text(12, 38, cam.replace('CAM_', ''), color='white', fontsize=11, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=2))
    savefig(fig, os.path.join(out_dir, 'cameras_box_map.jpg'))

def vis_front_traj(sample_token, nusc, ann_info, out_dir):
    cam = 'CAM_FRONT'
    sample = nusc.get('sample', sample_token)
    cam_tok = sample['data'][cam]
    data_path = nusc.get_sample_data_path(cam_tok)
    
    img = cv2.imread(data_path)

    if ann_info and 'gt_ego_fut_trajs' in ann_info:
        draw_ego_traj_on_img(img, ann_info['gt_ego_fut_trajs'], ann_info['gt_ego_fut_masks'], nusc, sample_token, cam, color=EGO_BGR, radius=8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor('black')
    ax.imshow(img_rgb)
    ax.axis('off')
    savefig(fig, os.path.join(out_dir, 'front_traj.jpg'))

def vis_bev_det_map(sample_token, nusc, map_annos, out_dir):
    sample = nusc.get('sample', sample_token)
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], selected_anntokens=sample['anns'])

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOR)
    bev_axis(ax)
    draw_map_bev(ax, map_annos)
    draw_boxes_bev(ax, boxes, color='#00ff00')
    ax.plot(0, 0, '^', color=EGO_COLOR, markersize=14, zorder=20)

    map_patches = [mpatches.Patch(color=MAP_RGB[i], label=MAP_CLASSES[i]) for i in range(len(MAP_CLASSES))]
    ax.legend(handles=map_patches + [mpatches.Patch(color='#00ff00', label='GT box')], loc='upper right', fontsize=8, ncol=1, facecolor='#1a1a2e', labelcolor='white', edgecolor='grey', framealpha=0.85)

    savefig(fig, os.path.join(out_dir, 'bev_det_map.jpg'))

def vis_occ(sample_token, nusc, occ_root, out_dir):
    sample = nusc.get('sample', sample_token)
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    occ_path = os.path.join(occ_root, scene_name, sample_token, 'labels.npz')

    if not os.path.exists(occ_path):
        return

    data = np.load(occ_path)
    if 'semantics' not in data:
        return

    sem = data['semantics']
    out_path = os.path.join(out_dir, 'occ.png')
    vis_occ_open3d(sem, out_path)

def vis_trajectory(sample_token, nusc, predict_helper, ann_info, out_dir):
    sample = nusc.get('sample', sample_token)
    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], selected_anntokens=sample['anns'])

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOR)
    bev_axis(ax)

    draw_boxes_bev(ax, boxes, color='#00ff00')
    draw_agent_trajs_bev(ax, boxes, predict_helper, sample_token, nusc)

    if ann_info and 'gt_ego_fut_trajs' in ann_info:
        draw_ego_traj_bev(ax, ann_info['gt_ego_fut_trajs'], ann_info['gt_ego_fut_masks'])
        if 'gt_ego_fut_cmd' in ann_info:
            cmd_names = ['LEFT','RIGHT','STRAIGHT','LANE_FOLLOW','CHANGE_LEFT','CHANGE_RIGHT']
            idx = int(np.argmax(ann_info['gt_ego_fut_cmd']))
            cmd = cmd_names[idx] if idx < len(cmd_names) else str(idx)
            ax.text(-BEV_RANGE+2, -BEV_RANGE+3, f'CMD: {cmd}', color='yellow', fontsize=13, fontweight='bold', bbox=dict(facecolor=BG_COLOR, alpha=0.7))

    ax.plot(0, 0, '^', color=EGO_COLOR, markersize=14, zorder=20)
    savefig(fig, os.path.join(out_dir, 'trajectory.jpg'))

def build_token_index(ann_file):
    with open(ann_file, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        infos = raw.get('infos', raw.get('data_infos', None))
        if infos is None:
            infos = list(raw.values())
    else:
        infos = raw
    index = {info['token']: info for info in infos if isinstance(info, dict) and 'token' in info}
    return index

def get_ann_info(index, token):
    if index is None or token not in index:
        return None
    info = index[token]
    ann = {}
    for k in ('gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status', 'target_point'):
        if k in info:
            ann[k] = np.array(info[k])
    if 'map_annos' in info:
        ann['map_annos'] = info['map_annos']
    return ann

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default='/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/nuscenes')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--ann_file', default='/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/infos/vad_nuscenes_infos_temporal_val.pkl')
    p.add_argument('--occ_root', default='/high_perf_store3/world-model/yongkangli/UniDriveVLA/data/nuscenes/gts')
    p.add_argument('--out_dir', default='outputs/gt_vis')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--token', default=None)
    p.add_argument('--tokens', nargs='+', default=None)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    predict_helper = PredictHelper(nusc)
    ann_index = build_token_index(args.ann_file) if args.ann_file else None

    MAP_LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
    nusc_maps = {}
    for loc in MAP_LOCATIONS:
        try:
            nusc_maps[loc] = NuScenesMap(dataroot=args.dataroot, map_name=loc)
        except Exception:
            pass

    if args.tokens:
        tokens = args.tokens
    elif args.token:
        tokens = [args.token]
    else:
        all_tokens = list(ann_index.keys()) if ann_index else [s['token'] for s in nusc.sample]
        step = max(1, len(all_tokens) // args.num_samples)
        tokens = all_tokens[::step][:args.num_samples]

    for i, tok in enumerate(tokens):
        out_dir = os.path.join(args.out_dir, tok)
        os.makedirs(out_dir, exist_ok=True)

        ann_info = get_ann_info(ann_index, tok)

        # Manually append 2 near-zero stopping waypoints for specific tokens
        MANUAL_STOP_TOKENS = {
            '17d4c42103af4608a0d58cfbc2fbc849': np.array([[0.0, 1.5], [0.0, 0.8]]),
        }
        if tok in MANUAL_STOP_TOKENS and ann_info and 'gt_ego_fut_trajs' in ann_info:
            extra = MANUAL_STOP_TOKENS[tok]
            ann_info['gt_ego_fut_trajs'] = np.vstack([ann_info['gt_ego_fut_trajs'], extra])
            ann_info['gt_ego_fut_masks'] = np.concatenate([ann_info['gt_ego_fut_masks'], np.ones(len(extra))])

        map_annos = (ann_info.get('map_annos') if ann_info else None) or None
        if not map_annos:
            map_annos = get_map_annos_from_api(nusc, nusc_maps, tok)

        vis_cameras_raw(tok, nusc, out_dir)
        vis_cameras_box_map(tok, nusc, map_annos, out_dir)
        vis_front_traj(tok, nusc, ann_info, out_dir)
        vis_bev_det_map(tok, nusc, map_annos, out_dir)
        vis_occ(tok, nusc, args.occ_root, out_dir)
        vis_trajectory(tok, nusc, predict_helper, ann_info, out_dir)

if __name__ == '__main__':
    main()