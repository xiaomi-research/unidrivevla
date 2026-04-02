import json

import numpy as np
import torch
import sys
import time
import math
from typing import List, Optional, Sequence, Tuple, Union, Dict

import open3d as o3d
from open3d import geometry
from mmengine.visualization import Visualizer
import matplotlib.pyplot as plt
import colorsys

class OccupancyVisualizer(Visualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 background_color=(0, 0, 0),
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 color_map=None):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        color_map = np.array(color_map)
        color_map = color_map[:, :3]
        self.background_color = background_color
        self.color_map = color_map

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)

    def right_callback(self, vis):
        self.flag_next = True
        return False

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.points_colors
            del self.view_control
            if hasattr(self, 'pcd'):
                del self.pcd

    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print(
                    'Playback continued, press [SPACE] to pause.', )
            else:
                print(
                    'Playback paused, press [SPACE] to continue.')
            self.flag_pause = not self.flag_pause
        return True

    def _initialize_o3d_vis(self):
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        glfw_key_escape = 256  # Esc
        glfw_key_space = 32  # Space
        glfw_key_right = 262  # Right
        o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.register_key_callback(glfw_key_escape, self.escape_callback)
        o3d_vis.register_key_action_callback(glfw_key_space,
                                             self.space_action_callback)
        o3d_vis.register_key_callback(glfw_key_right, self.right_callback)
        o3d_vis.create_window(width=1920, height=1080)

        self.view_control = o3d_vis.get_view_control()

        return o3d_vis

    def flow_to_color(self, vx, vy, max_magnitude=None):
        magnitude = np.sqrt(vx ** 2 + vy ** 2)
        angle = np.arctan2(vy, vx)

        # 将方向映射到色相 (0 到 1)
        hue = (angle + np.pi) / (2 * np.pi)

        # 将大小映射到亮度或饱和度
        if max_magnitude is None:
            max_magnitude = np.max(magnitude)
        if max_magnitude == 0:
            saturation = np.zeros_like(magnitude)
        else:
            saturation = np.clip(magnitude / max_magnitude + 1e-3, 0, 1)
        value = np.ones_like(saturation)

        # 将 HSV 转换为 RGB
        hsv = np.stack((hue, saturation, value), axis=-1)
        rgb = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

        # 将 RGB 转换为 0-255 范围的整数
        rgb = (rgb * 255).astype(np.uint8)

        return rgb

    def _voxel2points(self,
                      voxel,
                      voxel_flow,
                      voxelSize=(0.4, 0.4, 0.4),
                      range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                      ignore_labels=[17, 255]):
        if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = torch.zeros_like(voxel, dtype=torch.bool)
        for ignore_label in ignore_labels:
            mask = torch.logical_or(voxel == ignore_label, mask)
        mask = torch.logical_not(mask)
        occIdx = torch.where(mask)
        points = torch.cat((occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + range[0], \
                            occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
                            occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]), dim=1)
        if voxel_flow is not None:
            flow = voxel_flow[occIdx]
        else:
            flow = None
        return points, voxel[occIdx], flow

    def _voxel_profile(self,
                       voxel,
                       voxel_size=(0.4, 0.4, 0.4)
                       ):
        centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
        # centers = voxel
        wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                         torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                         torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
        yaw = torch.full_like(centers[:, 0:1], 0)
        return torch.cat((centers, wlh, yaw), dim=1)

    def _generate_the_ego_car(self):
        # ego_color black

        ego_range = [-1, -1, 0, 3, 1, 1.5]
        ego_voxel_size = [0.1, 0.1, 0.1]
        ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
        ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
        ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
        temp_x = np.arange(ego_xdim)
        temp_y = np.arange(ego_ydim)
        temp_z = np.arange(ego_zdim)
        ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
        ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
        ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
        ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
        ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
        # ego_rgb based on height, it generate rainbow color, the top is red, the bottom is blue
        ego_point_rgb = np.concatenate((ego_point_z, np.zeros_like(ego_point_z), 1 - ego_point_z), axis=-1)
        ego_point_rgb = ego_point_rgb * 255

        # ego_pint_rgb = np.zeros((ego_point_xyz.shape[0], 3))
        # ego_pint_rgb[:, 0] = (ego_point_xyz[:, 2] - ego_range[2]) / (ego_range[5] - ego_range[2])
        # ego_pint_rgb[:, 1] = 1 - ego_pint_rgb[:, 0]
        # ego_point_rgb = ego_pint_rgb * 255

        ego_point_xyzrgb = np.concatenate((ego_point_xyz, ego_point_rgb), axis=-1)

        return ego_point_xyzrgb

    def _my_compute_box_3d(self, center, size, heading_angle):
        h, w, l = size[:, 2], size[:, 0], size[:, 1]
        heading_angle = -heading_angle - math.pi / 2
        center[:, 2] = center[:, 2] + h / 2
        # R = rotz(1 * heading_angle)
        l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
        x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
        y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
        z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
        # corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
        corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
        corners_3d[..., 0] += center[:, 0:1]
        corners_3d[..., 1] += center[:, 1:2]
        corners_3d[..., 2] += center[:, 2:3]
        return corners_3d

    def voxellizer_points(self,
                          points,
                          ego_points,
                          bbox_corners,
                          linesets,
                          voxel_size=0.4,
                          frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                          vis_mode='replace',
                          points_size=4,
                          mode='xyzrgb',
                          points_color: Tuple[float] = (0.8, 0.8, 0.8),
                          show_color=True,
                          car_model_mesh=None,
                          ):
        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis()

        if hasattr(self, 'pcd') and vis_mode != 'add':
            self.o3d_vis.remove_geometry(self.pcd)
            # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            render_option.background_color = np.asarray(self.background_color)

        points = points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            if ego_points is not None:
                points = np.concatenate([points, ego_points], axis=0)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # rgb2bgr
            points_colors = points_colors[:, [2, 1, 0]]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        if show_color:
            voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            self.o3d_vis.add_geometry(voxelGrid)

        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        self.o3d_vis.add_geometry(line_sets)
        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        self.o3d_vis.add_geometry(mesh_frame)
        if car_model_mesh is not None:
            self.o3d_vis.add_geometry(car_model_mesh)

        if show_color:
            # update pcd
            self.o3d_vis.add_geometry(pcd)
            self.pcd = pcd
            self.points_colors = points_colors

    def vis_occ(self,
                occ_seg,
                occ_flow=None,
                save_path=None,
                view_json=None,
                ignore_labels: Optional[List[tuple]] = [0, 17],
                voxelSize=(0.4, 0.4, 0.4),
                points_size=50,
                points_color: Tuple[float] = (0.8, 0.8, 0.8),
                range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                vis_mode='add',
                wait_time=-1,
                car_model_mesh=None,
                use_car_model=False,
                ego_fut_trajs=None,
                show_color=True) -> None:
        if hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis()

        points, occ_voxel, occ_flow = self._voxel2points(occ_seg, occ_flow, ignore_labels=ignore_labels, voxelSize=voxelSize, range=range)
        points = points.numpy()
        occ_voxel = occ_voxel.numpy()
        if occ_flow is None:
            pts_color = self.color_map[occ_voxel.astype(int) % len(self.color_map)]
        else:
            vx = occ_flow[..., 0]
            vy = occ_flow[..., 1]
            pts_color = self.flow_to_color(vx, vy)
        seg_color = np.concatenate([points[:, :3], pts_color], axis=1)
        if use_car_model:
            if car_model_mesh is None:
                ego_points = self._generate_the_ego_car()
            else:
                ego_points = None
        else:
            ego_points = None

        bboxes = self._voxel_profile(torch.tensor(points), voxel_size=voxelSize)
        bboxes_corners = self._my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
        bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],
                              [3, 7]])  # lines along y-axis
        edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
        edges = edges + bases_[:, None, None]

        self.voxellizer_points(
            points=seg_color,
            voxel_size=voxelSize[0],
            ego_points=ego_points,
            bbox_corners=bboxes_corners.numpy(),
            linesets=edges.numpy(),
            mode='xyzrgb',
            vis_mode=vis_mode,
            show_color=show_color,
            car_model_mesh=car_model_mesh,
        )

        if ego_fut_trajs is not None:
            lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(ego_fut_trajs)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            self.o3d_vis.add_geometry(line_set)

        self.show(
            save_path,
            wait_time=wait_time,
            view_json=view_json
        )

    def show(self,
             save_path: Optional[str] = None,
             wait_time: int = -1,
             view_json=None) -> None:

        if view_json is not None:
            self.view_control.convert_from_pinhole_camera_parameters(view_json)

        if hasattr(self, 'o3d_vis'):
            if hasattr(self, 'view_port'):
                self.view_control.convert_from_pinhole_camera_parameters(
                    self.view_port)
            self.flag_exit = not self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            self.view_port = \
                self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
            if wait_time != -1:
                self.last_time = time.time()
                while time.time(
                ) - self.last_time < wait_time and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                while self.flag_pause and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501

            else:
                while not self.flag_next and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                self.flag_next = False
            self.o3d_vis.clear_geometries()
            try:
                del self.pcd
            except (KeyError, AttributeError):
                pass
            if save_path is not None:
                if not (save_path.endswith('.png')
                        or save_path.endswith('.jpg')):
                    save_path += '.png'
                self.o3d_vis.capture_screen_image(save_path)

            param = self.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            if self.flag_exit:
                self.o3d_vis.destroy_window()
                self.o3d_vis.close()
                self._clear_o3d_vis()
                sys.exit(0)
