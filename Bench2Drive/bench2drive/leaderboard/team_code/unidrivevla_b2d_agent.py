import os
import cv2
import json
import time
import math
import copy
import carla
import torch
import pathlib
import datetime
import importlib
import numpy as np

from PIL import Image
from scipy.optimize import fsolve
from pyquaternion import Quaternion
from torchvision import transforms as T

from mmcv import Config
from mmcv.runner import (
    load_checkpoint,
    wrap_fp16_model,
)
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate as mm_collate_to_batch_form

from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from team_code.planner import RoutePlanner as RoutePlannerVAD
from team_code.pid_controller import PIDController as PIDControllerVAD
from team_code.visualize import draw_bboxes3d
from leaderboard.autoagents import autonomous_agent

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

# B2D Sensor Calibration (same as HiP-AD)
LIDAR2IMG = {
    'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK': np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
}

LIDAR2CAM = {
    'CAM_FRONT': np.array([[1., 0., 0., 0.],
                           [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),

    'CAM_BACK': np.array([[-1., 0., 0., 0.],
                          [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61],
                          [0., 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]])
}

CAM2IMG = {
    'CAM_FRONT': np.array([[1142.51841 , 0., 800., 0.],
                           [0., 1142.51841, 450., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[1142.51840553, 0., 800., 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[1142.51841061, 0., 800, 0.],
                                 [0., 1142.51841, 450, 0.],
                                 [0., 0. ,1., 0.],
                                 [0., 0. ,0., 1.]]),

    'CAM_BACK': np.array([[560.166031, 0., 800., 0.],
                          [0. ,560.166031, 450., 0.],
                          [0. , 0., 1., 0.],
                          [0. , 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[1142.51840683, 0., 800., 0.],
                               [0., 1142.51841, 450., 0.],
                               [0., 0. , 1., 0.],
                               [0., 0. , 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[1142.51841041, 0., 800, 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0. , 1., 0.],
                                [0., 0. , 0., 1.]])
}

LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

unreal2cam = np.array([[0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0],
                               [0.0, 548.993771650447, 256.0, 0],
                               [0.0, 0.0, 1.0, 0],
                               [0, 0, 0, 1.0]])

topdown_extrinsics = np.array([[0.0, -0.0, -1.0, 50.0],
                               [0.0, 1.0, -0.0, 0.0],
                               [1.0, -0.0, 0.0, -0.0],
                               [0.0, 0.0, 0.0, 1.0]])

CAMERA = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# B2D Navigation Command Mapping
B2D_COMMAND_MAP = {
    0: "TURN LEFT",
    1: "TURN RIGHT",
    2: "GO STRAIGHT",
    3: "LANE FOLLOW",
    4: "CHANGE LANE LEFT",
    5: "CHANGE LANE RIGHT",
}

def get_entry_point():
    return 'UniDriveVLAB2DAgent'


class UniDriveVLAB2DAgent(autonomous_agent.AutonomousAgent):
    def sensors(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT'
            },
            # imu
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # gps
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
            },
            # speed
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
            },
        ]
        if IS_BENCH2DRIVE:
            sensors += [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 60.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                }]
        return sensors

    def setup(self, path_to_conf_file):
        self.device = "cuda"
        self.track = autonomous_agent.Track.SENSORS

        self.step = -1
        self.steer_step = 0
        self.last_steer = 0
        self.last_moving_step = -1
        self.last_moving_status = 0
        self.frame_rate = 20

        self.ckpt_path = path_to_conf_file.split('+')[1]
        self.config_path = path_to_conf_file.split('+')[0]
        self.save_name = path_to_conf_file.split('+')[-1]

        self.pidcontroller = PIDControllerVAD(
            turn_KP=1.0, turn_KI=0.75, turn_KD=0.0, turn_n=10,
            speed_KP=3.0, speed_KI=0.5, speed_KD=1.0, speed_n=10,
            brake_speed=0.4, brake_ratio=1.1,
            clip_delta=0.25, max_throttle=0.75,
            aim_dist_low=2.0, aim_dist_high=8.0,
            anti_windup=True,
        )

        self.wall_start = time.time()
        self.initialized = False
        self.use_bgr_img = True
        self.data_aug_conf = None

        cfg = Config.fromfile(self.config_path)

        # Configure for UniDriveVLA
        if hasattr(cfg, 'model'):
            if hasattr(cfg.model, 'planning_head'):
                # VLA head specific config
                cfg.model.planning_head.dataset_type = "bench2drive"
                # enable close-loop bank rotation (20Hz CARLA / 2Hz training = 10 banks)
                if hasattr(cfg.model.planning_head, 'unified_decoder_cfg'):
                    cfg.model.planning_head.unified_decoder_cfg.with_close_loop = True

        if hasattr(cfg, 'data_aug_conf'):
            self.data_aug_conf = cfg.data_aug_conf

        if hasattr(cfg, "plugin") and hasattr(cfg, "plugin_dir") and cfg.plugin:
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        if self.ckpt_path is not None:
            checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu')

        self.model.cuda()
        self.model.eval()
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.brake = 0.0
        control.throttle = 0.0

        self.prev_control = control
        self.prev_control_cache = []

        self.is_visualize = os.environ.get('IS_VISUALIZE', '0') == '1'
        self.visualize_interval = 2

        if self.is_visualize:
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'metas').mkdir()
            (self.save_path / 'images').mkdir()
        else:
            self.save_path = pathlib.Path(os.environ['SAVE_PATH'])

        self.lidar2cam = LIDAR2CAM
        self.lidar2img = LIDAR2IMG
        self.lidar2ego = LIDAR2EGO
        self.cam2img = CAM2IMG
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = ((lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)) -
                       math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA
                       * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180)
                       * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlannerVAD(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERA:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        target_xy = target_xy_next = near_node
        command = command_next = near_command

        if (math.isnan(compass) == True):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'bev': bev,
            'speed': speed,
            'compass': compass,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'target_xy': target_xy,
            'target_xy_next': target_xy_next,
            'command': command,
            'command_next': command_next
        }

        return result

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self.init()
        tick_data = self.tick(input_data)
        inputs = {}
        inputs['img'] = []
        inputs['folder'] = ''
        inputs['scene_token'] = ''
        inputs['lidar2img'] = []
        inputs['lidar2cam'] = []
        inputs['frame_idx'] = 0
        inputs['timestamp'] = self.step / self.frame_rate
        for cam in CAMERA:
            inputs['lidar2img'].append(self.lidar2img[cam])
            inputs['lidar2cam'].append(self.lidar2cam[cam])
            img = tick_data['imgs'][cam]
            if self.use_bgr_img:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs['img'].append(img)
        inputs['lidar2img'] = np.stack(inputs['lidar2img'], axis=0)
        inputs['lidar2cam'] = np.stack(inputs['lidar2cam'], axis=0)
        inputs['aug_config'] = self.get_augmentation()

        # data
        ego_pos = [tick_data['pos'][0], -tick_data['pos'][1]]

        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi / 2
        ego_theta_degree = ego_theta / np.pi * 180
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        ego_speed = tick_data['speed']
        acceleration = tick_data['acceleration']
        ego_accel = [acceleration[0], -acceleration[1], acceleration[2]]
        angular_velocity = -tick_data['angular_velocity']

        # status
        ego_status = np.zeros(6)
        ego_status[0] = ego_speed
        ego_status[1:3] = ego_accel[:2]
        ego_status[3:5] = angular_velocity[:2]
        ego_status[5] = self.pid_metadata['steer'] if hasattr(self, 'pid_metadata') else 0
        inputs['ego_status'] = np.array(ego_status, np.float32)

        # command - B2D has 6 commands (0-5)
        command = tick_data['command']
        command_next = tick_data['command_next']
        if command < 0: command = 4
        command -= 1
        command_onehot = np.zeros(6, np.float32)
        command_onehot[command] = 1
        inputs['command'] = command
        inputs['gt_ego_fut_cmd'] = command_onehot

        # target_point removed: route planner waypoints caused goal point to land on
        # obstacles when route was sparse. Model relies on gt_ego_fut_cmd for navigation.

        # metas
        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = ego_pos

        lidar2global = ego2world @ self.lidar2ego
        inputs['l2g_r_mat'] = lidar2global[0:3, 0:3]
        inputs['l2g_t'] = lidar2global[0:3, 3]
        inputs['lidar2global'] = lidar2global
        image_h, image_w = self.data_aug_conf['final_dim']
        inputs['image_wh'] = np.array([[image_w, image_h] for _ in CAMERA], dtype=np.float32)

        inputs = self.inference_only_pipeline(inputs)
        inputs = mm_collate_to_batch_form([inputs], samples_per_gpu=1)
        for key, value in inputs.items():
            if isinstance(value, DataContainer):
                inputs[key] = value.data[0]
            elif isinstance(value[0], DataContainer):
                inputs[key] = value[0].data
            else:
                inputs[key] = value
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        # Run model - UniDriveVLA uses the same interface
        outputs = self.model(
            img=inputs['img'],
            img_metas=inputs['img_metas'],
            projection_mat=inputs['projection_mat'],
            ego_status=inputs.get('ego_status'),
            gt_ego_fut_cmd=inputs['gt_ego_fut_cmd'],
            image_wh=inputs['image_wh'],
            timestamp=inputs['timestamp'],
            rescale=True,
            return_loss=False,
        )

        # control - aligned with HiP-AD
        plan_temp_name = 'plan_speed_5hz'
        plan_spat_name = 'plan_spat_2m'

        pred_temp_traj = None
        if 'final_planning' in outputs[0]['img_bbox']:
            pred_temp_traj = outputs[0]['img_bbox']['final_planning'].cpu().numpy()
            outputs[0]['img_bbox']['temporal_planning'] = outputs[0]['img_bbox']['final_planning']
        elif plan_temp_name in outputs[0]['img_bbox']:
            pred_temp_traj = outputs[0]['img_bbox'][plan_temp_name].cpu().numpy()
            outputs[0]['img_bbox']['temporal_planning'] = outputs[0]['img_bbox'][plan_temp_name]

        pred_spat_traj = None
        if plan_spat_name in outputs[0]['img_bbox']:
            pred_spat_traj = outputs[0]['img_bbox'][plan_spat_name].cpu().numpy()
            outputs[0]['img_bbox']['spatial_planning'] = outputs[0]['img_bbox'][plan_spat_name]

        # target_point removed; use trajectory endpoint direction as dummy target
        # (use_target_to_aim=False in PID so this value is not actually used for steering)
        dummy_target = pred_temp_traj[-1]

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
            pred_temp_traj, ego_speed, dummy_target)

        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)

        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'unidrivevla_b2d'
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan_temp'] = pred_temp_traj.tolist() if isinstance(pred_temp_traj, np.ndarray) else pred_temp_traj
        self.pid_metadata['plan_spat'] = pred_spat_traj.tolist() if isinstance(pred_spat_traj, np.ndarray) else pred_spat_traj
        self.pid_metadata['command'] = command
        self.pid_metadata['command_text'] = B2D_COMMAND_MAP.get(command, "LANE FOLLOW")
        self.pid_metadata['target_point'] = [float(dummy_target[0]), float(dummy_target[1])]

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info

        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

        if self.is_visualize and self.step % self.visualize_interval == 0:
            self.visualize(tick_data, inputs, outputs, pred_temp_traj, dummy_target)

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def visualize(self, tick_data, input_batch, output_batch, pred_planning, target_point):
        rw, rh = 960//2, 540//2

        pred_planning = np.concatenate([np.array([[0, 0]]), pred_planning])

        with_spatial_planning = False
        if 'spatial_planning' in output_batch[0]['img_bbox']:
            with_spatial_planning = True
            spatial_planning = output_batch[0]['img_bbox']['spatial_planning'].cpu().numpy()
            spatial_planning = np.concatenate([np.array([[0, 0]]), spatial_planning])

        ### plot img
        imgs = []
        for cam in CAMERA:
            img = tick_data['imgs'][cam]

            # draw agent box on image
            if 'boxes_3d' in output_batch[0]['img_bbox']:
                pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)

                img = draw_bboxes3d(img, pred_bboxes3d,
                                    intrinsic=self.cam2img[cam],
                                    extrinsic=self.lidar2cam[cam],
                                    color=(0, 0, 255))

            # draw ego traj on image
            if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                # spatial traj
                if with_spatial_planning:
                    coord3d = np.concatenate([spatial_planning,
                                              np.ones_like(pred_planning[:, :1]) * -1.8,
                                              np.ones_like(pred_planning[:, :1])], axis=1)
                    coord2d = (self.lidar2img[cam] @ coord3d.T).T
                    coord2d = coord2d[coord2d[:, 2] > 1e-5]
                    coord2d = coord2d[:, :2] / coord2d[:, 2:3]
                    img = self.draw_trajectory(img, coord2d, line_color=(0, 200, 255), line_thickness=10,
                                               point_color=(0, 200, 255), point_thickness=8, point_radius=8)

                # temporal traj
                coord3d = np.concatenate([pred_planning,
                                          np.ones_like(pred_planning[:, :1]) * -1.8,
                                          np.ones_like(pred_planning[:, :1])], axis=1)
                coord2d = (self.lidar2img[cam] @ coord3d.T).T
                coord2d = coord2d[coord2d[:, 2] > 1e-5]
                coord2d = coord2d[:, :2] / coord2d[:, 2:3]
                img = self.draw_trajectory(img, coord2d, line_color=(255, 0, 0), line_thickness=10,
                                           point_color=(255, 0, 0), point_thickness=8, point_radius=8)


            # draw target point on image
            if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                cmd_coord3d = np.concatenate([target_point,
                                              np.ones_like(target_point[:1]) * -1.8,
                                              np.ones_like(target_point[:1])], axis=0)
                cmd_coord2d = (self.lidar2img[cam] @ cmd_coord3d.T).T
                cmd_coord2d = cmd_coord2d[:2] / np.abs(cmd_coord2d[2:3])
                if np.isfinite(cmd_coord2d[0]) and np.isfinite(cmd_coord2d[1]):
                    img = cv2.circle(img, (int(cmd_coord2d[0]), int(cmd_coord2d[1])), 7, (255, 105, 120), 6)

            # text and resize
            img = cv2.putText(img, cam, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            img = cv2.resize(img, (rw, rh))
            imgs.append(img)

        ### plot bev
        bev_img = tick_data['bev']
        bev_dict = self.sensors()[-1]
        if bev_dict['id']=='bev':
            ## agent
            # draw agent box on bev
            if 'boxes_3d' in output_batch[0]['img_bbox']:
                pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)
                for idx, (pred_bbox3d, pred_label3d) in enumerate(zip(pred_bboxes3d, pred_labels3d)):
                    box = self.convert_bev_bbox(pred_bbox3d, bev_dict)
                    cv2.polylines(bev_img, [box], isClosed=True, color=(0, 0, 255), thickness=1)

                    # draw agent motion traj on bev
                    if pred_trajs3d is not None and pred_label3d in [0, 1, 2, 3]:
                        pred_traj = np.concatenate([pred_bbox3d[:2][None], pred_trajs3d[idx]])
                        bev_coord = self.convert_bev_coord(pred_traj, bev_dict)
                        bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 0), line_thickness=1,
                                                       point_color=(0, 200, 0), point_thickness=1, point_radius=2)

            ### ego
            # draw ego box on bev
            box = self.convert_bev_bbox((0, 0, 0, 1.84, 4.89, 1.49, 0), bev_dict)
            cv2.polylines(bev_img, [box], isClosed=True, color=(255, 255, 0), thickness=1)

            # draw ego spatial traj
            if with_spatial_planning:
                bev_coord = self.convert_bev_coord(spatial_planning, bev_dict)
                bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 255), line_thickness=1,
                                               point_color=(0, 200, 255), point_thickness=1, point_radius=2)
            # draw ego temporal traj
            bev_coord = self.convert_bev_coord(pred_planning, bev_dict)
            bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(255, 0, 0), line_thickness=1,
                                           point_color=(255, 0, 0), point_thickness=1, point_radius=2)

            # draw bev target point
            bev_coord = self.convert_bev_coord(target_point, bev_dict)
            cv2.circle(bev_img, (int(bev_coord[0]), int(bev_coord[1])), 3, (255, 105, 120), 2)

        # text and resize
        cmd_str = str(tick_data['command']).split('.')[-1]
        bev_img = cv2.putText(bev_img, cmd_str, (15, bev_dict['height']-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # text and resize
        # bev_img = cv2.putText(bev_img, "cmd:{}".format(self.pid_metadata['command']), (15, bev_dict['height']-95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "speed:{:.2f}".format(self.pid_metadata['speed']), (15, bev_dict['height']-75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "steer:{:.2f}".format(self.pid_metadata['steer']), (15, bev_dict['height']-55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "throttle:{:.2f}".format(self.pid_metadata['throttle']), (15, bev_dict['height']-35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "brake:{:.2f}".format(self.pid_metadata['brake']), (15, bev_dict['height']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, str(tick_data['command']), (15, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        bev_img = cv2.resize(bev_img, (rh*2, rh*2))

        # merge
        front = imgs[0]
        front_left = imgs[1]
        front_right = imgs[2]
        back = imgs[3]
        back_left = imgs[4]
        back_right = imgs[5]

        line1 = np.hstack([front_left, front, front_right])
        line2 = np.hstack([back_right, back, back_left])
        merge_img = np.vstack([line1, line2])
        merge_img = np.hstack([merge_img, bev_img])

        frame = self.step

        Image.fromarray(merge_img).save(self.save_path / 'images' / ('%04d.jpg' % frame))

        outfile = open(self.save_path / 'metas' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def get_bboxes(self, output_batch):
        pred_bboxes3d = copy.deepcopy(output_batch[0]['img_bbox']['boxes_3d'].cpu().numpy())
        pred_scores3d = copy.deepcopy(output_batch[0]['img_bbox']['scores_3d'].cpu().numpy())
        pred_labels3d = copy.deepcopy(output_batch[0]['img_bbox']['labels_3d'].cpu().numpy())

        pred_mask3d = pred_scores3d > 0.3
        pred_bboxes3d = pred_bboxes3d[pred_mask3d]
        pred_labels3d = pred_labels3d[pred_mask3d]

        pred_trajs3d = None
        if 'trajs_3d' in output_batch[0]['img_bbox']:
            pred_trajs3d = copy.deepcopy(output_batch[0]['img_bbox']['trajs_3d'].cpu().numpy())
            pred_trajs_score = copy.deepcopy(output_batch[0]['img_bbox']['trajs_score'].cpu().numpy())

            pred_trajs3d = pred_trajs3d[pred_mask3d]
            pred_trajs_score = pred_trajs_score[pred_mask3d]

            # select top-one
            pred_trajs3d = pred_trajs3d[np.arange(len(pred_trajs3d)), pred_trajs_score.argmax(-1)]

        return pred_bboxes3d, pred_labels3d, pred_trajs3d

    def draw_trajectory(self, image, coord2d, line_color=(255, 0, 0), line_thickness=2,
                        point_color=(255, 0, 0), point_thickness=2, point_radius=None):

        for i in range(len(coord2d) - 1):
            point_start = coord2d[i]
            point_end = coord2d[i + 1]

            point_start = (int(point_start[0]), int(point_start[1]))
            point_end = (int(point_end[0]), int(point_end[1]))

            cv2.line(image, point_start, point_end, line_color, line_thickness)

            # draw points
            if point_radius is not None:
                cv2.circle(image, point_start, point_radius, point_color, point_thickness)
                cv2.circle(image, point_end, point_radius, point_color, point_thickness)

        return image

    def convert_bev_bbox(self, bbox, bev_dict):
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        center = (bev_width / 2 + bbox[0] * bev_ratio_w,
                  bev_height / 2 - bbox[1] * bev_ratio_h)
        size = (bbox[3] * bev_ratio_w, bbox[4] * bev_ratio_h)
        angle = - bbox[6] / np.pi * 180

        rect = (center, size, angle)
        bev_bbox = cv2.boxPoints(rect)
        bev_bbox = np.int0(bev_bbox)

        return bev_bbox

    def convert_bev_coord(self, coord3d, bev_dict):
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        if len(coord3d.shape) == 1:
            coord_x = coord3d[0:1] * bev_ratio_w
            coord_y = coord3d[1:2] * bev_ratio_h
        else:
            coord_x = coord3d[:, 0:1] * bev_ratio_w
            coord_y = coord3d[:, 1:2] * bev_ratio_h

        offset_u = bev_width / 2 + coord_x
        offset_v = bev_height / 2 - coord_y
        coord2d = np.concatenate([offset_u, offset_v], axis=-1)

        return coord2d
