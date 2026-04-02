import os

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations3D_E2E(object):
    """Load Annotations3D for E2E.

    Args:
        with_hist_traj (bool, optional): Whether to load historical trajectory.
            Defaults to False.
    """
    def __init__(self,
                 with_hist_traj=False,
                 hist_steps=4,
                 use_cumsum=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_hist_traj = with_hist_traj
        self.hist_steps = hist_steps
        self.use_cumsum = use_cumsum

    def _load_hist_traj(self, results):
        """Private function to load historical trajectory.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded historical trajectory.
        """
        # Try to find gt_ego_his_trajs in results or results['ann_info']
        ego_his_trajs = results.get('gt_ego_his_trajs')
        if ego_his_trajs is None and 'ann_info' in results:
            ego_his_trajs = results['ann_info'].get('gt_ego_his_trajs')

        if ego_his_trajs is not None:
            ego_his_np = np.asarray(ego_his_trajs, dtype=np.float32)

            Th = min(self.hist_steps, ego_his_np.shape[0])
            ego_his_np = ego_his_np[:Th]

            if self.use_cumsum:
                cums = np.cumsum(ego_his_np, axis=0)
                hist_traj_np = cums - cums[-1:]
            else:
                hist_traj_np = ego_his_np.copy()
                hist_traj_np = hist_traj_np - hist_traj_np[-1:]

            results['gt_ego_his_trajs'] = ego_his_np
            # Map to 'hist_traj' as requested
            results['hist_traj'] = hist_traj_np
        else:
            results['hist_traj'] = None
            results['gt_ego_his_trajs'] = None

        return results

    def __call__(self, results):
        
        if self.with_hist_traj:
            results = self._load_hist_traj(results)

        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points
        return results

@PIPELINES.register_module()
class LoadOccWorldLabels(object):
    def __init__(self, data_root, input_dataset='gts'):
        self.data_root = data_root
        self.input_dataset = input_dataset

    def __call__(self, results):
        token = results['sample_idx']
        scene_name = results['scene_name']

        # Path format: data/nuscenes/gts/{scene_name}/{token}/labels.npz
        label_file = os.path.join(self.data_root, self.input_dataset, scene_name, token, 'labels.npz')

        if os.path.exists(label_file):
            try:
                label = np.load(label_file)
                # OccWorld stores semantics in 'semantics' key
                occ = label['semantics']
                results['gt_occ_dense'] = occ
            except Exception as e:
                print(f"Error loading OccWorld label for {token}: {e}")
                results['gt_occ_dense'] = None
        else:
            # print(f"Warning: OccWorld label not found for {token} at {label_file}")
            results['gt_occ_dense'] = None

        return results