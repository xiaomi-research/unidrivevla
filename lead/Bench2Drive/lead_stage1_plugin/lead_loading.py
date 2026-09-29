import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class LoadLeadMultiViewImageFromFiles(object):
    """Load LEAD RGB images and split the concatenated camera strip.

    LEAD stores the three perspective cameras in one horizontal 1152x384 JPEG.
    UniDriveVLA's Stage1 pipeline expects a list of per-camera images, so this
    loader splits the strip into three 384x384 views.
    """

    def __init__(self, to_float32=False, color_type="unchanged", num_cams=3):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.num_cams = num_cams

    def __call__(self, results):
        filenames = results["img_filename"]
        if isinstance(filenames, str):
            filenames = [filenames]

        if len(filenames) != 1:
            imgs = [mmcv.imread(name, self.color_type) for name in filenames]
        else:
            strip = mmcv.imread(filenames[0], self.color_type)
            if strip is None:
                raise FileNotFoundError(filenames[0])
            h, w = strip.shape[:2]
            if w % self.num_cams != 0:
                raise ValueError(
                    f"LEAD image width {w} is not divisible by num_cams={self.num_cams}: {filenames[0]}"
                )
            view_w = w // self.num_cams
            imgs = [strip[:, i * view_w:(i + 1) * view_w] for i in range(self.num_cams)]

        if self.to_float32:
            imgs = [img.astype(np.float32) for img in imgs]

        results["filename"] = filenames
        results["img"] = imgs
        results["img_shape"] = [img.shape for img in imgs]
        results["ori_shape"] = [img.shape for img in imgs]
        results["pad_shape"] = [img.shape for img in imgs]
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results


@PIPELINES.register_module()
class LeadTFV6ImageAugmentation(object):
    """Apply the official TFV6 appearance augmentation to the full RGB strip."""

    def __init__(self, probability=0.2, seed=0):
        self.probability = float(probability)
        self.seed = int(seed)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got {self.probability}"
            )
        try:
            import imgaug
            from imgaug import augmenters as ia
        except ImportError as exc:
            raise ImportError(
                "LeadTFV6ImageAugmentation requires the official LEAD dependency "
                "imgaug==0.4.0. Install it in the training environment."
            ) from exc

        imgaug.imgaug.seed(self.seed)
        probability = self.probability
        self.augmenter = ia.Sequential(
            [
                ia.Sometimes(probability, ia.GaussianBlur((0, 1.0))),
                ia.Sometimes(
                    probability,
                    ia.AdditiveGaussianNoise(
                        loc=0,
                        scale=(0.0, 0.05 * 255),
                        per_channel=0.5,
                    ),
                ),
                ia.Sometimes(
                    probability,
                    ia.Dropout((0.01, 0.1), per_channel=0.5),
                ),
                ia.Sometimes(
                    probability,
                    ia.Multiply((1 / 1.2, 1.2), per_channel=0.5),
                ),
                ia.Sometimes(
                    probability,
                    ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5),
                ),
                ia.Sometimes(
                    probability,
                    ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25),
                ),
            ],
            random_order=True,
        )

    def __call__(self, results):
        images = results["img"]
        if not images:
            return results
        heights = {image.shape[0] for image in images}
        channels = {image.shape[2] for image in images}
        if len(heights) != 1 or channels != {3}:
            raise ValueError(
                "TFV6 strip augmentation requires equal-height three-channel views, "
                f"got {[image.shape for image in images]}"
            )

        widths = [image.shape[1] for image in images]
        strip = np.concatenate(
            [np.clip(image, 0, 255).astype(np.uint8) for image in images],
            axis=1,
        )
        augmented = np.asarray(self.augmenter(image=strip), dtype=np.float32)
        split_points = np.cumsum(widths)[:-1]
        results["img"] = list(np.split(augmented, split_points, axis=1))
        return results


@PIPELINES.register_module()
class LeadSparse4DAdaptor(object):
    """Sparse4D adaptor variant that also tensorizes LEAD Stage1 planning keys."""

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(np.stack(input_dict["lidar2img"]))
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(np.stack(input_dict["cam_intrinsic"]))
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            "gt_map_labels",
            "gt_map_pts",
            "gt_agent_fut_trajs",
            "gt_agent_fut_masks",
        ]:
            if key in input_dict:
                input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False)

        for key in [
            "gt_ego_fut_trajs",
            "gt_ego_fut_masks",
            "gt_ego_fut_trajs_4hz",
            "gt_ego_fut_masks_4hz",
            "gt_ego_fut_trajs_5hz",
            "gt_ego_fut_masks_5hz",
            "gt_ego_fut_trajs_2hz",
            "gt_ego_fut_masks_2hz",
            "gt_ego_fut_cmd",
            "target_point_previous",
            "target_point",
            "target_point_next",
            "gt_route",
            "gt_route_masks",
            "gt_target_speed",
            "gt_brake",
            "ego_status",
            "ego_status_mask",
        ]:
            if key in input_dict:
                input_dict[key] = DC(
                    to_tensor(input_dict[key]),
                    stack=True,
                    cpu_only=False,
                    pad_dims=None,
                )

        for key in list(input_dict.keys()):
            if key.startswith("gt_ego_spat_trajs_") or key.startswith("gt_ego_spat_masks_"):
                input_dict[key] = DC(
                    to_tensor(input_dict[key]),
                    stack=True,
                    cpu_only=False,
                    pad_dims=None,
                )

        return input_dict

    def limit_period(self, val, offset=0.5, period=np.pi):
        return val - np.floor(val / period + offset) * period
