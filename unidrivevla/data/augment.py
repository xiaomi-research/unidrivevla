"""Training-split augmentation, configured by `augment`; everything is off by default.

The geometric one (`geometry.flip_p`, `geometry.rotate_deg`: the official ResizeCropFlipImage's flip and rotation)
is drawn once per sequence by the loader and applied by the model's transform, which moves each camera's projection
with its image so the 3D labels stay put. The other image augmentations are photometric.
"""

import random
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

EGO_STATE = ("ego_velocity", "ego_acceleration", "ego_angular_velocity")


class EgoDropout:
    """Zeroes the ego history with probability `history` and the ego velocity / acceleration / angular velocity with
    probability `status` (the command stays), so the planner cannot lean on ego-state shortcuts. Runs on the canonical
    sample, before the model's transform."""

    def __init__(self, history: float, status: float):
        self.history, self.status = history, status

    def __call__(self, features: dict, targets: dict):
        features = dict(features)
        if random.random() < self.history:
            features["history_trajectory"] = np.zeros_like(features["history_trajectory"])
        if random.random() < self.status:
            for key in EGO_STATE:
                features[key] = np.zeros_like(features[key])
        return features, targets


def option(section, key: str) -> float:
    return float(OmegaConf.select(section, key) or 0.0) if section is not None else 0.0


def ego_dropout(section) -> Optional[EgoDropout]:
    history, status = option(section, "history_dropout"), option(section, "ego_status_dropout")
    return EgoDropout(history, status) if history or status else None


def geometry(section):
    """Draw a geometric transform using the configured RNG stream.

    ``rng_mode=pr10_numpy`` reproduces the pinned PR10 ResizeCropFlipImage draws for the fixed nuScenes recipe,
    including draws for fixed resize/crop/3D-rotation values. The default Python RNG preserves other recipes.
    """
    flip, rotate = option(section, "geometry.flip_p"), option(section, "geometry.rotate_deg")
    rng_mode = (str(OmegaConf.select(section, "geometry.rng_mode", default="python"))
                if section is not None else "python")
    if rng_mode not in ("python", "pr10_numpy"):
        raise ValueError(f"unknown geometry RNG mode {rng_mode!r}")
    if rng_mode == "pr10_numpy" and (abs(flip - 0.5) > 1e-12 or abs(rotate - 5.4) > 1e-12):
        raise ValueError("PR10 nuScenes geometry requires rand_flip=0.5 and rot_lim=(-5.4, 5.4)")
    if not flip and not rotate and rng_mode != "pr10_numpy":
        return None

    def draw():
        if rng_mode == "pr10_numpy":
            # Exact PR10 get_augmentation order: resize, bottom percentage, horizontal crop, flip, rotation,
            # and 3D rotation. In this recipe the first three and the last are fixed at 0.6/0/0/0.
            np.random.uniform(0.6, 0.6)
            np.random.uniform(0.0, 0.0)
            np.random.uniform(0.0, 0.0)
            did_flip = bool(np.random.choice([0, 1]))
            angle = float(np.random.uniform(-5.4, 5.4))
            np.random.uniform(0.0, 0.0)
            return {"flip": did_flip, "rotate": angle}
        return {"flip": random.random() < flip, "rotate": random.uniform(-rotate, rotate)}

    draw.rng_mode = rng_mode
    return draw


class FrameGeometry:
    """Gives a frame its own geometric augmentation when the loader drew none for its sequence."""

    def __init__(self, draw):
        self.draw = draw

    def __call__(self, features: dict, targets: dict):
        return (features if "augment" in features else {**features, "augment": self.draw()}), targets


class PhotoMetricDistortion:
    """The official PhotoMetricDistortionMultiViewImage on one camera image: on float32 BGR, each step with probability
    0.5 (brightness, contrast before or after the HSV steps, saturation, hue, a channel permutation), drawing from numpy's
    generator in its order; clipped and truncated to uint8 once at the end, as the head's clamp(0, 255).byte()."""

    def __init__(self, brightness_delta=32.0, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue_delta=18.0):
        self.brightness_delta, self.contrast, self.saturation, self.hue_delta = brightness_delta, contrast, saturation, hue_delta

    def __call__(self, image):
        import cv2
        from numpy import random
        from PIL import Image

        img = np.asarray(image, dtype=np.float32)[:, :, ::-1].copy()
        if random.randint(2):
            img += random.uniform(-self.brightness_delta, self.brightness_delta)
        mode = random.randint(2)
        if mode == 1 and random.randint(2):
            img *= random.uniform(*self.contrast)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # mmcv.bgr2hsv
        if random.randint(2):
            img[..., 1] *= random.uniform(*self.saturation)
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if mode == 0 and random.randint(2):
            img *= random.uniform(*self.contrast)
        if random.randint(2):
            img = img[..., random.permutation(3)]
        return Image.fromarray(np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1]))


def image_augment(section):
    """PIL -> PIL transform, or None when every image augmentation is off."""
    import torchvision.transforms as T

    steps = []
    if option(section, "color_jitter.p"):
        jitter = T.ColorJitter(*(option(section, f"color_jitter.{key}") for key in ("brightness", "contrast", "saturation", "hue")))
        steps.append(T.RandomApply([jitter], p=option(section, "color_jitter.p")))
    if option(section, "grayscale_p"):
        steps.append(T.RandomGrayscale(option(section, "grayscale_p")))
    if option(section, "blur_p"):
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=option(section, "blur_p")))
    if option(section, "photometric"):  # last: it clips once, after every other step
        steps.append(PhotoMetricDistortion())
    return T.Compose(steps) if steps else None
