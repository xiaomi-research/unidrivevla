"""python unidrivevla/tools/check_image_pipeline.py: the camera preprocessing of the nuScenes / Bench2Drive profiles
equals the official pipeline's on random 1600 x 900 images: ResizeCropFlipImage with its get_augmentation crops
(training with flip and rotation, evaluation), pixels and projection, and PhotoMetricDistortionMultiViewImage under the
same numpy draws, both up to the head's clamp(0, 255).byte(). The references below are the official code. Exits 1 on a
mismatch."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from unidrivevla.adapters.unidrivevla import Profile, camera_inputs  # noqa: E402
from unidrivevla.data.augment import PhotoMetricDistortion  # noqa: E402


def official_aug(H, W, final, train, flip=False, rotate=0.0):
    """get_augmentation with data_aug_conf resize_lim (0.6, 0.6), bot_pct_lim (0, 0)."""
    fH, fW = final
    resize = 0.6 if train else max(fH / H, fW / W)
    newW, newH = int(W * resize), int(H * resize)
    crop_h = int((1 - 0.0) * newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW))) if train else int(max(0, newW - fW) / 2)
    return dict(resize=resize, crop=(crop_w, crop_h, crop_w + fW, crop_h + fH), flip=flip, rotate=rotate)


def official_img_transform(img, aug):
    """ResizeCropFlipImage._img_transform on a uint8 image: the image and its 4x4 on lidar2img."""
    H, W = img.shape[:2]
    resize, crop, flip, rotate = aug["resize"], aug["crop"], aug["flip"], aug["rotate"]
    img = Image.fromarray(img).resize((int(W * resize), int(H * resize))).crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = np.array(img.rotate(rotate)).astype(np.float32)
    mat = np.eye(3)
    mat[:2, :2] *= resize
    mat[:2, 2] -= np.array(crop[:2])
    if flip:
        mat = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]) @ mat
    r = rotate / 180 * np.pi
    rot = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    centre = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    rot[:2, 2] = -rot[:2, :2] @ centre + centre
    out = np.eye(4)
    out[:3, :3] = rot @ mat
    return img, out


def official_photometric(img):
    """PhotoMetricDistortionMultiViewImage on one float32 BGR image (mmcv.bgr2hsv / hsv2bgr are cv2.cvtColor)."""
    from numpy import random
    if random.randint(2):
        img += random.uniform(-32, 32)
    mode = random.randint(2)
    if mode == 1:
        if random.randint(2):
            img *= random.uniform(0.5, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if random.randint(2):
        img[..., 1] *= random.uniform(0.5, 1.5)
    if random.randint(2):
        img[..., 0] += random.uniform(-18, 18)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if mode == 0:
        if random.randint(2):
            img *= random.uniform(0.5, 1.5)
    if random.randint(2):
        img = img[..., random.permutation(3)]
    return img


def head_bytes(img):
    return np.clip(img, 0, 255).astype(np.uint8)


ok = True
rng = np.random.default_rng(0)
intrinsic = np.array([[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]])
with tempfile.TemporaryDirectory() as tmp:
    for name in ("nuscenes", "b2d"):
        profile = Profile.from_config(OmegaConf.to_container(OmegaConf.load(PACKAGE / "config" / "profile" / f"{name}.yaml"),
                                                             resolve=True))
        final = tuple(profile.image_hw)
        bgr = rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8)
        path = f"{tmp}/{name}.png"
        Image.fromarray(bgr[:, :, ::-1]).save(path)
        cams = [dict(view=v, path=path, intrinsic=intrinsic.tolist(), size=(1600, 900), sensor2ego=np.eye(4).tolist())
                for v in profile.cameras]
        for train, flip, rotate in ((True, True, 3.7), (True, False, -5.1), (False, False, 0.0)):
            features = {"cameras": cams, **({"augment": {"flip": flip, "rotate": rotate}} if train else {})}
            out = camera_inputs(features, profile, train=train)
            memory = camera_inputs({**features, "cameras": [{**cam, "image": bgr[:, :, ::-1].copy()}
                                                            for cam in cams]}, profile, train=train)
            assert all(np.array_equal(out[key].numpy(), memory[key].numpy()) for key in out), "memory/path mismatch"
            ref_img, ref_mat = official_img_transform(bgr, official_aug(900, 1600, final, train, flip, rotate))
            k = np.eye(4)
            k[:3, :3] = intrinsic
            same_img = np.array_equal(out["img"][0].numpy().transpose(1, 2, 0), head_bytes(ref_img))
            mat_err = np.abs(out["projection_mat"][0].numpy() - (ref_mat @ k).astype(np.float32)).max()
            ok &= same_img and mat_err < 1e-3
            print(f"{name} {'train' if train else 'eval '} flip={flip} rotate={rotate:+.1f}: image identical {same_img}, "
                  f"projection max diff {mat_err:.1e}")
    photometric, same = PhotoMetricDistortion(), 0
    for seed in range(200):
        img = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        np.random.seed(seed)
        ref = head_bytes(official_photometric(img.astype(np.float32)))
        np.random.seed(seed)
        ours = np.asarray(photometric(Image.fromarray(img[:, :, ::-1])))[:, :, ::-1]
        same += np.array_equal(ours, ref)
    ok &= same == 200
    print(f"photometric distortion: {same}/200 images identical")
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
