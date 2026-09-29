import torch
import torch.nn as nn
import torchmetrics
from beartype import beartype
from torch.nn import functional as F

import lead.common.common_utils as common_utils
from lead.common.constants import SOURCE_DATASET_NAME_MAP, SourceDataset
from lead.training.config_training import TrainingConfig


class BEVDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        num_classes: int,
        device: torch.device,
        source_data: int,
    ):
        """Dense BEV decoder for BEV semantic segmentation.

        Args:
            config: Training configuration object
            num_classes: Number of semantic classes to predict
            device: Device to run the model on
            source_data: Source dataset enum value for which this decoder is responsible
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.source_data = source_data

        self.net = nn.Sequential(
            nn.Conv2d(
                self.config.bev_features_chanels,
                self.config.bev_features_chanels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.bev_features_chanels,
                num_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Upsample(
                size=(self.config.lidar_height_pixel, self.config.lidar_width_pixel),
                mode="bilinear",
                align_corners=False,
            ),
        )

        # Mask to mark occluded part of the BEV grid
        h, w = self.config.lidar_height_pixel, self.config.lidar_width_pixel
        valid_bev_pixels = torch.ones((h, w))
        center_y = h // 2
        center_x = -(self.config.min_x_meter * self.config.pixels_per_meter)
        for y in range(h):
            for x in range(w):
                valid_bev_pixels[y, x] = float(
                    common_utils.is_point_in_camera_frustum(
                        x=x,
                        y=y,
                        config=self.config,
                        center_x=center_x,
                        center_y=center_y,
                    ),
                )

        # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
        self.valid_bev_pixels = valid_bev_pixels.unsqueeze(0).to(
            dtype=self.config.torch_float_type,
            device=device,
        )  # (1, H, W)

    @beartype
    def compute_loss(self, pred: torch.Tensor, data: dict, loss: dict, log: dict):
        """
        Compute BEV semantic segmentation loss.

        Args:
            pred: (B, C, H, W) BEV semantic prediction tensor
            data: dict containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        Returns:
            None
        """
        if not self.config.use_bev_semantic:
            return

        # Mask for samples from the correct source dataset
        source_dataset = data["source_dataset"].to(
            pred.device,
            dtype=torch.long,
            non_blocking=True,
        )  # (B,)
        source_mask = (source_dataset == self.source_data).float()  # (B,)
        if source_mask.sum() == 0:
            return  # No samples from this source dataset in the batch
        dataset_name = SOURCE_DATASET_NAME_MAP[self.source_data]
        prefix = f"{dataset_name}_"
        if self.source_data == SourceDataset.CARLA:
            prefix = ""

        label = data[f"{prefix}bev_semantic"].to(
            pred.device,
            dtype=torch.long,
            non_blocking=True,
        )
        visible_label = self.mask_label(label).long()  # Mark invisible pixels to -1

        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Compute per-sample loss
            loss_bev_per_sample = F.cross_entropy(
                pred.float(),
                visible_label,
                reduction="none",
                ignore_index=-1,  # Ignore the invisible pixels in the back
            )  # (B, H, W)
            loss_bev_per_sample = loss_bev_per_sample.mean(dim=(1, 2))  # (B,)

            # Mask out losses from other data sources
            loss_bev = (
                loss_bev_per_sample * source_mask
            ).sum() / source_mask.sum().clamp(min=1)

        # Add dataset name prefix
        loss[f"{prefix}loss_bev_semantic"] = loss_bev

        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
        ):
            subset = source_mask.bool()
            log[f"{prefix}bev_semantic/output_min"] = pred.min().item()
            log[f"{prefix}bev_semantic/output_max"] = pred.max().item()
            pred_classes = pred[subset].argmax(dim=1)
            subset_label = visible_label[subset]
            valid_pixels = subset_label >= 0
            if valid_pixels.any():
                miou = torchmetrics.functional.jaccard_index(
                    pred_classes[valid_pixels],
                    subset_label[valid_pixels],
                    task="multiclass",
                    num_classes=self.num_classes,
                )
                f1 = torchmetrics.functional.f1_score(
                    pred_classes[valid_pixels],
                    subset_label[valid_pixels],
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                )
                log[f"{prefix}metric/bev_semantic_miou"] = miou.item()
                log[f"{prefix}metric/bev_semantic_f1"] = f1.item()

    @beartype
    def forward(self, bev_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the BEV decoder.

        Args:
            bev_feature_grid: (B, D, H, W) BEV feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) BEV feature grid after passing through the decoder
        """
        return self.net(bev_feature_grid)

    @beartype
    def mask_label(self, label: torch.Tensor):
        """
        Args:
            label: (B, H, W) BEV semantic label tensor
        Returns:
            label: (B, H, W) Masked BEV semantic label tensor
        """
        valid_bev_pixels = self.valid_bev_pixels
        visible_bev_semantic_label = (
            label * valid_bev_pixels
        )  # (B, H, W). Set invisible pixels to 0
        visible_bev_semantic_label = (
            valid_bev_pixels - 1
        ) + visible_bev_semantic_label  # Set invisible pixels to -1
        return visible_bev_semantic_label
