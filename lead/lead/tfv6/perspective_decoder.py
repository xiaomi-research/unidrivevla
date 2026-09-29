import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from beartype import beartype

from lead.common.constants import SOURCE_DATASET_NAME_MAP, SourceDataset
from lead.training.config_training import TrainingConfig


class PerspectiveDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        in_channels: int,
        out_channels: int,
        perspective_upsample_factor: int,
        modality: str,
        device: torch.device,
        source_data: int,
    ):
        """Decodes a low resolution perspective grid to a full resolution output. E.g. semantic segmentation, depth

        Args:
            config: Training configuration object
            in_channels: Feature channels of the input feature grid
            out_channels: Feature channels of the output feature grid
            perspective_upsample_factor: Upsampling factor from input feature grid to output
            modality: "semantic" or "depth"
            device: torch device
            source_data: Source dataset identifier (e.g., SourceDataset.CARLA)
        """
        super().__init__()
        self.modality = modality
        self.config = config
        self.device = device
        self.source_data = source_data
        self.scale_factor_0 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_0
        )
        self.scale_factor_1 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_1
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.config.deconv_channel_num_0, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_0,
                self.config.deconv_channel_num_1,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_1,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                out_channels,
                3,
                1,
                1,
                bias=True,
            ),
        )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        data: dict[str, torch.Tensor],
        loss: dict,
        log: dict,
    ):
        """Compute loss and metrics for the given modality.

        Args:
            prediction: (B, C, H, W) Prediction tensor
            data: dict containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        Returns:
            None
        """
        if self.config.use_semantic:
            # Mask for samples from the correct source dataset
            source_dataset = data["source_dataset"].to(
                prediction.device,
                dtype=torch.long,
                non_blocking=True,
            )  # (B,)
            source_mask = (source_dataset == self.source_data).float()  # (B,)

            if source_mask.sum() == 0:
                return  # No samples from this source dataset in the batch

            label = data[self.modality].to(
                prediction.device,
                dtype=torch.long,
                non_blocking=True,
            )

            # Compute loss per sample
            with torch.amp.autocast(device_type="cuda", enabled=False):
                if self.modality == "semantic":
                    loss_per_sample = F.cross_entropy(
                        prediction.float(),
                        label,
                        reduction="none",
                    )  # (B, H, W)
                    loss_per_sample = loss_per_sample.mean(dim=(1, 2))  # (B,)
                else:
                    loss_per_sample = F.l1_loss(
                        prediction.float(),
                        label.float(),
                        reduction="none",
                    )  # (B, H, W)
                    loss_per_sample = loss_per_sample.mean(dim=(1, 2))  # (B,)

                # Mask out losses from other data sources
                loss_value = (
                    loss_per_sample * source_mask
                ).sum() / source_mask.sum().clamp(min=1)

            # Add dataset name prefix
            prefix = SOURCE_DATASET_NAME_MAP[self.source_data]
            if self.source_data == SourceDataset.CARLA:
                prefix = ""
            loss.update({f"{prefix}loss_{self.modality}": loss_value})

            if (
                "iteration" in data
                and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
            ):
                subset = source_mask.bool()
                subset_pred = prediction[subset]
                subset_label = label[subset]
                log[f"{prefix}{self.modality}/output_min"] = subset_pred.min().item()
                log[f"{prefix}{self.modality}/output_max"] = subset_pred.max().item()
                if self.modality == "semantic":
                    miou = torchmetrics.functional.jaccard_index(
                        subset_pred,
                        subset_label,
                        task="multiclass",
                        num_classes=self.config.num_semantic_classes,
                    )
                    f1 = torchmetrics.functional.f1_score(
                        subset_pred,
                        subset_label,
                        task="multiclass",
                        num_classes=self.config.num_semantic_classes,
                        average="macro",
                    )
                    log[f"{prefix}metric/semantic_miou"] = miou.item()
                    log[f"{prefix}metric/semantic_f1"] = f1.item()
                else:
                    mae = torchmetrics.functional.mean_absolute_error(
                        subset_pred.float(),
                        subset_label.float(),
                    )
                    log[f"{prefix}metric/depth_mae"] = mae.item()

    def forward(self, data: dict, image_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the decoder.

        Args:
            data: dict containing the ground truth labels and masks
            image_feature_grid: (B, D, H, W) Image feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) Prediction tensor
        """
        x = self.deconv1(image_feature_grid)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor_0,
            mode="bilinear",
            align_corners=False,
        )
        x = self.deconv2(x)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor_1,
            mode="bilinear",
            align_corners=False,
        )
        x = self.deconv3(x)

        # Ensure output size matches expected size
        expected_h = self.config.final_image_height
        expected_w = self.config.final_image_width
        if x.shape[2] != expected_h or x.shape[3] != expected_w:
            height_error = abs(x.shape[2] - expected_h) / expected_h * 100
            width_error = abs(x.shape[3] - expected_w) / expected_w * 100
            if max(height_error, width_error) > 10:
                raise ValueError(
                    f"Output size mismatch too large: got ({x.shape[2]}, {x.shape[3]}), expected ({expected_h}, {expected_w})",
                )
            x = F.interpolate(
                x,
                size=(expected_h, expected_w),
                mode="bilinear",
                align_corners=False,
            )

        if self.modality == "depth":
            x = x.squeeze(1)
        return x
