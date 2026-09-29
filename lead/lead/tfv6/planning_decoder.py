import logging
import math

import jaxtyping as jt
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

import lead.common.common_utils as common_utils
from lead.common.constants import RadarLabels
from lead.tfv6 import transfuser_utils as fn
from lead.training.config_training import TrainingConfig

logger = logging.getLogger(__name__)


class PlanningDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        input_bev_channels: int,
        config: TrainingConfig,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.planning_context_encoder = PlanningContextEncoder(
            config=self.config,
            input_bev_channels=input_bev_channels,
            device=self.device,
        )

        # Number of queries: route + waypoints + target_speed (flexible based on config)
        num_queries = 0
        if self.config.predict_spatial_path:
            num_queries += self.config.num_route_points_prediction
        if self.config.predict_temporal_spatial_waypoints:
            num_queries += self.config.num_way_points_prediction
        if self.config.predict_target_speed:
            num_queries += 1

        self.query = nn.Parameter(
            torch.zeros(
                1,
                num_queries,
                self.config.transfuser_token_dim,
            ),
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.config.transfuser_token_dim,
                self.config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=self.config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(self.config.transfuser_token_dim),
        )

        # Only create decoders if needed
        if self.config.predict_spatial_path:
            self.route_decoder = nn.Linear(config.transfuser_token_dim, 2)
        if self.config.predict_temporal_spatial_waypoints:
            self.wp_decoder = nn.Linear(config.transfuser_token_dim, 2)
            if self.config.use_navsim_data:
                self.heading_decoder = nn.Linear(config.transfuser_token_dim, 1)
        if self.config.predict_target_speed:
            self.target_speed_decoder = nn.Sequential(
                nn.Linear(
                    self.config.transfuser_token_dim,
                    self.config.transfuser_token_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    self.config.transfuser_token_dim,
                    len(self.config.target_speed_classes),
                ),
            )

        self.tp_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "bs bev_dim height_bev width_bev"],
        radar_features: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
    ) -> tuple[
        jt.Float[torch.Tensor, "B n_checkpoints 2"] | None,
        jt.Float[torch.Tensor, "B n_waypoints 2"],
        jt.Float[torch.Tensor, "B speed_classes"] | None,
        jt.Float[torch.Tensor, " B"] | None,
        jt.Float[torch.Tensor, "B n_waypoints"] | None,
    ]:
        """
        Args:
            bev_features: BEV features.
            radar_features: Radar features.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            route: Spatial path.
            waypoints: Spatial and temporal path.
            target_speed: Target speed distribution.
            target_speed_scalar: Target speed in m/s.
            headings: Heading predictions (if using NavSim data).
        """
        self.kv = context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            radar_logits=radar_features,
            radar_predictions=radar_predictions,
            data=data,
            log=log,
        )

        bs = context_tokens.shape[0]

        queries = self.transformer_decoder(self.query.repeat(bs, 1, 1), context_tokens)

        # Split the queries flexibly based on what we're predicting
        query_idx = 0
        route = None
        waypoints = None
        headings = None
        target_speed_dist = None
        target_speed_scalar = None

        if self.config.predict_spatial_path:
            route_queries = queries[
                :,
                query_idx : query_idx + self.config.num_route_points_prediction,
            ]
            route = torch.cumsum(self.route_decoder(route_queries), 1)
            query_idx += self.config.num_route_points_prediction

        if self.config.predict_temporal_spatial_waypoints:
            waypoints_queries = queries[
                :,
                query_idx : query_idx + self.config.num_way_points_prediction,
            ]
            waypoints = torch.cumsum(self.wp_decoder(waypoints_queries), 1)
            if self.config.use_navsim_data:
                headings = torch.cumsum(self.heading_decoder(waypoints_queries), 1)
            query_idx += self.config.num_way_points_prediction

        if self.config.predict_target_speed:
            target_speed_query = queries[:, query_idx]
            target_speed_dist = self.target_speed_decoder(target_speed_query)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                target_speed_softmax = torch.softmax(target_speed_dist.float(), dim=-1)
                target_speed_scalar = decode_two_hot(
                    target_speed_softmax,
                    self.config.target_speed_classes,
                    self.device,
                )

        return (
            route,
            waypoints,
            target_speed_dist,
            target_speed_scalar,
            headings.squeeze(-1) if headings is not None else None,
        )

    @beartype
    def compute_loss(self, predictions, data: dict, loss: dict, log: dict):
        # Prepare loss dictionary
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.config.predict_temporal_spatial_waypoints:
                waypoints_label = data["future_waypoints"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )[:, : self.config.num_way_points_prediction]

                loss["loss_spatio_temporal_waypoints"] = F.l1_loss(
                    predictions.pred_future_waypoints.float(),
                    waypoints_label.float(),
                    reduction="none",
                ).mean()

                if self.config.use_navsim_data:
                    heading_label = data["future_yaws"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    )
                    loss["loss_spatio_temporal_waypoints"] = (
                        loss["loss_spatio_temporal_waypoints"]
                        + F.l1_loss(
                            predictions.pred_headings.float(),
                            heading_label.float(),
                        ).mean()
                    )

            if self.config.predict_target_speed:
                brake_label = data["brake"].to(
                    self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                target_speed_distribution = encode_two_hot(
                    data["target_speed"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    ),
                    self.config.target_speed_classes,
                    brake=brake_label,
                )
                loss["loss_target_speed"] = F.cross_entropy(
                    predictions.pred_target_speed_distribution.float(),
                    target_speed_distribution,
                )

            if self.config.predict_spatial_path:
                route_label = data["route"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )
                loss["loss_spatial_route"] = F.l1_loss(
                    predictions.pred_route.float(),
                    route_label.float(),
                )  # ADE
                loss["loss_spatial_route"] += F.l1_loss(
                    predictions.pred_route[:, -1, :].float(),
                    route_label[:, -1, :].float(),
                )  # FDE

        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
        ):
            if self.config.predict_spatial_path:
                route_label = data["route"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )
                log.update(
                    {
                        "metric/route_ade": common_utils.average_displacement_error(
                            predictions.pred_route,
                            route_label,
                        ),
                        "metric/route_fde": common_utils.final_displacement_error(
                            predictions.pred_route,
                            route_label,
                        ),
                    },
                )

            if self.config.predict_target_speed:
                brake_label = data["brake"].to(
                    self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                target_speed_distribution = encode_two_hot(
                    data["target_speed"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    ),
                    self.config.target_speed_classes,
                    brake=brake_label,
                )
                target_speed_labels = decode_two_hot(
                    target_speed_distribution,
                    self.config.target_speed_classes,
                    self.device,
                )
                log.update(
                    {
                        "metric/target_speed_error": torch.mean(
                            torch.abs(
                                predictions.pred_target_speed_scalar
                                - target_speed_labels,
                            ),
                        ).item(),
                        "metric/target_speed_correlation": torch.corrcoef(
                            torch.stack(
                                [
                                    predictions.pred_target_speed_scalar,
                                    target_speed_labels,
                                ],
                            ),
                        )[0, 1].item(),
                    },
                )

            if self.config.predict_temporal_spatial_waypoints:
                waypoints_label = data["future_waypoints"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )[:, : self.config.num_way_points_prediction]
                log.update(
                    {
                        "metric/waypoints_ade": common_utils.average_displacement_error(
                            predictions.pred_future_waypoints,
                            waypoints_label,
                        ),
                        "metric/waypoints_fde": common_utils.final_displacement_error(
                            predictions.pred_future_waypoints,
                            waypoints_label,
                        ),
                    },
                )

                if self.config.use_navsim_data:
                    heading_label = data["future_yaws"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    )
                    log["metric/heading_ade"] = common_utils.average_displacement_error(
                        predictions.pred_headings,
                        heading_label,
                    )


@beartype
def decode_two_hot(
    two_hot_label: jt.Float[torch.Tensor, "B C"],
    class_values: list[float],
    device: torch.device,
) -> jt.Float[torch.Tensor, " B"]:
    """Decode a two-hot encoded tensor into a scalar representation.

    Args:
        two_hot_label: The two-hot encoded tensor. Must be between 0 and 1 and sum to 1 along the last dimension.
        class_values: List of class values (e.g., target_speeds or throttle_classes).
        device: Device to place tensors on.

    Returns:
        The decoded scalar tensor.
    """
    classes = torch.tensor(
        class_values,
        device=device,
        dtype=two_hot_label.dtype,
    ).unsqueeze(0)
    decoded = (two_hot_label * classes).sum(axis=-1)
    return decoded


@beartype
def encode_two_hot(
    scalar_values: jt.Float[torch.Tensor, " B"],
    class_values: list[float],
    brake: jt.Bool[torch.Tensor, " B"],
) -> jt.Float[torch.Tensor, "B C"]:
    """Encode scalar values into two-hot representation with linear interpolation.

    Args:
        scalar_values: Scalar values to encode (e.g., speeds or throttle values).
        class_values: List of class bin values (e.g., [0.0, 4.0, 8.0, ...] for speeds).
        brake: Optional boolean mask. If provided, positions where True will be encoded as class 0.

    Returns:
        Two-hot encoded distribution.
    """
    assert all(scalar_values >= 0.0)
    target_speeds = torch.tensor(
        class_values,
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels = torch.zeros(
        len(scalar_values),
        len(target_speeds),
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels[brake, 0] = 1.0
    non_brake = ~brake
    scalars = scalar_values[non_brake]
    last_bin = scalars >= target_speeds[-1]
    labels[non_brake & (scalar_values >= target_speeds[-1]), -1] = 1.0

    # Interpolation between bins
    interp_mask = ~last_bin
    if interp_mask.any():
        interp_speeds = scalars[interp_mask]
        upper_idx = torch.searchsorted(target_speeds, interp_speeds, right=False)
        lower_idx = upper_idx - 1

        lower_val = target_speeds[lower_idx]
        upper_val = target_speeds[upper_idx]

        lower_weight = (upper_val - interp_speeds) / (upper_val - lower_val)
        upper_weight = (interp_speeds - lower_val) / (upper_val - lower_val)

        row_idx = torch.where(non_brake)[0][interp_mask]
        labels[row_idx, lower_idx] = lower_weight
        labels[row_idx, upper_idx] = upper_weight

    return labels


class PlanningContextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        input_bev_channels: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config: TrainingConfig = config

        self.num_status_tokens = 0

        if self.config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using velocity encoder.")

        if self.config.use_acceleration:
            self.num_status_tokens += 1
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using acceleration encoder.")

        if self.config.use_discrete_command:
            self.num_status_tokens += 1
            self.command_encoder = nn.Sequential(
                nn.Linear(
                    self.config.discrete_command_dim,
                    self.config.transfuser_token_dim,
                ),
            )
            logger.info("Using discrete command encoder.")

        if self.config.use_tp:
            self.num_status_tokens += 1
            self.tp_encoder = nn.Linear(2, config.transfuser_token_dim)
            logger.info("Using target point encoder.")

        if self.config.use_previous_tp:
            self.num_status_tokens += 1
            logger.info("Using previous target point encoder.")

        if self.config.use_next_tp:
            self.num_status_tokens += 1
            logger.info("Using next target point encoder.")

        if self.config.use_past_positions:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past positions encoder.")
            self.past_positions_encoder = nn.Linear(2, config.transfuser_token_dim)

        if self.config.use_past_speeds:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past speeds encoder.")
            self.past_speeds_encoder = nn.Linear(1, config.transfuser_token_dim)

        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            self.num_status_tokens += self.config.num_radar_queries
            self.radar_encoder = nn.Linear(
                self.config.radar_token_dim,
                config.transfuser_token_dim,
            )
            logger.info(
                f"Using radar encoder with {self.config.num_radar_queries} tokens.",
            )

        self.cosine_pos_embeding = PositionEmbeddingSine(
            config,
            self.config.transfuser_token_dim // 2,
            normalize=True,
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_status_tokens, self.config.transfuser_token_dim),
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels,
            self.config.transfuser_token_dim,
            kernel_size=1,
        )
        self.reset_parameters()

        self.target_points_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

    def reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "B C H W"],
        radar_logits: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
    ) -> jt.Float[torch.Tensor, "B N D"]:
        """
        Args:
            bev_features: Raw BEV features.
            radar_logits: Radar logits.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            context_tokens: Output tokens for planning transformer decoder.
        """
        # Load data
        if self.config.use_velocity:
            velocity = (
                data["speed"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
        if self.config.use_discrete_command:
            command = data["command"].to(
                self.device,
                dtype=self.config.torch_float_type,
            )

        status_tokens = []

        # Encode speed
        if self.config.use_velocity:
            velocity_token = self.velocity_encoder(
                velocity / self.config.max_speed,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode acceleration
        if self.config.use_acceleration:
            acceleration = (
                data["acceleration"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
            acceleration_token = self.acceleration_encoder(
                acceleration / self.config.max_acceleration,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(acceleration_token)

        # Encode command
        if self.config.use_discrete_command:
            command_token = self.command_encoder(command).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(command_token)

        # Encode target point
        if self.config.use_tp:
            target_point = data["target_point"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            target_point = target_point / self.target_points_normalization_constants
            tp_token = self.tp_encoder(target_point).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(tp_token)

        if self.config.use_previous_tp:
            previous_tp = data["target_point_previous"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            previous_tp = previous_tp / self.target_points_normalization_constants
            previous_tp_token = self.tp_encoder(previous_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(previous_tp_token)

        if self.config.use_next_tp:
            next_tp = data["target_point_next"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            next_tp = next_tp / self.target_points_normalization_constants
            next_tp_token = self.tp_encoder(next_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(next_tp_token)

        # Encode radar
        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            radar_token = self.radar_encoder(radar_logits).reshape(
                -1,
                self.config.num_radar_queries,
                self.config.transfuser_token_dim,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_pos_embed = fn.gen_sineembed_for_position(
                fn.unit_normalize_bev_points(
                    radar_predictions[..., [RadarLabels.X, RadarLabels.Y]].reshape(
                        -1,
                        2,
                    ),
                    self.config,
                ),
                self.config.transfuser_token_dim,
            ).reshape(
                radar_token.shape,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_token = (
                radar_token + radar_pos_embed
            )  # (bs, num_radar_queries, transfuser_token_dim)
            status_tokens.append(radar_token)

        # Concatenate status tokens if any
        has_statuses = False
        if len(status_tokens) > 0:
            status_tokens = torch.cat(
                status_tokens,
                dim=1,
            )  # (bs, num_status_tokens, transfuser_token_dim)
            has_statuses = True

        # Process BEV features
        context_tokens = self.dimension_adapter(
            bev_features,
        )  # (bs, transfuser_token_dim, height, width)

        # Concatenate and add positional embeddings
        if has_statuses:
            context_tokens = context_tokens + self.cosine_pos_embeding(
                context_tokens,
            )  # (bs, transfuser_token_dim, height, width)
            context_tokens = torch.flatten(
                context_tokens,
                start_dim=2,
            )  # (bs, transfuser_token_dim, height * width)
            context_tokens = torch.permute(
                context_tokens,
                (0, 2, 1),
            )  # (bs, height * width, transfuser_token_dim)

            status_tokens = (
                status_tokens + self.status_pos_embedding
            )  # (bs, num_status_tokens, transfuser_token_dim)
            context_tokens = torch.cat(
                [context_tokens, status_tokens],
                dim=1,
            )  # (bs, height * width + num_status_tokens, transfuser_token_dim)

        return context_tokens


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.config = config
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(self.config.torch_float_type).contiguous()
