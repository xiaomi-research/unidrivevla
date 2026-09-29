from typing import Any

import jaxtyping as jt
import timm
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

from lead.tfv6 import transfuser_utils as fn
from lead.training.config_training import TrainingConfig


class TransfuserBackbone(nn.Module):
    """TransFuser backbone network for multi-modal sensor fusion.

    Implements the TransFuser architecture that fuses RGB image and LiDAR features
    using transformer-based attention mechanisms across multiple resolution levels.
    """

    @beartype
    def __init__(self, device: torch.device, config: TrainingConfig) -> None:
        """Initialize the TransFuser backbone with dual encoder branches and fusion modules.

        Args:
            device: Torch device for model placement (CPU or CUDA).
            config: Training configuration containing architecture parameters.
        """
        super().__init__()
        self.device = device
        self.config = config

        # Image branch
        self.image_encoder = timm.create_model(
            config.image_architecture,
            pretrained=True,
            features_only=True,
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors),
        )
        image_start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            image_start_index += 1
        self.num_image_features = self.image_encoder.feature_info.info[
            image_start_index + 3
        ]["num_chs"]

        # LiDAR branch
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=2 if config.LTF else 1,
            features_only=True,
        )
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1
        self.num_lidar_features = self.lidar_encoder.feature_info.info[
            lidar_start_index + 3
        ]["num_chs"]
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    kernel_size=1,
                )
                for i in range(0, 4)
            ],
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    kernel_size=1,
                )
                for i in range(0, 4)
            ],
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors),
        )

        # Fusion transformers
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    config=config,
                )
                for i in range(0, 4)
            ],
        )

        # Post-fusion convs
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[image_start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size=(
                self.config.lidar_height_pixel // self.config.bev_down_sample_factor,
                self.config.lidar_width_pixel // self.config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.up_conv4 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.c5_conv = nn.Conv2d(
            self.num_lidar_features,
            self.config.bev_features_chanels,
            (1, 1),
        )

    @jt.jaxtyped(typechecker=beartype)
    def top_down(
        self,
        x: jt.Float[torch.Tensor, "B C H W"],
    ) -> jt.Float[torch.Tensor, "B C2 H2 W2"]:
        """Apply top-down feature pyramid processing to BEV features.

        Progressively upsamples and refines features through multiple resolution levels
        to create a higher-resolution bird's-eye-view representation.

        Args:
            x: Input BEV feature tensor from the LiDAR encoder.

        Returns:
            Upsampled and refined BEV feature tensor at target resolution.
        """
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
        p3 = F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)
        return p3

    def forward(
        self,
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the TransFuser backbone with data preprocessing.

        Extracts RGB and LiDAR inputs from the data dictionary, handles device placement
        and dtype conversion, and optionally generates positional encodings for LTF mode.

        Args:
            data: Dictionary containing sensor data with keys:
                - 'rgb': RGB image tensor
                - 'rasterized_lidar': LiDAR pseudo-image (if not using LTF mode)

        Returns:
            Tuple of (lidar_features, image_features):
                - lidar_features: BEV feature map for planning tasks
                - image_features: Image feature map for perception tasks
        """
        rgb = data["rgb"].to(
            self.device,
            dtype=self.config.torch_float_type,
            non_blocking=True,
        )
        if self.config.LTF:
            x = torch.linspace(0, 1, self.config.lidar_width_pixel)
            y = torch.linspace(0, 1, self.config.lidar_height_pixel)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

            lidar = torch.zeros(
                (
                    rgb.shape[0],
                    2,
                    self.config.lidar_height_pixel,
                    self.config.lidar_width_pixel,
                ),
                device=rgb.device,
            )
            lidar[:, 0] = y_grid.unsqueeze(0)  # Top down positional encoding
            lidar[:, 1] = x_grid.unsqueeze(0)  # Left right positional encoding
        else:
            lidar = data["rasterized_lidar"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
        return self._forward(rgb, lidar)

    @jt.jaxtyped(typechecker=beartype)
    def _forward(
        self,
        image: jt.Float[torch.Tensor, "B 3 img_h img_w"],
        lidar: jt.Float[torch.Tensor, "B 1 bev_h bev_w"]
        | jt.Float[torch.Tensor, "B 2 bev_h bev_w"]
        | None,
    ) -> tuple[
        jt.Float[torch.Tensor, "B D1 H1 W1"],
        jt.Float[torch.Tensor, "B D2 H2 W2"],
    ]:
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image: RGB image.
            lidar: Pseudo-image LiDAR.
        Returns:
            lidar_features: BEV feature map for planning.
            image_features: Image feature map for perception.
        """
        image_features = fn.normalize_imagenet(image)
        lidar_features = lidar

        if self.config.channel_last:
            image = image.to(memory_format=torch.channels_last)
            if lidar is not None:
                lidar = lidar.to(memory_format=torch.channels_last)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers,
                self.image_encoder.return_layers,
                image_features,
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )

        # Loop through the 4 blocks of the network.
        for i in range(4):
            # Branch-specific forward pass
            image_features = self.forward_layer_block(
                image_layers,
                self.image_encoder.return_layers,
                image_features,
            )
            lidar_features = self.forward_layer_block(
                lidar_layers,
                self.lidar_encoder.return_layers,
                lidar_features,
            )
            image_features, lidar_features = self.fuse_features(
                image_features,
                lidar_features,
                i,
            )

        return lidar_features, image_features

    @beartype
    def forward_layer_block(
        self,
        layers: Any,
        return_layers: dict[str, str],
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block.

        Args:
            layers: Iterator starting at the current layer block of the target network.
            return_layers: TIMM dictionary describing at which intermediate layers features are returned.
            features: Input features.

        Returns:
            Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    @jt.jaxtyped(typechecker=beartype)
    def fuse_features(
        self,
        image_features: jt.Float[torch.Tensor, "B C H W"],
        lidar_features: jt.Float[torch.Tensor, "B C2 H2 W2"],
        layer_idx: int,
    ) -> tuple[jt.Float[torch.Tensor, "B C H W"], jt.Float[torch.Tensor, "B C2 H2 W2"]]:
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        Args:
            image_features: Features from the image branch
            lidar_features: Features from the LiDAR branch
            layer_idx: Transformer layer index.
        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer,
            lidar_embd_layer,
        )

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer,
        )
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """GPT-style transformer module for cross-modal feature fusion.

    Implements a transformer that fuses image and LiDAR features using learned
    positional embeddings and multi-head self-attention across both modalities.
    """

    @beartype
    def __init__(self, n_embd: int, config: TrainingConfig) -> None:
        """Initialize the GPT fusion transformer.

        Args:
            n_embd: Embedding dimension (number of feature channels).
            config: Training configuration containing transformer hyperparameters
        """
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.config.img_vert_anchors * self.config.img_horz_anchors
                + self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            ),
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for layer in range(config.n_layer)
            ],
        )
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @beartype
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear and layer norm modules.

        Applies custom initialization strategies based on configuration parameters
        to improve training stability and convergence.

        Args:
            module: PyTorch module to initialize (Linear or LayerNorm).
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self,
        image_tensor: jt.Float[torch.Tensor, "B C img_h img_w"],
        lidar_tensor: jt.Float[torch.Tensor, "B C lidar_h lidar_w"],
    ) -> tuple[
        jt.Float[torch.Tensor, "B C img_h img_w"],
        jt.Float[torch.Tensor, "B C lidar_h lidar_w"],
    ]:
        """
        Fusion transformer forward pass.

        Args:
            image_tensor: image tensor
            lidar_tensor: lidar tensor
        Returns:
            image_tensor_out: fused image tensor
            lidar_tensor_out: fused lidar tensor
        """
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[:, self.config.img_vert_anchors * self.config.img_horz_anchors :, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class Block(nn.Module):
    """Transformer block with self-attention and feed-forward layers.

    Implements a standard transformer block with pre-normalization,
    multi-head self-attention, and an MLP with residual connections.
    """

    @beartype
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        """Initialize a transformer block.

        Args:
            n_embd: Embedding dimension (feature channels).
            n_head: Number of attention heads.
            block_exp: Expansion factor for MLP hidden dimension.
            attn_pdrop: Dropout probability for attention weights.
            resid_pdrop: Dropout probability for residual connections.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Apply transformer block with attention and feed-forward processing.

        Uses pre-normalization and residual connections for stable training.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd).

        Returns:
            Output tensor of same shape as input with attention and MLP applied.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention module.

    Implements scaled dot-product attention across multiple heads with
    learnable query, key, and value projections.
    """

    @beartype
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        """Initialize multi-head self-attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            attn_pdrop: Dropout probability for attention weights.
            resid_pdrop: Dropout probability for output projection.

        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Compute multi-head self-attention.

        Projects input to queries, keys, and values, applies scaled dot-product
        attention independently for each head, then concatenates and projects
        the results.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd).

        Returns:
            Attention output tensor of shape (batch, sequence_length, n_embd).
        """
        b, t, c = x.size()
        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = (
            self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        q = (
            self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
