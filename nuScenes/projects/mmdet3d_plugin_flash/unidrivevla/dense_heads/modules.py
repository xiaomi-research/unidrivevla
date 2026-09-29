import torch
from torch import nn
from mmdet.models.builder import build_head
from einops import rearrange
from torch.cuda.amp import autocast

class OccLatentDecoder(nn.Module):
    def __init__(self, qwen_dim=2048, occworld_vae_config=None, pretrained_vae_path=None):
        super().__init__()
        if occworld_vae_config is None:
            raise ValueError("Please provide the OccWorld VAE config dict.")
        self.vae = build_head(occworld_vae_config)

        if pretrained_vae_path:
            checkpoint = torch.load(pretrained_vae_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k[4:] if k.startswith('vae.') else k: v for k, v in state_dict.items()}
            if not new_state_dict:
                new_state_dict = state_dict
            self.vae.load_state_dict(new_state_dict, strict=False)

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.latent_dim = occworld_vae_config['decoder_cfg']['z_channels']
        self.feat_proj = nn.Linear(qwen_dim, self.latent_dim)

    def forward(self, occ_tokens):
        B = occ_tokens.shape[0]
        latent_tokens = self.feat_proj(occ_tokens)
        latent_grid = rearrange(latent_tokens, 'b (h w) c -> b c 1 h w', h=25, w=25)
        shapes = [(200, 200), (100, 100), (50, 50)]
        target_shape = (B, 1, 200, 200, 16)
        logits = self.vae.forward_decoder(z=latent_grid, shapes=shapes, input_shape=target_shape)
        occ_logits = logits.squeeze(1).permute(0, 4, 3, 1, 2)
        return occ_logits

class DenseDepthNet(nn.Module):
    def __init__(self, embed_dims=256, in_channels=None, num_depth_layers=1, equal_focal=100, max_depth=60, loss_weight=1.0):
        super(DenseDepthNet, self).__init__()
        self.embed_dims = embed_dims
        self.in_channels = in_channels if in_channels is not None else embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, embed_dims, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, 1, kernel_size=1),
        )

    def forward(self, x, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
            focal = focal.view(-1, 1, 1, 1)

        depth = self.depth_layer(x)
        depth = depth.exp()
        depth = depth * focal / self.equal_focal

        if gt_depths is not None:
            loss = self.loss(depth, gt_depths)
            return loss
        return depth

    def loss(self, depth_preds, gt_depths):
        # depth_preds: (B*N, 1, H, W)
        # gt_depths: (B*N, H, W) or (B*N, 1, H, W)
        
        if gt_depths.dim() == 3:
            gt_depths = gt_depths.unsqueeze(1)
            
        pred = depth_preds.permute(0, 2, 3, 1).contiguous().reshape(-1)
        gt = gt_depths.permute(0, 2, 3, 1).contiguous().reshape(-1)
        
        fg_mask = torch.logical_and(
            gt > 0.0, torch.logical_not(torch.isnan(pred))
        )
        gt = gt[fg_mask]
        pred = pred[fg_mask]
        pred = torch.clip(pred, 0.0, self.max_depth)
        
        with autocast(enabled=False):
            error = torch.abs(pred - gt).sum()
            _loss = (
                error
                / max(1.0, len(gt))
                * self.loss_weight
            )
        return _loss
