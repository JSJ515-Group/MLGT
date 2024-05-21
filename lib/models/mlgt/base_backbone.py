import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.mlgt.utils import combine_tokens, recover_tokens

from timm.models.vision_transformer import Block as Block1

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # For original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_x = None
        self.pos_embed_z = None

    def finetune_track(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE

        # Resize patch embedding
        if new_patch_size != self.patch_size:
            print('inconsistent patch size with the pretrained weights, interpolate the weight')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        self.ID = [1, 5, 8, 11]
        self.scale = [1.0, 1.0, 1.0, 1.0]
        norm_layer = nn.LayerNorm
        self.norm1 = nn.ModuleList([norm_layer(self.embed_dim) for _ in range(len(self.ID))])
        self.decoder = nn.ModuleList([
            MAE_Decoder(self.embed_dim, 768, 768, s, self.patch_embed.num_patches, 1, 8, 4.0, True, norm_layer) for
            s in self.scale])  # 改768

        # For patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # For search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # For template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)

    def forward_features(self, z, x, template_mask, search_feat_len, threshold, tgt_type):
        x = self.patch_embed(x)
        z = self.patch_embed(z)

        x += self.pos_embed_x
        z += self.pos_embed_z

        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        decisions = []
        latent = []
        for i, blk in enumerate(self.blocks):
            x, decision = blk(x, template_mask, search_feat_len, threshold=threshold, tgt_type=tgt_type)
            if decision is not None and self.training:
                map_size = decision.shape[1]
                decision = decision[:, :, -1].sum(dim=-1, keepdim=True) / map_size
                decisions.append(decision)

            if i in self.ID:
                latent.append(self.norm1[self.ID.index(i)](x))

        x = recover_tokens(x, mode=self.cat_mode)

        if self.training:
            decisions = torch.cat(decisions, dim=-1)  # .mean(dim=-1, keepdim=True)
        return self.norm(x), decisions, latent

    def forward(self, z, x, template_mask, search_feat_len, threshold, tgt_type, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.

        Args:
            z (torch.Tensor): Template feature, [B, C, H_z, W_z].
            x (torch.Tensor): Search region feature, [B, C, H_x, W_x].

        Returns:
            x (torch.Tensor): Merged template and search region feature, [B, L_z+L_x, C].
            attn : None.
        """

        x, decisions, latent = self.forward_features(z, x, template_mask, search_feat_len, threshold, tgt_type)
        pred = [self.decoder[i](latent[i]) for i in range(len(latent))]
        return x, decisions, pred


class MAE_Decoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=256, out_dim=27, scale=1., num_patches=196, depth=1, num_heads=8, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.embed = nn.Linear(inp_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block1(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # pred head
        hidden = embed_dim
        if scale == 3.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
                      LayerNorm(embed_dim//2),
                      nn.GELU(),
                      nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)]
            hidden = embed_dim//4
        elif scale == 2.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)]
            hidden = embed_dim//2
        elif scale == 1.0:
            layers = []
        elif scale == 0.5:
            layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
        layers.append(nn.Conv2d(hidden, out_dim, kernel_size=1))
        self.pred = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.embed(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, L, d]

        # x = x[:, 64:]
        # # predictor projection
        # H = W = 16
        # x = x.transpose(1, 2).reshape(x.size(0), -1, H, W)
        # x = self.pred(x)
        # x = x.flatten(2, 3).transpose(1, 2)

        return x

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim//2, dtype=np.float)
    omega /= embed_dim/2.
    omega = 1./10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
