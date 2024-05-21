"""
Basic MLGT model.
"""

import os

import torch
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.mlgt.vit import vit_base_patch16_224_base, vit_base_patch16_224_large
from lib.utils.box_ops import box_xyxy_to_cxcywh

import importlib
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

class MLGT(nn.Module):
    """
    This is the base class for MLGT.
    """

    def __init__(self, transformer, box_head, head_type='CORNER', tgt_type='allmax'):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.conv1 = nn.Conv2d(3072,768,1)
        self.head_type = head_type
        if head_type == 'CORNER' or head_type == 'CENTER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.tgt_type = tgt_type

    def forward(self, template: torch.Tensor, search: torch.Tensor, template_mask=None, threshold=0.):
        x, decisions, pre = self.backbone(z=template, x=search, template_mask=template_mask, search_feat_len=self.feat_len_s,
                                     threshold=threshold, tgt_type=self.tgt_type)

        # # Forward head
        # feat_last = x
        # if isinstance(x, list):
        #     feat_last = x[-1]

        out1= self.forward_head(pre, None)
        # out1 = self.forward_head(x, None)
        out1['decisions'] = decisions

        return out1
        # out = self.forward_head(feat_last, None)
        # out['decisions'] = decisions
        # return out

    def forward_head(self, feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        feat_1 = feature[0]
        feat_2 = feature[1]
        feat_3 = feature[2]
        feat_4 = feature[3]
        # cat_feature = feature
        cat_feature = feat_1+feat_2+feat_3+feat_4
        # cat_feature = self.conv1(cat_feature)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # opt1 = (feat_1.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # opt2 = (feat_2.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # opt3 = (feat_3.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # opt4 = (feat_4.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()

        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, 16, 16)

        # bs, Nq, C, HW = opt2.size()
        # opt_feat2 = opt2.view(-1, C, 16, 16)
        #
        # bs, Nq, C, HW = opt3.size()
        # opt_feat3 = opt3.view(-1, C, 16, 16)
        #
        # bs, Nq, C, HW = opt4.size()
        # opt_feat4 = opt4.view(-1, C, 16, 16)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box1, score_map1 = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box1)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out1 = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map1,
                    }
            # pred_box2, score_map2 = self.box_head(opt_feat2, True)
            # outputs_coord = box_xyxy_to_cxcywh(pred_box2)
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out2 = {'pred_boxes': outputs_coord_new,
            #         'score_map': score_map2,
            #         }
            # pred_box3, score_map3 = self.box_head(opt_feat3, True)
            # outputs_coord = box_xyxy_to_cxcywh(pred_box3)
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out3 = {'pred_boxes': outputs_coord_new,
            #         'score_map': score_map3,
            #         }
            # pred_box4, score_map4 = self.box_head(opt_feat4, True)
            # outputs_coord = box_xyxy_to_cxcywh(pred_box4)
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out4 = {'pred_boxes': outputs_coord_new,
            #         'score_map': score_map1,
            #         }
            return out1

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr1, bbox1, size_map1, offset_map1 = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord1 = bbox1
            outputs_coord_new1 = outputs_coord1.view(bs, Nq, 4)
            out1 = {'pred_boxes': outputs_coord_new1,
                    'score_map': score_map_ctr1,
                    'size_map': size_map1,
                    'offset_map': offset_map1}
            # score_map_ctr2, bbox2, size_map2, offset_map2 = self.box_head(opt_feat2, gt_score_map)
            # # outputs_coord = box_xyxy_to_cxcywh(bbox)
            # outputs_coord2 = bbox2
            # outputs_coord_new2 = outputs_coord2.view(bs, Nq, 4)
            # out2 = {'pred_boxes': outputs_coord_new2,
            #         'score_map': score_map_ctr2,
            #         'size_map': size_map2,
            #         'offset_map': offset_map2}
            # score_map_ctr3, bbox3, size_map3, offset_map3 = self.box_head(opt_feat3, gt_score_map)
            # # outputs_coord = box_xyxy_to_cxcywh(bbox)
            # outputs_coord3 = bbox3
            # outputs_coord_new3 = outputs_coord3.view(bs, Nq, 4)
            # out3 = {'pred_boxes': outputs_coord_new3,
            #         'score_map': score_map_ctr3,
            #         'size_map': size_map3,
            #         'offset_map': offset_map3}
            # score_map_ctr4, bbox4, size_map4, offset_map4 = self.box_head(opt_feat4, gt_score_map)
            # # outputs_coord = box_xyxy_to_cxcywh(bbox)
            # outputs_coord4 = bbox4
            # outputs_coord_new4 = outputs_coord4.view(bs, Nq, 4)
            # out4 = {'pred_boxes': outputs_coord_new4,
            #         'score_map': score_map_ctr4,
            #         'size_map': size_map4,
            #         'offset_map': offset_map4}
            return out1
        else:
            raise NotImplementedError


def build_mlgt(cfg, training=True):
    # current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = 'D:/2023\MLGT\lib\pretrained'
    if cfg.MODEL.PRETRAIN_FILE and ('MLGT' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_patch16_224-b5f2ef4d.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        elif cfg.MODEL.PRETRAIN_FILE == 'deit_base_distilled_patch16_224-df68dfff.pth':
            backbone = vit_base_patch16_224_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, distilled=True)
            hidden_dim = backbone.embed_dim
            patch_start_index = 2
        else:
            raise NotImplementedError
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large':
        if cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_large.pth':
            backbone = vit_base_patch16_224_large(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = MLGT(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD.TYPE,
        tgt_type=cfg.MODEL.TGT_TYPE
    )

    if 'MLGT' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print('load pretrained model from ' + cfg.MODEL.PRETRAIN_FILE)
    return model

# def transform(img, img_size):
#     img = transforms.Resize(img_size)(img)
#     img = transforms.ToTensor()(img)
#     return img
#
# def visualize_predict(model, img, img_size, patch_size, device):
#     img_pre = transform(img, img_size)
#     attention = visualize_attention(model, img_pre, patch_size, device)
#     plot_attention(img, attention)
#
# def visualize_attention(model, img, patch_size, device):
#     # make the image divisible by the patch size
#     w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
#     img = img[:, :w, :h].unsqueeze(0)
#
#     w_featmap = img.shape[-2] // patch_size
#     h_featmap = img.shape[-1] // patch_size
#
#     attentions = model.get_last_selfattention(img.to(device))
#
#     nh = attentions.shape[1]  # number of head
#
#     # keep only the output patch attention
#     attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
#
#     attentions = attentions.reshape(nh, w_featmap, h_featmap)
#     attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
#
#     return attentions
#
# def plot_attention(img, attention):
#     n_heads = attention.shape[0]
#
#     plt.figure(figsize=(10, 10))
#     text = ["Original Image", "Head Mean"]
#     for i, fig in enumerate([img, np.mean(attention, 0)]):
#         plt.subplot(1, 2, i+1)
#         plt.imshow(fig, cmap='inferno')
#         plt.title(text[i])
#     plt.show()
#
#     plt.figure(figsize=(10, 10))
#     for i in range(n_heads):
#         plt.subplot(n_heads//3, 3, i+1)
#         plt.imshow(attention[i], cmap='inferno')
#         plt.title(f"Head n: {i+1}")
#     plt.tight_layout()
#     plt.show()
# from lib.config.mlgt import config as config_module
# # config_module = importlib.import_module('lib.config.%s.config' % settings.script_name)
# cfg = config_module.cfg
# model = build_mlgt(cfg)
# path = 'D:/2023\MLGT\lib/train\output1\checkpoints'
# checkpoinct = torch.load(path, map_location='cpu')
# model.load_state_dict(checkpoinct)
# path = './bike5.jpg'
# img = Image.open(path)
# factor_reduce = 2
# img_size = tuple(np.array(img.size[::-1]) // factor_reduce)
# visualize_predict(model, img, img_size, 16, device)