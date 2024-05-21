import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ...utils.heapmap_utils import generate_heatmap
from ...utils.mask_utils import generate_mask_cond


class MLGTActor(BaseActor):
    """
    Actor for training MLGT models.
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        # out_dict = self.forward_pass(data)
        pred1 = self.forward_pass(data)

        # Compute losses
        # loss, status = self.compute_losses(out_dict, data)
        loss, status = self.compute_losses(pred1, data)
        return loss, status

    def forward_pass(self, data):
        # Currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                        data['template_anno'][0])

        if len(template_list) == 1:
            template_list = template_list[0]

        # out_dict = self.net(template=template_list, search=search_img, template_mask=box_mask_z)
        pred1 = self.net(template=template_list, search=search_img, template_mask=box_mask_z)
        # return out_dict
        return pred1

    def compute_losses(self, pred1, gt_dict, return_status=True, entropy=False):
        # def compute_losses(self, pred_dict, gt_dict, return_status=True, entropy=False):
        # # GT gaussian map
        # gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        # gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
        #                                     self.cfg.MODEL.BACKBONE.STRIDE)
        # gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        #
        # # Get boxes
        # pred_boxes = pred_dict['pred_boxes']
        # if torch.isnan(pred_boxes).any():
        #     raise ValueError('ERROR: network outputs is NAN! stop training')

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes1 = pred1['pred_boxes']
        # pred_boxes2 = pred2['pred_boxes']
        # pred_boxes3 = pred3['pred_boxes']
        # pred_boxes4 = pred4['pred_boxes']
        num_queries = pred_boxes1.size(1)
        # # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        # # (B,4) --> (B,1,4) --> (B,N,4)
        # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        #                                                                                                    max=1.0)
        pred_boxes_vec1 = box_cxcywh_to_xyxy(pred_boxes1).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # pred_boxes_vec2 = box_cxcywh_to_xyxy(pred_boxes2).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # pred_boxes_vec3 = box_cxcywh_to_xyxy(pred_boxes3).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # pred_boxes_vec4 = box_cxcywh_to_xyxy(pred_boxes4).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4
        # # Compute GIoU and IoU
        # try:
        #     giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # except:
        #     giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # # Compute L1 loss
        # l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # # Compute location loss
        # if 'score_map' in pred_dict:
        #     location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss = torch.tensor(0.0, device=l1_loss.device)

        # compute giou and iou
        try:
            giou_loss1, iou1 = self.objective['giou'](pred_boxes_vec1, gt_boxes_vec)  # (BN,4) (BN,4)
            # giou_loss2, iou2 = self.objective['giou'](pred_boxes_vec2, gt_boxes_vec)  # (BN,4) (BN,4)
            # giou_loss3, iou3 = self.objective['giou'](pred_boxes_vec3, gt_boxes_vec)  # (BN,4) (BN,4)
            # giou_loss4, iou4 = self.objective['giou'](pred_boxes_vec4, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss1, iou1 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # giou_loss2, iou2 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # giou_loss3, iou3 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # giou_loss4, iou4 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss1 = self.objective['l1'](pred_boxes_vec1, gt_boxes_vec)  # (BN,4) (BN,4)
        # l1_loss2 = self.objective['l1'](pred_boxes_vec2, gt_boxes_vec)  # (BN,4) (BN,4)
        # l1_loss3 = self.objective['l1'](pred_boxes_vec3, gt_boxes_vec)  # (BN,4) (BN,4)
        # l1_loss4 = self.objective['l1'](pred_boxes_vec4, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred1:
            location_loss1 = self.objective['focal'](pred1['score_map'], gt_gaussian_maps)
        else:
            location_loss1 = torch.tensor(0.0, device=l1_loss1.device)
        # if 'score_map' in pred2:
        #     location_loss2 = self.objective['focal'](pred2['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss2 = torch.tensor(0.0, device=l1_loss2.device)
        # if 'score_map' in pred3:
        #     location_loss3 = self.objective['focal'](pred3['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss3 = torch.tensor(0.0, device=l1_loss3.device)
        # if 'score_map' in pred4:
        #     location_loss4 = self.objective['focal'](pred4['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss4 = torch.tensor(0.0, device=l1_loss4.device)

        # Weighted sum
        loss = self.loss_weight['giou'] * giou_loss1 + self.loss_weight['l1'] * l1_loss1 + self.loss_weight[
            'focal'] * location_loss1
        # loss = self.loss_weight['giou'] * (0.2*giou_loss2 + 0.2*giou_loss3 + 0.6*giou_loss4)+ self.loss_weight['l1'] * (0.0*l1_loss1 + 0.2*l1_loss2+0.2*l1_loss3+0.6*l1_loss4) + self.loss_weight['focal'] * (0.0*location_loss1 + 0.2*location_loss2+0.4*location_loss3+0.4*location_loss4)

        if entropy and pred1['decisions'] != []:
            epsilon = 1e-5
            prob1 = pred1['decisions']
            prob2 = 1 - pred1['decisions']
            entropy_loss = (1 + prob1 * torch.log2(prob1 + epsilon) + prob2 * torch.log2(prob2 + epsilon)).mean()
            loss += entropy_loss

        if return_status:
            # Status for log
            mean_iou = iou1.detach().mean()
            # mean_iou = (0.0*iou1 + 0.2 * iou2 + 0.2 * iou3 + 0.6 * iou4).detach().mean()
            if entropy and pred1['decisions'] != []:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss1.item(),
                          'Ls/l1': l1_loss1.item(),
                          'Ls/loc': location_loss1.item(),
                          'Ls/entropy': entropy_loss.item(),
                          'IoU': mean_iou.item()}
            else:
                # status = {'Ls/total': loss.item(),
                #           'Ls/giou': (0.0*giou_loss1 + 0.2*giou_loss2 + 0.2*giou_loss3 + 0.6*giou_loss4).item(),
                #           'Ls/l1': (0.0*l1_loss1 + 0.2*l1_loss2+0.2*l1_loss3+0.6*l1_loss4).item(),
                #           'Ls/loc': (0.0*location_loss1 + 0.2*location_loss2+0.4*location_loss3+0.4*location_loss4).item(),
                #           'IoU': mean_iou.item()}
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss1.item(),
                          'Ls/l1': l1_loss1.item(),
                          'Ls/loc': location_loss1.item(),
                          'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss
