from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        # 超参数，定义在/model/utils/config.py中

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES  # 这个是预定义好的，[8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS  # [0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]  #[16, ]

        # define the conv relu layers processing input feature map
        # 定义预处理层， 3x3卷积
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 定义前背景分类器，2(前景/背景)*9（anchor数量）
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 定义bbox回归器，4(四个偏移量)*9（anchor数量）
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        # 定义推荐层
        # 效果是处理掉回归之后不符合条件的anchor boxes
        # 如回归后边界超限的，宽⾼过⼩的，得分太低的（使⽤NMS⾮极⼤抑制）
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        # 定义⽬标推荐层，这个层和上⾯的推荐层的区别在于
        # 推荐层proposal是没有结合标注信息的，仅仅依赖于binary classificator算class score，把超限的、得分低的排除。
        # ⽽target layer是结合了ground truth信息，计算的不是⼆分类probability，⽽
        # 是计算与标注框的重叠⽐例IoU，排除IoU太低的框。
        # 定义在/lib/model/rpn/anchor_target_layer.py
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)
        # base_feat是ResNet101的layer3产出的特征图
        # shape=(bs,256*expansion,H/16,w/16) = (bs,1024,14,14)

        # return feature map after conv relu layer

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # 先进⾏3x3卷积操作，对特征进⾏预处理
        # 得到shape=(bs,512,14,14)

        # get rpn classification score
        # 得到RPN前背景分类score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        # 得到softmax分类结果
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        # 得到bbox偏移量结果
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # 将分类结果和bbox偏移量结果放⼊proposal层产⽣rois， 把偏移后越界的，score太低的，都丢掉
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            # 使⽤交叉熵计算前背景分类损失
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            # 使⽤L1计算bbox偏移量损失
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
