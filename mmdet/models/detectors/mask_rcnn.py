from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch


@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_no_normalize,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        proposals = []
        for meta in img_metas:
            filename = "data/coco/proposals/" + meta["filename"].split("/", 2)[-1].split(".")[0] + ".pth"
            proposal = torch.load(filename, map_location=img.device)

            h, w, _ = meta["img_shape"]
            proposal[:, 0] *= w
            proposal[:, 1] *= h
            proposal[:, 2] *= w
            proposal[:, 3] *= h

            proposals.append(proposal)

        # # RPN forward and loss
        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                       self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         gt_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg)
        #     losses.update(rpn_losses)
        # else:
        #     proposal_list = proposals

        proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img, img_no_normalize, img_metas, proposal_list, proposals,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

