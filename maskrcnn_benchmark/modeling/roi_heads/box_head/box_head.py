# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.previous_boxes = []
        self.max_cache_boxes = 5

    def forward(self, features, proposals, orig_images, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)

            if len(self.previous_boxes) < self.max_cache_boxes:
                self.previous_boxes.append(result)
            else:
                self.previous_boxes.append(result)
                self.previous_boxes = self.previous_boxes[1:]

                # new_boxes = self.remove_unexpected_boxes()

            result = self.remove_saturated(result, orig_images)

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def remove_saturated(self, boxes, images_orig):

        import numpy as np
        from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

        n_boxes = []
        for boxlist, original in zip(boxes, images_orig):
            stay = []
            for box, score in zip(boxlist.bbox, boxlist.get_field("scores")):
                x = box[0]
                y = box[1]
                x_ = box[2]
                y_ = box[3]

                region = np.array(original)[x.int():x_.int(), y.int():y_.int()]
                if region.shape[0] <= 0 or region.shape[1] <= 1:
                    continue

                avg = region.mean()
                std = region.std()

                # TODO learn threshold
                if avg <= 240:
                    new_box = BoxList([[x, y, x_, y_]], boxlist.size, mode="xyxy")
                    new_box.add_field("scores", torch.unsqueeze(score,-1).cpu())
                    stay.append(new_box)
                else:
                    print("removing_sat bbox")

                # else:
                #     plt.imshow(original)
                #     plt.show()
                #     plt.imshow(region)
                #     plt.show()
            if len(stay) > 0:
                n_boxes.append(cat_boxlist(stay))

        boxes = n_boxes
        return boxes




def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
