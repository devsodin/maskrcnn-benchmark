# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from skimage.color import rgb2hsv, hsv2rgb, rgb2ycbcr, ycbcr2rgb
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor


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

        self.ohem = cfg.MODEL.ROI_HEADS.OHEM

        self.custom_post_process = cfg.MODEL.ROI_BOX_HEAD.CUSTOM_POSTPROCESS
        self.previous_frames_result = []
        self.num_previous_frames = cfg.MODEL.ROI_BOX_HEAD.PREVIOUS_FRAMES
        self.previous_frame = None
        self.frames_cached = []
        self.count = 0

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

                if self.ohem:
                    x = self.feature_extractor(features, proposals)
                    class_logits, box_regression = self.predictor(x)
                    proposals = self.loss_evaluator.mining(
                        [class_logits], [box_regression]
                    )

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)

            if self.custom_post_process:
                # post_result = remove_saturated_hsv(result, orig_images)
                post_result = remove_saturated_ycbcr(result, orig_images)

                if self.previous_frame is None:
                    self.previous_frame = result
                else:
                    # post_result = self.last_frame_spatial_coherence(post_result)
                    self.previous_frame = result

                result = post_result

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def last_frame_spatial_coherence(self, result):
        from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

        ious = []
        displacement_percent = 0.05
        iou_threshold = 0.6

        if len(result) != len(self.previous_frame):
            assert "different sizes"

        # calculate ious between new and old frame
        for res, prev in zip(result, self.previous_frame):
            ious.append(boxlist_iou(res, prev))

        for idx, iou in zip(range(len(ious)), ious):
            if iou.nelement():
                if (iou > iou_threshold).item():
                    continue
                else:
                    # on low intersection value, check if box has changed size a lot
                    keep_idx = []

                    for res_box, old_box, j in zip(result[idx].bbox, self.previous_frame[idx].bbox,
                                                   range(self.previous_frame[idx].bbox.shape[1])):

                        im_size = result[idx].size
                        max_tol = (torch.Tensor([im_size[0], im_size[1]]) * displacement_percent).to(res_box.device)

                        # print(res_box, old_box, torch.dist(res_box, old_box).item())
                        if (res_box - old_box)[:2].norm() < max_tol.norm():
                            keep_idx.append(j)
                        else:
                            print("reeeemove")

                    result[idx].bbox = result[idx].bbox[keep_idx,:]
                    for field in result[idx].fields():
                        result[idx].extra_fields[field] = result[idx].get_field(field)[keep_idx]

            # bbox with no intersection
            else:
                # save bbox with highest score, asuming the other is FP
                for res_box, old_box, j in zip(result[idx].bbox, self.previous_frame[idx].bbox,
                                               range(self.previous_frame[idx].bbox.shape[1])):

                    win = (res_box.get_field("scores") > old_box.get_field("scores")).tolist()

                    win = [i for i, pos in zip(range(len(win)), win) if win > 0]
                    result[idx].bbox = result[idx].bbox[win]
                    for field in result[idx].fields():
                        result[idx].extra_fields[field] = result[idx].get_field(field)[win]

        return result

    """
    TODO rehacer esto con calma, que me esta costando la vida
    """

    def remove_unexpected_boxes(self, result):
        """
        Method to remove boxes detected on last frame that not appear on the cached frames.
        :return:
        """
        from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

        weights = torch.arange(1 / self.num_previous_frames, 1.,
                               step=1 / self.num_previous_frames)

        votes = torch.zeros([len(result), len(self.previous_frames_result)])
        for boxlist, i in zip(result, range(len(result))):
            ious = None
            for old_boxlist_list, weight, j in zip(self.previous_frames_result, weights,
                                                   range(len(self.previous_frames_result))):
                for old_boxlist in old_boxlist_list:
                    iou = boxlist_iou(boxlist, old_boxlist)
                    if ious is None:
                        ious = iou
                    else:
                        if iou.nelement():
                            ious += iou

                ious *= weight

                if ious.nelement():
                    for i, iou in zip(range(ious.shape[0]), ious):
                        if iou > 1e-4:
                            if len(boxlist) > 0:
                                votes[i, j] += 1
                        else:
                            if len(boxlist) > 0:
                                votes[i, j] -= 1
                else:
                    if len(boxlist) > 0:
                        votes[i, j] -= 1

        # votes
        votes = votes.sum(dim=1)

        n_result = []
        for vote, result in zip(votes, result):
            if vote > 0.5 * self.num_previous_frames:
                n_result.append(result)

        return n_result

        # match_value = 0 if self.previous_frames_result else None
        #
        # for old_boxes, decay in zip(self.previous_frames_result[::-1],
        #                             torch.arange(start=self.num_previous_frames, end=0, step=-1)):
        #     result[0].bbox = result[0].bbox.cuda()
        #     iou = boxlist_iou(result[0], old_boxes[0])
        #
        #     if iou.nelement() > 0 and iou.sum().item() > 1e-4:
        #         match_value += 1
        #
        # # TODO refinar, ahora es como siempre. si una cuenta todas ok
        # print(match_value)
        # if match_value is not None:
        #     if match_value > self.num_previous_frames / 2:
        #         # hay coincidencia con frames anteriores pero no hay ahora
        #         if len(result[0]) == 0:
        #             return self.previous_frames_result[-1]
        #         # hay coincidencia con frames anteriores y hay output ahora
        #         else:
        #             return result
        #     else:
        #         # no hay coincidencia con frames anteriores y hay respuesta
        #         if len(result[0]) > 0:
        #             print("unexpected box")
        #             t = torch.empty(0, 4)
        #             new_box = BoxList(t.cuda(), result[0].bbox.size)
        #             return [new_box]
        #         else:
        #             return result
        # else:
        #     return result

def remove_saturated_ycbcr(boxes, images_orig):

    for boxlist, original, idx in zip(boxes, images_orig, range(len(boxes))):
        # if boxlist has bboxes check
        if len(boxlist) > 0:
            original = rgb2ycbcr(np.array(original))
            save_idx = []
            for box, i in zip(boxlist.bbox, range(len(boxlist))):
                x, y, x_, y_ = box[0], box[1], box[2], box[3]

                region = np.array(original)[x.int():x_.int(), y.int():y_.int()]
                if region.shape[0] <= 0 or region.shape[1] <= 1:
                    continue

                avg_y = region[:, :, 0].mean()
                std_y = region[:,:, 0].std()
                max_y = region[:,:, 0].max()

                if avg_y + std_y > 200 and max_y > 240:
                    print("removing sat zone - max {}, avg {}".format(max_y, avg_y))
                else:
                    save_idx.append(i)

            boxes[idx].bbox = boxlist.bbox[save_idx]

            for field in boxlist.fields():
                boxes[idx].extra_fields[field] = boxlist.get_field(field)[save_idx]
        else:
            continue

    return boxes


def remove_saturated_hsv(boxes, images_orig):

    for boxlist, original, idx in zip(boxes, images_orig, range(len(boxes))):
        # if boxlist has bboxes check
        if len(boxlist) > 0:
            original = rgb2hsv(np.array(original))
            save_idx = []
            for box, i in zip(boxlist.bbox, range(len(boxlist))):
                x, y, x_, y_ = box[0], box[1], box[2], box[3]

                region = np.array(original)[x.int():x_.int(), y.int():y_.int()]
                if region.shape[0] <= 0 or region.shape[1] <= 1:
                    continue

                avg_s = region[:, :, 1].mean() / 255
                std = region.std()
                avg_v = region[:, :, 2].mean()

                # TODO learn threshold
                if avg_v >= 0.9 and avg_s <= 0.1:
                    print("removing sat bbox", std)
                    # _plot_region(region)
                elif avg_v < 0.1:
                    print("removing very obscured bbox", std)
                    # _plot_region(region)
                else:
                    save_idx.append(i)

            boxes[idx].bbox = boxlist.bbox[save_idx]

            for field in boxlist.fields():
                boxes[idx].extra_fields[field] = boxlist.get_field(field)[save_idx]
        else:
            continue

    return boxes

def _plot_region(region):
    from matplotlib import pyplot as plt

    region = hsv2rgb(region)
    plt.imshow(region)
    plt.show()


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
