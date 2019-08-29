import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter
from matplotlib.patches import Polygon
from pycocotools import coco, cocoeval


def showAnns(coco_object, anns, color):
    """
    "Overrided" function from pycocotools to use custom colors.
    :param coco_object: object from pycocotools API (self references)
    :param anns: annotations to show
    :param color: color for the annotations
    :return:
    """

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly))
            else:
                # mask
                t = coco_object.imgs[ann['image_id']]
                if type(ann['segmentation']['counts']) == list:
                    rle = coco.maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                else:
                    rle = [ann['segmentation']]
                m = coco.maskUtils.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                for i in range(3):
                    img[:, :, i] = ColorConverter.to_rgba(color)[i]
                ax.imshow(np.dstack((img, m * 0.5)))
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def delete_seq_code(csv):
    csv['image'] = csv.image.map(lambda x: int(x.split("-")[1].split(".")[0]))
    return csv


def reformat_csvs(detections, split='test'):
    d = {}

    if split == 'val':
        seqs = range(16, 19)
    elif split == 'test':
        seqs = range(1, 19)
    else:
        raise ('invalid split - {}'.format(split))

    for seq in seqs:
        seq_rows = detections.loc[detections['image'].str.contains("{:03d}-".format(seq))].copy(deep=True)
        seq_rows = delete_seq_code(seq_rows)
        seq_rows.sort_values(by='image', inplace=True)
        d[seq] = seq_rows
        print(d[seq])

    return d


def calc_challenge_metrics(loc_df, detect_df, dt, out_folder, split):
    for image in dt.loadImgs(ids=dt.getImgIds()):
        detect_on_image(detect_df, dt, image)
        localize_on_image(dt, image, loc_df)

    detect_df.sort_values('image', inplace=True)
    detect_df.reset_index(inplace=True, drop=True)
    loc_df.sort_values('image', inplace=True)
    loc_df.reset_index(inplace=True, drop=True)

    detect_csvs = reformat_csvs(detect_df, split=split)
    loc_csvs = reformat_csvs(loc_df, split=split)

    if not os.path.exists(os.path.join(out_folder, "detection")):
        os.makedirs(os.path.join(out_folder, "detection"))

    for k, csv in detect_csvs.items():
        csv.to_csv(os.path.join(out_folder, "detection", "{}.csv".format(k)), index=False, header=False)

    if not os.path.exists(os.path.join(out_folder, "localization")):
        os.makedirs(os.path.join(out_folder, "localization"))

    for k, csv in loc_csvs.items():
        csv.to_csv(os.path.join(out_folder, "localization", "{}.csv".format(k)), index=False, header=False)


def detect_on_image(detections, dt, image):
    detect = 0
    confidence = 0
    anns = dt.getAnnIds(imgIds=image['id'], catIds=[1])
    if anns:
        detect = 1
        for ann in dt.loadAnns(anns):
            if confidence == 0:
                confidence = ann['score']
            if confidence < ann['score']:
                confidence = ann['score']

    detections.loc[-1] = [image['file_name'], detect, confidence]
    detections.index += 1
    detections.sort_index()


def localize_on_image(dt, image, localization):
    anns = dt.getAnnIds(imgIds=image['id'], catIds=[1])
    if anns:
        for ann in dt.loadAnns(anns):
            x, y, w, h = ann['bbox']
            centroid_x = x + 0.5 * w
            centroid_y = y + 0.5 * h

            localization.loc[-1] = [image['file_name'], centroid_x, centroid_y, ann['score']]
            localization.index += 1
            localization.sort_index()


def show_bbox(bbox):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y]]

    p = Polygon(poly, fill=False)

    return p


def evalutate(gt, dt, mode='bbox'):
    evaluation = cocoeval.COCOeval(gt, dt, mode)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()


def save_validation_images(gt, dt_bbox, dt_segm, im_ids, save_dir):
    for im_id in im_ids:
        gt_im_annots = gt.getAnnIds(imgIds=im_id, catIds=[1])
        pred_im_annots = dt_bbox.getAnnIds(imgIds=im_id, catIds=[1])

        pred_annots = dt_bbox.loadAnns(ids=pred_im_annots)
        gt_annots = gt.loadAnns(ids=gt_im_annots)

        print("validating ", results_data['images_folder'] + gt.imgs[im_id]['file_name'])

        fig, (ax_orig, ax_masks) = plt.subplots(2, 1, figsize=(10, 15))
        ax_orig.imshow(plt.imread(results_data['images_folder'] + gt.imgs[im_id]['file_name']))
        ax_masks.imshow(plt.imread(results_data['images_folder'] + gt.imgs[im_id]['file_name']))

        if len(gt_annots) > 0:
            gt_bboxes = []

            for gt_annot in gt_annots:
                gt_polygon = show_bbox(gt_annot["bbox"])
                gt_bboxes.append(gt_polygon)

            annIds = gt.getAnnIds(imgIds=im_id, catIds=[1])
            anns = gt.loadAnns(annIds)
            showAnns(gt, anns, gt_color)

            p = PatchCollection(gt_bboxes, alpha=0.3)
            p.set_facecolor('none')
            p.set_edgecolor(gt_color)
            p.set_linewidth(3)
            ax_masks.add_collection(p)

        if len(pred_annots) > 0:
            pred_bboxes = []

            for pred_annot in pred_annots:
                pred_polygon = show_bbox(pred_annot["bbox"])
                ax_masks.annotate("{:.2f}".format(pred_annot['score']), xy=pred_polygon.xy[1], annotation_clip=False,
                                  color='white')
                pred_bboxes.append(pred_polygon)
            if dt_segm is not None:
                annIds = dt_segm.getAnnIds(imgIds=im_id, catIds=[1])
                anns = dt_segm.loadAnns(annIds)
                showAnns(dt_segm, anns, pred_color)

            p = PatchCollection(pred_bboxes, alpha=0.3)
            p.set_facecolor('none')
            p.set_edgecolor(pred_color)
            p.set_linewidth(3)

            ax_masks.add_collection(p)
        print("saving to: ", os.path.join(save_dir, gt.imgs[im_id]['file_name']))
        #plt.show()

        ax_orig.axis('off')
        ax_masks.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, gt.imgs[im_id]['file_name']))
        plt.clf()
        plt.close()


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument("--ims", action='store_true', default=False)
    ap.add_argument("--no_metrics", action='store_false', default=True)
    ap.add_argument("--dataset", type=str)

    experiment = "results/colon_classif_cosine+blur"

    data_dict = {
        "cvc-val": {
            'annotation_file': "../datasets/CVC-VideoClinicDBtrain_valid/annotations/val.json",
            'images_folder': "../datasets/CVC-VideoClinicDBtrain_valid/images/",
            'results_folder': "../{}/inference/cvc-clinic-val/",
            'split': "val"
        },
        "cvc-test": {
            'annotation_file': "../datasets/cvcvideoclinicdbtest/annotations/test.json",
            'images_folder': "../datasets/cvcvideoclinicdbtest/images/",
            'results_folder': "../{}/inference/cvc-clinic-test/",
            'split': "test"
        },
        "etis": {
            'annotation_file': "../datasets/ETIS-LaribPolypDB/annotations/train.json",
            'images_folder': "../datasets/ETIS-LaribPolypDB/images/",
            'results_folder': "../{}/inference/etis-larib/",
            'split': "val"
        },
        "cvc-classif": {
            'annotation_file': "../datasets/CVC-classification/annotations/train.json",
            'images_folder': "../datasets/CVC-classification/images/",
            'results_folder': "../results/clinic612-dcn/inference/cvc-classification/",
            'split': "val"
        },
        "cvc-colondb-val": {
            'annotation_file': "../datasets/cvc-colondb-300/annotations/train.json",
            'images_folder': "../datasets/cvc-colondb-300/images/",
            'results_folder': "../{}/inference/cvc-colondb-val/",
            'split': "val"
        }
    }

    params = ap.parse_args()

    print(params)
    print(params.dataset)
    results_data = data_dict[params.dataset]
    save_ims = params.ims
    calc_metrics = params.no_metrics

    gt_color = "blue"
    pred_color = "gold"

    # ground truth
    gt = coco.COCO(results_data['annotation_file'])

    # predictions
    eval_segm = os.path.exists(os.path.join(results_data['results_folder'].format(experiment), "segm.json"))
    dt_bbox = gt.loadRes(os.path.join(results_data['results_folder'].format(experiment), "bbox.json"))
    det_segm = gt.loadRes(os.path.join(results_data['results_folder'].format(experiment), "segm.json")) if eval_segm else None

    evalutate(gt, dt_bbox, 'bbox')
    if eval_segm:
        evalutate(gt, det_segm, 'segm')

    detection_df = pd.DataFrame(columns=['image', "has_polyp", "confidence"])
    localization_df = pd.DataFrame(columns=['image', "center_x", "center_y", "confidence"])
    if calc_metrics:
        calc_challenge_metrics(localization_df, detection_df, dt_bbox, results_data['results_folder'].format(experiment), results_data['split'])

    if save_ims:
        im_ids = gt.getImgIds()
        save_dir = os.path.join(results_data['results_folder'].format(experiment), "ims")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_validation_images(gt, dt_bbox, det_segm, im_ids, save_dir)
