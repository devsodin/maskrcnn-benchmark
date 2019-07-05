from matplotlib import pyplot as plt
from pycocotools import coco, cocoeval
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pandas as pd
import os


def remap_names(csv):
    csv['image'] = csv.image.map(lambda x: "".join(x.split("0")))
    #csv['image'] = csv.image.map(lambda x: "".join(x.split("0")).split("-")[1].split(".")[0])
    return csv

def delete_seq_code(csv):
    print(csv['image'])
    csv['image'] = csv.image.map(lambda x: int(x.split("-")[1].split(".")[0]))
    print(csv['image'])
    return csv

def reformat_csvs(detections, split='test'):
    d = {}
    seqs = []

    if split == 'val':
        seqs = range(16, 19)
    elif split == 'test':
        seqs = range(1, 19)
    else:
        raise ('invalid split - {}'.format(split))

    for seq in seqs:
        print(seq)
        if split == 'test':
            seq_rows = detections.loc[detections['image'].str.startswith("{}-".format(seq))].copy(deep=True)
            print(seq_rows)
        else:
            seq_rows = detections.loc[detections['image'].str.contains("{:03d}-".format(seq))].copy(deep=True)
        print(seq_rows)
        seq_rows = delete_seq_code(seq_rows)
        seq_rows.sort_values(by='image', inplace=True)
        d[seq] = seq_rows
        print(d[seq])

    return d


def calc_challenge_detection(dt, out_folder, split):
    detections = pd.DataFrame(columns=['image', 'detection', "confidence"])
    for image in dt.loadImgs(ids=dt.getImgIds()):
        detect = 0
        confidence = 0
        anns = dt.getAnnIds(imgIds=image['id'])
        if anns:
            detect = 1
            for ann in dt.loadAnns(anns):
                confidence += ann['score']

            confidence /= len(anns)

        detections.loc[-1] = [image['file_name'], detect, confidence]
        detections.index += 1
        detections.sort_index()

    detections.sort_values('image', inplace=True)
    detections.reset_index(inplace=True, drop=True)

    detection_csvs = reformat_csvs(detections, split=split)

    if not os.path.exists(os.path.join(out_folder, 'detection')):
        os.makedirs(os.path.join(out_folder, 'detection'))

    for k, csv in detection_csvs.items():
        csv.to_csv(os.path.join(out_folder, 'detection', "{}.csv".format(k)))


def calc_challenge_localization(dt, out_folder, split):
    localization = pd.DataFrame(columns=['image', "center_x", "center_y", "confidence"])
    for image in dt.loadImgs(ids=dt.getImgIds()):
        anns = dt.getAnnIds(imgIds=image['id'])
        if anns:
            for ann in dt.loadAnns(anns):
                x, y, w, h = ann['bbox']
                centroid_x = x + 0.5 * w
                centroid_y = y + 0.5 * h

                localization.loc[-1] = [image['file_name'], centroid_x, centroid_y, ann['score']]
                localization.index += 1
                localization.sort_index()

    localization.sort_values('image', inplace=True)
    localization.reset_index(inplace=True, drop=True)

    localization_csvs = reformat_csvs(localization, split=split)

    if not os.path.exists(os.path.join(out_folder, 'localization')):
        os.makedirs(os.path.join(out_folder, 'localization'))

    for k, csv in localization_csvs.items():
        csv.to_csv(os.path.join(out_folder, 'localization', "{}.csv".format(k)))




def show_bbox(bbox):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y]]

    p = Polygon(poly)

    return p


def evalutate(gt, dt, mode='bbox'):
    evaluation = cocoeval.COCOeval(gt, dt, mode)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()


def save_validation_images(gt, dt, im_ids, save_dir):
    for im_id in im_ids:
        gt_im_annots = gt.getAnnIds(imgIds=im_id)
        pred_im_annots = dt.getAnnIds(imgIds=im_id)

        pred_annots = dt.loadAnns(ids=pred_im_annots)
        gt_annots = gt.loadAnns(ids=gt_im_annots)

        print(images_folder + gt.imgs[im_id]['file_name'])

        fig, ax = plt.subplots()
        ax.imshow(plt.imread(images_folder + gt.imgs[im_id]['file_name']))
        if len(gt_annots) > 0:
            gt_polygons = []
            for gt_annot in gt_annots:
                gt_polygon = show_bbox(gt_annot["bbox"])
                gt_polygons.append(gt_polygon)

            p = PatchCollection(gt_polygons, alpha=0.3)
            p.set_color(gt_color)
            ax.add_collection(p)

        if len(pred_annots) > 0:
            pred_polygons = []
            for pred_annot in pred_annots:
                pred_polygon = show_bbox(pred_annot["bbox"])
                pred_polygons.append(pred_polygon)

            p = PatchCollection(pred_polygons, alpha=0.3)
            p.set_color(pred_color)

            ax.add_collection(p)

        plt.savefig(save_dir + gt.imgs[im_id]['file_name'])
        plt.clf()
        plt.close()


if __name__ == '__main__':
    #annotation_file = "../datasets/CVC-VideoClinicDBtrain_valid/annotations/val.json"
    #images_folder = "../datasets/CVC-VideoClinicDBtrain_valid/images/"
    #results_file = "../inference/cvc-clinic-val/bbox.json"

    annotation_file = "../datasets/cvcvideoclinicdbtest/annotations/test.json"
    images_folder = "../datasets/cvcvideoclinicdbtest/images/"
    results_file = "../inference/cvc-clinic-test/bbox.json"

    gt_color = "c"
    pred_color = "m"

    # ground truth
    gt = coco.COCO(annotation_file)

    # predictions
    dt = gt.loadRes(results_file)

    #evalutate(gt, dt, 'bbox')

    calc_challenge_detection(dt, "../inference/cvc-clinic-test", "test")
    calc_challenge_localization(dt, "../inference/cvc-clinic-test", "test")

    # im_ids = gt.getImgIds()
    # save_validation_images(gt,dt,im_ids,"../inference/cvc-clinic-test/test-mask/")
