from matplotlib import pyplot as plt
from pycocotools import coco, cocoeval
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pandas as pd
import os


def remap_names(csv):
    csv['image'] = csv.image.map(lambda x: "".join(x.split("0")))
    return csv


def calc_challenge_detection(dt, out_folder, split):
    detections = pd.DataFrame(columns=['image', 'detection', "confidence"])
    for image in dt.loadImgs(ids=dt.getImgIds()):
        detect = 0
        confidence = -1
        anns = dt.getAnnIds(imgIds=image['id'])
        if anns:
            detect = 1
            confidence = 0
            for ann in dt.loadAnns(anns):
                confidence += ann['score']

            confidence /= len(anns)

        detections.loc[-1] = [image['file_name'], detect, confidence]
        detections.index += 1
        detections.sort_index()
    detections.sort_values('image', inplace=True)
    detections.reset_index(inplace=True, drop=True)
    print(detections)
    detections = remap_names(detections)
    detections.to_csv(os.path.join(out_folder, "detection-{}.csv".format(split)))


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
    print(localization)
    detections = remap_names(localization)
    detections.to_csv(os.path.join(out_folder, "localization-{}.csv".format(split)))


def show_bbox(bbox, color):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y]]

    p = Polygon(poly)

    for point in poly:
        plt.scatter(point[0], point[1], c=color)

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
                gt_polygon = show_bbox(gt_annot["bbox"], gt_color)
                gt_polygons.append(gt_polygon)

            p = PatchCollection(gt_polygons, alpha=0.3)
            p.set_color(gt_color)
            ax.add_collection(p)

        if len(pred_annots) > 0:
            pred_polygons = []
            for pred_annot in pred_annots:
                pred_polygon = show_bbox(pred_annot["bbox"], pred_color)
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

    evalutate(gt, dt, 'bbox')

    calc_challenge_detection(dt, "../inference/cvc-clinic-test", "test")
    calc_challenge_localization(dt, "../inference/cvc-clinic-test", "test")

    im_ids = gt.getImgIds()
    save_validation_images(gt,dt,im_ids,"../inference/cvc-clinic-test/test/")
