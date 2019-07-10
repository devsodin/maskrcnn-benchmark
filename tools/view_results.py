from matplotlib import pyplot as plt
from pycocotools import coco, cocoeval
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pycocotools.mask import decode, encode
import pandas as pd
import os


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
    anns = dt.getAnnIds(imgIds=image['id'])
    if anns:
        detect = 1
        for ann in dt.loadAnns(anns):
            confidence += ann['score']

        confidence /= len(anns)
    detections.loc[-1] = [image['file_name'], detect, confidence]
    detections.index += 1
    detections.sort_index()


def localize_on_image(dt, image, localization):
    anns = dt.getAnnIds(imgIds=image['id'])
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

def show_mask(mask):

    return


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

        print("validating ",results_data['images_folder'] + gt.imgs[im_id]['file_name'])

        fig, ax = plt.subplots()
        ax.imshow(plt.imread(results_data['images_folder'] + gt.imgs[im_id]['file_name']))

        if len(gt_annots) > 0:
            gt_bboxes = []

            for gt_annot in gt_annots:
                gt_polygon = show_bbox(gt_annot["bbox"])
                gt_bboxes.append(gt_polygon)

            annIds = gt.getAnnIds(imgIds=im_id)
            anns = gt.loadAnns(annIds)
            gt.showAnns(anns)

            p = PatchCollection(gt_bboxes, alpha=0.3)
            p.set_facecolor('none')
            p.set_edgecolor(gt_color)
            p.set_linewidth(3)
            ax.add_collection(p)

        if len(pred_annots) > 0:
            pred_bboxes = []

            for pred_annot in pred_annots:
                pred_polygon = show_bbox(pred_annot["bbox"])
                pred_bboxes.append(pred_polygon)

            annIds = dt.getAnnIds(imgIds=im_id)
            anns = dt.loadAnns(annIds)
            dt.showAnns(anns)

            p = PatchCollection(pred_bboxes, alpha=0.3)
            p.set_facecolor('none')
            p.set_edgecolor(pred_color)
            p.set_linewidth(3)

            ax.add_collection(p)
        print("saving to: ", os.path.join(save_dir, gt.imgs[im_id]['file_name']))
        plt.show()

        plt.savefig(os.path.join(save_dir, gt.imgs[im_id]['file_name']))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    results_data = {
        'annotation_file': "../datasets/CVC-VideoClinicDBtrain_valid/annotations/val.json",
        'images_folder': "../datasets/CVC-VideoClinicDBtrain_valid/images/",
        'results_folder': "../out_train_classif/inference/cvc-clinic-val/",
        'split': "val"

    }

    # results_data = {
    #    'annotation_file': "../datasets/cvcvideoclinicdbtest/annotations/test.json",
    #    'images_folder': "../datasets/cvcvideoclinicdbtest/images/",
    #    'results_folder': "../inference/cvc-clinic-test/",
    #    'split': "test"
    # }


    save_ims = True

    gt_color = "orange"
    pred_color = "blue"

    # ground truth
    gt = coco.COCO(results_data['annotation_file'])

    # predictions
    dt = gt.loadRes(os.path.join(results_data['results_folder'], "bbox.json"))

    evalutate(gt, dt, 'bbox')
    evalutate(gt, dt, 'segm')

    detection = pd.DataFrame(columns=['image', "has_polyp", "confidence"])
    localization = pd.DataFrame(columns=['image', "center_x", "center_y", "confidence"])
    calc_challenge_metrics(localization, detection, dt, results_data['results_folder'], results_data['split'])

    if save_ims:
        im_ids = gt.getImgIds()
        save_dir = os.path.join(results_data['results_folder'], "ims")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_validation_images(gt, dt, im_ids, save_dir)

