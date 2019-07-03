from matplotlib import pyplot as plt
from pycocotools import coco, cocoeval
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def show_bbox(bbox, color):
    [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y]]

    p = Polygon(poly)

    for point in poly:
        print(point)
        plt.scatter(point[0], point[1], c=color)

    return p


if __name__ == '__main__':

    annotation_file = "datasets/CVC-VideoClinicDBtrain_valid/annotations/val.json"
    images_folder = "datasets/CVC-VideoClinicDBtrain_valid/images/"
    results_file = "inference/cvc-clinic-val/bbox.json"

    gt_color = "c"
    pred_color = "m"

    gt = coco.COCO(annotation_file)
    dt = gt.loadRes(results_file)
    evaluation = cocoeval.COCOeval(gt, dt, "bbox")

    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

    ims_id = gt.getImgIds()

    for im_id in ims_id:
        print(im_id)
        gt_im_annots = gt.getAnnIds(imgIds=im_id)
        pred_im_annots = dt.getAnnIds(imgIds=im_id)

        pred_annots = dt.loadAnns(ids=dt.getAnnIds(imgIds=im_id))
        gt_annots = gt.loadAnns(ids=gt_im_annots)

        print(pred_annots)
        print(gt_annots)

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

        plt.savefig("inference/cvc-clinic-val/val/" + gt.imgs[im_id]['file_name'])
        plt.clf()
        plt.close()
