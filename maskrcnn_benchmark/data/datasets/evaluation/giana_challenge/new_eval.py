import logging
import os
import time
from glob import glob

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pycocotools import coco
from scipy.ndimage.measurements import label


def save_detection_plot(output_folder, threshold, vid_folder, video_gt, video_pred):
    title = "Video: {} - threshold: {}".format(vid_folder.split("/")[-1], threshold)
    plt.title(title)
    plt.plot(video_gt, color='blue')
    plt.plot(video_pred, color='gold')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.savefig(os.path.join(output_folder, "detect_plot-{}-{}.png".format(vid_folder.split("/")[-1], threshold)))
    plt.clf()


def process_video(results_csv, output_with_confidence, thresholds, masks_files):
    video_len = len(masks_files) + 1
    det = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
    loc = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])

    for threshold in thresholds:

        video_gt = np.zeros((video_len, 1))
        video_predictions = np.zeros((video_len, 1))

        first_polyp = -1
        first_detected_polyp_det = -1
        first_detected_polyp_loc = -1

        tp_det, fp_det, fn_det, tn = 0, 0, 0, 0
        tp_loc, fp_loc, fn_loc, tn = 0, 0, 0, 0

        # 8-connected
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

        for frame in masks_files:
            polyp_n = int(frame.split("/")[-1].split("_")[0].split("-")[1])
            im_frame = cv2.imread(frame, 0)

            is_polyp = im_frame.sum() > 0
            video_gt[polyp_n] = 1 if is_polyp else 0

            labeled_frame, max_polyp = label(im_frame, structure=kernel)

            if is_polyp and first_polyp == -1:
                first_polyp = polyp_n

            frame_output = results_csv.loc[results_csv.image.astype("int") == polyp_n]
            if output_with_confidence:
                frame_output = frame_output.loc[frame_output.confidence >= threshold]

            if frame_output.empty:
                if is_polyp:
                    fn_det += 1
                    fn_loc += max_polyp
                else:
                    tn += 1
            else:
                pred_out = frame_output.has_polyp.tolist()[0]
                if pred_out:
                    if is_polyp:
                        tp_det += 1
                        if first_detected_polyp_det == -1:
                            first_detected_polyp_det = polyp_n
                    else:
                        fp_det += 1
                else:
                    if is_polyp:
                        fn_det += 1
                    else:
                        tn += 1

                video_predictions[polyp_n] += 1

                already_detected = []
                for detection_row in frame_output.iterrows():
                    detection = detection_row[1]
                    centroid_x = int(detection[2])
                    centroid_y = int(detection[3])
                    if is_polyp:
                        if im_frame[centroid_y, centroid_x] == 255:
                            if labeled_frame[centroid_y, centroid_x] not in already_detected:
                                tp_loc += 1
                                already_detected += [labeled_frame[centroid_y, centroid_x]]

                                if first_detected_polyp_loc == -1:
                                    first_detected_polyp_loc = polyp_n
                        else:
                            fp_loc += 1
                    else:
                        fp_loc += 1

                detected_in_frame = len(set(already_detected))
                fn_loc += (max_polyp - detected_in_frame)

        plot_folder = os.path.join(output_folder,"plots")
        # save_detection_plot(plot_folder, threshold, masks_files, video_gt, video_predictions)
        rt_det = first_detected_polyp_det - first_polyp if first_detected_polyp_det != -1 else np.NaN
        rt_loc = first_detected_polyp_loc - first_polyp if first_detected_polyp_loc != -1 else np.NaN

        det.loc[-1] = [threshold, tp_det, fp_det, tn, fn_det, rt_det]
        det.index += 1
        det.sort_index()
        det.reset_index(inplace=True, drop=True)

        loc.loc[-1] = [threshold, tp_loc, fp_loc, tn, fn_loc, rt_loc]
        loc.index += 1
        loc.sort_index()
        loc.reset_index(inplace=True, drop=True)

    return det, loc


def calc_average_results_with_metrics(df_results, output_folder, file_ext):
    avg = pd.DataFrame(
        columns=["TP", "FP", "TN", "FN", 'Accuracy', "Precision", "Recall", "Specificity", "Mean RT", "RT std"])

    thresholds = list(set(df_results.threshold))

    for threshold in thresholds:
        df_results_thresh = df_results[df_results.threshold == threshold]

        tp = df_results_thresh.TP.sum()
        fp = df_results_thresh.FP.sum()
        fn = df_results_thresh.FN.sum()
        tn = df_results_thresh.TN.sum()
        mean_rt = df_results_thresh.RT.mean(skipna=True)
        std_rt = df_results_thresh.RT.std(skipna=True)

        acc = (tp + tn) / (tp + fp + fn + tn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (fp + tn)

        avg.loc[-1] = [tp, fp, tn, fn, acc, pre, rec, spec, mean_rt, std_rt]
        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)

    avg.to_csv(os.path.join(output_folder, "average_{}.csv".format(file_ext)), )


def giana_mkdir(output_folder):
    if not os.path.exists(os.path.join(output_folder, "results")):
        os.makedirs(os.path.join(output_folder, "results"))
    if not os.path.exists(os.path.join(output_folder, "giana_results")):
        os.makedirs(os.path.join(output_folder, "giana_results"))


def compute_detections_and_localizations(coco_dt, sequences, output_folder):
    det_loc_results = pd.DataFrame(columns=['image', "has_polyp", "center_x", "center_y", "confidence"])

    for image in coco_dt.loadImgs(ids=coco_dt.getImgIds()):

        detect = 0
        anns = coco_dt.getAnnIds(imgIds=image['id'])

        if anns:
            detect = 1
            avg_confidence = 0
            for ann in coco_dt.loadAnns(anns):
                confidence = ann['score']
                avg_confidence += confidence
                x, y, w, h = ann['bbox']
                centroid_x = x + 0.5 * w
                centroid_y = y + 0.5 * h

                det_loc_results.loc[-1] = [image['file_name'], detect, centroid_x, centroid_y, confidence]
                det_loc_results.index += 1
                det_loc_results.sort_index()
        #
        # else:
        #     det_loc_results.loc[-1] = [image['file_name'], detect, 0, 0, 1.]
        #     det_loc_results.index += 1
        #     det_loc_results.sort_index()

    det_loc_results.sort_values(by='image', inplace=True)
    det_loc_results[['seq', 'image']] = det_loc_results['image'].str.split("-", expand=True)
    det_loc_results['image'] = det_loc_results['image'].map(lambda x: str(int(x.split(".")[0])))

    for sequence in sequences:
        det_loc_results[det_loc_results['image'].str.contains(sequence)].to_csv(
            os.path.join(output_folder, "r{}".format(sequence)), index=False)

    return det_loc_results


def do_giana_eval(dataset_folder,
                  output_folder,
                  annot_file
                  ):
    results_folder = os.path.join(output_folder, "results")
    giana_results_folder = os.path.join(output_folder, "giana_results")

    giana_mkdir(output_folder)

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Evaluating results (GIANA Challenge)")

    masks_folder = os.path.join(dataset_folder, "masks")
    thresholds = [x / 10 for x in range(10)]
    sequences = set([mask.split("/")[-1].split("-")[0] for mask in glob(os.path.join(masks_folder, "*.tif"))])

    gt = coco.COCO(annot_file)
    dt_bbox = gt.loadRes(os.path.join(output_folder, "bbox.json"))
    det_loc_results = compute_detections_and_localizations(dt_bbox, sequences, results_folder)

    logger.info("Metrics calculated (GIANA Challenge)")

    all_det = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])
    all_loc = pd.DataFrame(columns=["threshold", "TP", "FP", "TN", "FN", "RT"])

    for sequence in sequences:
        masks_files = sorted(os.listdir(masks_folder))
        masks_files = [os.path.join(dataset_folder, "masks", mask) for mask in masks_files if sequence + "-" in mask]
        det, loc = process_video(det_loc_results[det_loc_results['seq'] == sequence], True, thresholds,
                                 masks_files)

        all_det = pd.concat([all_det, det])
        all_loc = pd.concat([all_loc, loc])

        det.to_csv(os.path.join(giana_results_folder, "d_{}".format(sequence)))
        loc.to_csv(os.path.join(giana_results_folder, "l_{}".format(sequence)))

    calc_average_results_with_metrics(all_det, giana_results_folder, "det")
    calc_average_results_with_metrics(all_loc, giana_results_folder, "loc")


if __name__ == '__main__':
    output_folder = "results_october/vanilla/inference/cvc-clinic-test/"
    dataset_root_folder = "datasets/cvcvideoclinicdbtest"
    dataset_ann = dataset_root_folder + "/annotations/test.json"
    folder_detection = os.path.join(output_folder, "detection")
    folder_localization = os.path.join(output_folder, "localization")
    folder_gt = os.path.join(dataset_root_folder, "masks")
    import time
    t = time.time()
    do_giana_eval(dataset_root_folder, output_folder, dataset_ann)
    print("new", time.time() - t)
