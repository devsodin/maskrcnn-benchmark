import logging
import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools import coco
from scipy.ndimage.measurements import label


def calc_challenge_metrics(loc_df, detect_df, dt, out_folder, dataset):
    for image in dt.loadImgs(ids=dt.getImgIds()):
        detect_on_image(detect_df, dt, image)
        localize_on_image(dt, image, loc_df)

    detect_df.sort_values('image', inplace=True)
    detect_df.reset_index(inplace=True, drop=True)
    loc_df.sort_values('image', inplace=True)
    loc_df.reset_index(inplace=True, drop=True)

    detect_csvs = reformat_csvs(detect_df, dataset=dataset)
    loc_csvs = reformat_csvs(loc_df, dataset=dataset)

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
            if confidence == 0:
                confidence = ann['score']
            if confidence < ann['score']:
                confidence = ann['score']

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


def delete_seq_code(csv):
    csv['image'] = csv.image.map(lambda x: int(x.split("-")[1].split(".")[0]))
    return csv


def reformat_csvs(detections, dataset):
    d = {}

    if dataset == 'cvc-val':
        seqs = range(16, 19)
    elif dataset == 'cvc-test':
        seqs = range(1, 19)
    elif dataset == 'cvc-segmented-test':
        seqs = [5, 15, 16]
    elif dataset == 'cvc-elipses-test':
        seqs = [5, 15, 16]
    else:
        raise ('invalid split - {}'.format(dataset))

    for seq in seqs:
        seq_rows = detections.loc[detections['image'].str.contains("{:03d}-".format(seq))].copy(deep=True)
        seq_rows = delete_seq_code(seq_rows)
        seq_rows.sort_values(by='image', inplace=True)
        d[seq] = seq_rows

    return d


def do_giana_eval(results_folder, folder_detection, folder_localization, folder_gt, root_folder_output, annot_file,
                  dataset_name):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Evaluating results (GIANA Challenge)")

    detection_df = pd.DataFrame(columns=['image', "has_polyp", "confidence"])
    localization_df = pd.DataFrame(columns=['image', "center_x", "center_y", "confidence"])

    # ground truth
    gt = coco.COCO(annot_file)

    if dataset_name == 'cvc-clinic-test':
        dataset = 'cvc-test'
    elif dataset_name == 'cvc-video-segmented-test':
        dataset = "cvc-segmented-test"
    elif dataset_name == 'cvc-video-elipses-test':
        dataset = "cvc-elipses-test"
    else:
        raise ("invalid  giana dataset evaluation")

    # predictions
    dt_bbox = gt.loadRes(os.path.join(results_folder, "bbox.json"))
    calc_challenge_metrics(localization_df, detection_df, dt_bbox, results_folder, dataset)
    logger.info("Metrics calculated (GIANA Challenge)")

    folder_output_detection = os.path.join(root_folder_output, "detection")
    folder_output_localization = os.path.join(root_folder_output, "localization")
    average_detection_output_file = os.path.join(folder_output_detection, "average.csv")
    average_localization_output_file = os.path.join(folder_output_localization, "average.csv")
    thresholds = [x / 10 for x in range(10)]

    if not os.path.exists(folder_output_detection):
        os.makedirs(folder_output_detection)
    if not os.path.exists(folder_output_localization):
        os.makedirs(folder_output_localization)

    files_detection = sorted(os.listdir(folder_detection))
    files_localization = sorted(os.listdir(folder_localization))

    results_detection = {}
    results_localization = {}
    for detection, localization in zip(files_detection, files_localization):
        detection_csv = os.path.join(folder_detection, detection)
        detection_df = pd.read_csv(detection_csv, header=None)
        detection_confidence = detection_df.shape[1] > 2

        localization_csv = os.path.join(folder_localization, localization)
        localization_df = pd.read_csv(localization_csv, header=None)
        localization_confidence = localization_df.shape[1] > 3

        # both named the same
        vid_name = localization_csv.split("/")[-1].split(".")[0]
        gt_vid_folder = os.path.join(folder_gt, vid_name)

        plot_folder = os.path.join(root_folder_output, "plots")
        logger.info("Processing video {} - (GIANA Challenge)".format(vid_name))
        res_detection, res_localization = generate_results_per_video((detection_df, localization_df),
                                                                     (detection_confidence, localization_confidence),
                                                                     thresholds, gt_vid_folder,
                                                                     plot_folder=plot_folder)

        pd.DataFrame.from_dict(res_detection, columns=["TP", "FP", "FN", "TN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_detection, "d{}.csv".format(vid_name)))
        results_detection[vid_name] = res_detection

        pd.DataFrame.from_dict(res_localization, columns=["TP", "FP", "FN", "TN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_localization, "l{}.csv".format(vid_name)))
        results_localization[vid_name] = res_localization

    calculate_average_results(results_detection, thresholds, average_detection_output_file)
    calculate_average_results(results_localization, thresholds, average_localization_output_file)


def calculate_average_results(results_dict: dict, thresholds, output_file):
    avg = pd.DataFrame(columns=["TP", "FP", "FN", "TN", 'Accuracy', "Precision", "Recall", "Specificity", "Mean RT"])
    for threshold in thresholds:
        # TP, FP, FN, TN, RT
        results = [0, 0, 0, 0]
        srt = 0
        drt = 0
        for vid, res_dict in results_dict.items():
            results = [res + new for res, new in zip(results, res_dict[threshold][:-1])]
            srt = srt + res_dict[threshold][-1] if res_dict[threshold][-1] != -1 else srt
            drt = drt + 1 if res_dict[threshold][-1] != -1 else drt

        tp, fp, fn, tn = results[0], results[1], results[2], results[3]
        acc = (tp + tn) / (tp + fp + fn + tn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (fp + tn)
        mean_rt = srt / drt
        results = [int(x) for x in results]
        row = results + [acc, pre, rec, spec, mean_rt]
        avg.loc[-1] = row

        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)

    avg.to_csv(output_file)


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


def process_video_for_detection(file, has_confidence, thresh, vid_folder, plot_folder):
    video_len = len(os.listdir(vid_folder)) +1
    #len(set([im.split("_Polyp")[0] for im in os.listdir(vid_folder)])) + 1
    video_gt = np.zeros((video_len, 1))
    video_pred = np.zeros((video_len, 1))

    first_polyp = -1
    first_detected_polyp = -1

    tp, fp, fn, tn = 0, 0, 0, 0
    for frame in sorted(os.listdir(vid_folder)):
        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        is_polyp = np.asarray(im_frame).sum() > 0
        video_gt[polyp_n] = 1 if is_polyp else 0

        if is_polyp and first_polyp == -1:
            first_polyp = polyp_n

        frame_output = file.loc[file[0] == polyp_n]
        if has_confidence:
            frame_output = frame_output.loc[frame_output[2] >= thresh]

        if frame_output.empty:
            if is_polyp:
                fn += 1
            else:
                tn += 1
        else:
            pred_out = frame_output[1].tolist()[0]
            if pred_out:
                if is_polyp:
                    tp += 1
                    if first_detected_polyp == -1:
                        first_detected_polyp = polyp_n
                else:
                    fp += 1
            else:
                if is_polyp:
                    fn += 1
                else:
                    tn += 1

            video_pred[polyp_n] += 1

    save_detection_plot(plot_folder, thresh, vid_folder,video_gt, video_pred)
    rt = first_detected_polyp - first_polyp if first_detected_polyp != -1 else -1

    return [tp, fp, fn, tn, rt], video_gt, video_pred


def process_video_for_localization(file, has_confidence, threshold, vid_folder):
    tp, fp, fn, tn = 0, 0, 0, 0
    first_polyp = -1
    first_detected_polyp = -1

    for frame in sorted(os.listdir(vid_folder)):
        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        im_frame_np = np.asarray(im_frame)
        is_polyp = im_frame_np.sum() > 0

        # 8-connected
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        labeled_frame, max_polyp = label(im_frame, structure=kernel)

        if is_polyp and first_polyp == -1:
            first_polyp = polyp_n
        frame_output = file.loc[file[0] == polyp_n]
        if has_confidence:
            frame_output = frame_output.loc[frame_output[3] >= threshold]

        if frame_output.empty:
            if is_polyp:
                fn += max_polyp
            else:
                tn += 1
        else:
            already_detected = []

            for detection_row in frame_output.iterrows():
                detection = detection_row[1]
                frame_pred = True
                centroid_x = int(detection[2])
                centroid_y = int(detection[1])
                if frame_pred:
                    if is_polyp:
                        if im_frame_np[centroid_x, centroid_y] == 255:
                            if labeled_frame[centroid_x, centroid_y] not in already_detected:
                                tp += 1
                                already_detected += [labeled_frame[centroid_x, centroid_y]]

                                if first_detected_polyp == -1:
                                    first_detected_polyp = polyp_n
                        else:
                            fp += 1
                    else:
                        fp += 1
                else:
                    if not is_polyp:
                        tn += 1

            detected_in_frame = len(set(already_detected))
            fn += (max_polyp - detected_in_frame)

    rt = first_detected_polyp - first_polyp if first_detected_polyp != -1 else -1
    return [tp, fp, fn, tn, rt]


def generate_results_per_video(videos, confidences, thresholds, gt, plot_folder):
    detect_dict = {}
    local_dict = {}
    for threshold in thresholds:
        # TODO change plots
        res_detection, _, _ = process_video_for_detection(videos[0], confidences[0], threshold, gt, plot_folder)
        res_localization = process_video_for_localization(videos[1], confidences[1], threshold, gt)

        detect_dict[threshold] = res_detection
        local_dict[threshold] = res_localization
    return detect_dict, local_dict
