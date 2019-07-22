import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label


def calculate_average_results(results_dict: dict, thresholds, output_file):

    avg = pd.DataFrame(columns=["TP", "FP", "FN", "TN", 'Accuracy', "Precision", "Recall", "Specificity", "Mean RT"])
    for threshold in thresholds:
        # TP, FP, FN, TN, RT
        results = [0, 0, 0, 0]
        sums = [0, 0, 0, 0]
        srt = 0
        drt = 0
        for vid, res_dict in results_dict.items():
            results = [res + new for res, new in zip(results, res_dict[threshold][:-1])]
            sums = [val + new for val, new in zip(sums,results)]
            print(sums)
            srt = srt + res_dict[threshold][-1] if res_dict[threshold][-1] != -1 else srt
            drt = drt + 1 if res_dict[threshold][-1] != -1 else drt

        tp, fp, fn, tn = sums[0], sums[1], sums[2], sums[3]
        acc = (tp + tn) / (tp + fp + fn + tn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (fp + tn)
        mean_rt = srt / drt
        row = results + [acc, pre, rec, spec, mean_rt]
        avg.loc[-1] = row

        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)
    print(avg)

    avg.to_csv(output_file)


def save_detection_plot(output_folder, threshold, vid_folder, video_gt, video_pred):
    title = "Video: {} - threshold: {}".format(vid_folder.split("/")[-1], threshold)
    plt.title(title)
    plt.plot(video_gt, color='blue')
    plt.plot(video_pred, color='gold')
    plt.savefig(os.path.join(output_folder, "detect_plot-{}-{}.png".format(vid_folder.split("/")[-1], threshold)))


def process_video_for_detection(file, has_confidence, thresh, vid_folder):
    video_len = len(os.listdir(vid_folder)) + 1
    video_gt = np.zeros((video_len, 1))
    video_pred = np.zeros((video_len, 1))

    first_polyp = -1
    first_detected_polyp = -1

    tp, fp, fn, tn = 0, 0, 0, 0
    for frame in sorted(os.listdir(vid_folder)):

        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        is_polyp = np.asarray(im_frame).sum() > 0
        video_gt[polyp_n] = 1.1 if is_polyp else 0

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

            video_pred[polyp_n] = 0.9

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


def generate_results_per_video(videos, confidences, thresholds, gt):
    detect_dict = {}
    local_dict = {}
    for threshold in thresholds:
        # TODO change plots
        res_detection, _, _ = process_video_for_detection(videos[0], confidences[0], threshold, gt)
        res_localization = process_video_for_localization(videos[1], confidences[1], threshold, gt)
        print(threshold, "done")

        detect_dict[threshold] = res_detection
        local_dict[threshold] = res_localization
    return detect_dict, local_dict


def main():
    folder_detection = "../out/test_minsize_anchors/inference/cvc-clinic-test/detection/"
    folder_localization = "../out/test_minsize_anchors/inference/cvc-clinic-test/localization/"
    folder_gt = "/home/yael/Downloads/PolypDetectionTest/GT"
    folder_output = "../out/test_minsize_anchors/inference/cvc-clinic-test/res/"
    folder_output_detection = os.path.join(folder_output, "detection")
    folder_output_localization = os.path.join(folder_output, "localization")
    average_detection_output_file = os.path.join(folder_output_detection, "average.csv")
    average_localization_output_file = os.path.join(folder_output_localization, "average.csv")
    thresholds = [x / 5 for x in range(5)]

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

        res_detection, res_localization = generate_results_per_video((detection_df, localization_df),
                                                                     (detection_confidence, localization_confidence),
                                                                     thresholds, gt_vid_folder)

        pd.DataFrame.from_dict(res_detection, columns=["TP", "FP", "FN", "TN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_detection, "d{}.csv".format(vid_name)))
        results_detection[vid_name] = res_detection

        pd.DataFrame.from_dict(res_localization, columns=["TP", "FP", "FN", "TN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_localization, "l{}.csv".format(vid_name)))
        results_localization[vid_name] = res_localization

    calculate_average_results(results_detection, thresholds, average_detection_output_file)
    calculate_average_results(results_localization, thresholds, average_localization_output_file)


if __name__ == '__main__':
    main()
