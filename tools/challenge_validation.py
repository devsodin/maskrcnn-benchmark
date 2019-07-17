import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label

def polyp_detection_results(csv_folder, output_folder, gt_folder, threshold=0.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    res_dict = {}
    for results_file in sorted(os.listdir(csv_folder)):
        results_csv = os.path.join(csv_folder, results_file)
        file = pd.read_csv(results_csv, header=None)

        has_confidence = file.shape[1] > 2
        vid_name = results_file.split("/")[-1].split(".")[0]
        vid_folder = os.path.join(gt_folder, vid_name)

        fn, fp, rt, tn, tp, video_gt, video_pred = process_video(file, has_confidence, threshold,
                                                                 vid_folder)

        save_detection_plot(output_folder, threshold, vid_folder, video_gt, video_pred)

        res_dict[vid_name] = [tp, fp, fn, tn, rt]
        print("output: {}, total ims: {}, gt ims: {}".format([tp, fp, fn, tn, rt], sum([tp, fp, fn, tn]),
                                                             len(sorted(os.listdir(vid_folder)))))

    pd.DataFrame.from_dict(res_dict, orient="index", columns=["TP", "FP", "FN", "TN", "RT"]).to_csv(
        output_folder + "/res{}.csv".format(threshold), index=False)

    return res_dict


def calculate_average_detection_results(results_dict):
    metrics_sum = [0, 0, 0, 0, 0]

    for video, results in results_dict.items():
        metrics_sum += results[video]



def process_video(file, has_confidence, thresh, vid_folder):
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

    return fn, fp, rt, tn, tp, video_pred, video_gt


def save_detection_plot(output_folder, threshold, vid_folder, video_gt, video_pred):
    title = "Video: {} - threshold: {}".format(vid_folder.split("/")[-1], threshold)
    plt.title(title)
    plt.plot(video_gt, color='blue')
    plt.plot(video_pred, color='gold')
    plt.savefig(os.path.join(output_folder, "detect_plot-{}-{}.png".format(vid_folder.split("/")[-1], threshold)))


def polyp_localization_challenge(csv_folder, output_folder, gt_folder, threshold=0.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    res_dict = {}
    for results_file in sorted(os.listdir(csv_folder)):
        results_csv = os.path.join(csv_folder, results_file)
        file = pd.read_csv(results_csv, header=None)

        has_confidence = file.shape[1] > 4
        vid_name = results_file.split("/")[-1].split(".")[0]
        vid_folder = os.path.join(gt_folder, vid_name)

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        first_polyp = -1
        first_detected_polyp = -1

        for frame in sorted(os.listdir(vid_folder)):
            polyp_n = int(frame.split("_")[0].split("-")[1])
            im_frame = Image.open(os.path.join(vid_folder, frame))
            im_frame_np = np.asarray(im_frame)
            is_polyp = im_frame_np.sum() > 0


            labeled_frame, max_polyp = label(im_frame)

            if is_polyp and first_polyp == -1:
                first_polyp = polyp_n

            frame_output = file.loc[file[0] == polyp_n]
            if has_confidence:
                frame_output = frame_output.loc[frame_output[4] >= threshold]

            if frame_output.empty:
                if is_polyp:
                    fn += 1
                else:
                    tn += 1
            else:
                already_detected = []

                for detection in frame_output.iterrows():
                    detection = detection[1]
                    frame_pred = True
                    centroid_x = int(detection[2])
                    centroid_y = int(detection[1])

                    if frame_pred:
                        if is_polyp:
                            print(im_frame_np.shape)
                            if im_frame_np[centroid_x, centroid_y] == 1:
                                print(labeled_frame[centroid_x, centroid_y])
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

        print(vid_name, [tp, fp, fn, tn, rt], len(sorted(os.listdir(vid_folder))), sum([tp, fp, fn, tn, rt]))


if __name__ == '__main__':
    detection_csv = pd.DataFrame(
        columns=["TP", "FP", "FN", "TN", "RT", "Acc", "Pre", "Rec", "F_score", "Reaction_time"])
    for threshold in range(0, 10):
        # polyp_detection_results("/home/yael/Downloads/challenge_CVC_2/challenge/Detection/CVC2/",
        #                         "/home/yael/Downloads/challenge_CVC_2/challenge/res_yael/Detection/CVC2/",
        #                         "/home/yael/Downloads/PolypDetectionTest/GT", threshold=threshold / 10)

        polyp_localization_challenge("/home/yael/Downloads/challenge_CVC_2/challenge/Localization/CVC2/",
                                "/home/yael/Downloads/challenge_CVC_2/challenge/res_yael/Localization/CVC2/",
                                "/home/yael/Downloads/PolypDetectionTest/GT", threshold=threshold / 10)