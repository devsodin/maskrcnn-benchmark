import os
import numpy as np
import pandas as pd
from PIL import Image


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

        fn, fp, rt, tn, tp = process_video(file, has_confidence, threshold,
                                           vid_folder)

        res_dict[vid_name] = [tp, fp, fn, tn, rt]
        print("output: {}, total ims: {}, gt ims: {}".format([tp, fp, fn, tn, rt], sum([tp, fp, fn, tn]), len(sorted(os.listdir(vid_folder)))))

    pd.DataFrame.from_dict(res_dict, orient="index", columns=["TP", "FP", "FN", "TN", "RT"]).to_csv(
        output_folder + "/res{}.csv".format(threshold), index=False)


def process_video(file, has_confidence, threshold, vid_folder):
    first_polyp = -1
    first_detected_polyp = -1

    tp, fp, fn, tn = 0, 0, 0, 0
    for frame in sorted(os.listdir(vid_folder)):

        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        is_polyp = np.asarray(im_frame).sum() > 0

        if is_polyp and first_polyp == -1:
            first_polyp = polyp_n

        frame_output = file.loc[file[0] == polyp_n]
        if has_confidence:
            frame_output = frame_output.loc[frame_output[2] >= threshold]

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

    rt = first_detected_polyp - first_polyp if first_detected_polyp != -1 else -1
    return fn, fp, rt, tn, tp


if __name__ == '__main__':
    for threshold in range(0, 10):
        polyp_detection_results("../out/test_cj/inference/cvc-clinic-test/detection",
                                "../out/test_cj/inference/cvc-clinic-test/res_detect",
                                "/home/yael/Downloads/PolypDetectionTest/GT", threshold=threshold / 10)
