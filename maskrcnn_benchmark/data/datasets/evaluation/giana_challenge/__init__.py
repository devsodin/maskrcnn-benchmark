import os

from .giana_eval import *


def giana_eval(dataset, predictions, output_folder, **_):
    dataset_root_folder = "/".join(dataset.root.split("/")[:-1])
    folder_detection = os.path.join(output_folder, "detection")
    folder_localization = os.path.join(output_folder, "localization")
    giana_results_folder = os.path.join(output_folder, "giana_results")
    folder_gt = os.path.join(dataset_root_folder, "masks")
    do_giana_eval(output_folder, folder_detection, folder_localization, folder_gt, giana_results_folder,
                  dataset.annotation_file, dataset.name)

    return
