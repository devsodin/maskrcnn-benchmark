import datetime
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools import mask
from scipy.misc import imsave
from scipy.ndimage import label

ROOT_DIR_CVC_TRAIN_VAL = "../datasets/CVC-VideoClinicDBtrain_valid"
ROOT_DIR_CVC_TEST = "../datasets/cvcvideoclinicdbtest"
ROOT_DIR_CVC_CLASSIFICATION = "../datasets/CVC-classification"
ROOT_DIR_ETIS = "../datasets/ETIS-LaribPolypDB"
ROOT_DIR_CVC_612 = "../datasets/cvc-colondb-612"
ROOT_DIR_CVC_300 = "../datasets/cvc-colondb-300"

MASK_EXTENSION = "_Polyp"
SPECULAR_EXTENSION = "_Specular"

INFO_CVC = {
    "description": "CVC-clinic",
    "url": "",
    "version": "1.0",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

INFO_ETIS = {
    "description": "Etis-Train",
    "url": "",
    "version": "1.0",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

# ANNOTATIONS
# id, category id, iscrowd, segmentation, image_id, area, bbox(x,y,w,h)

CATEGORIES_MULTI = [
    {
        'id': 1,
        'name': 'polyp',
        'supercategory': 'polyp',
    },
    {
        'id': 2,
        'name': 'AD',
        'supercategory': 'polyp',
    },
    {
        'id': 3,
        'name': 'ASS',
        'supercategory': 'polyp',
    },
    {
        'id': 4,
        'name': 'HP',
        'supercategory': 'polyp',
    },
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'Polyp',
        'supercategory': 'polyp',
    },

    {
        'id': 2,
        'name': 'Specular',
        'supercategory': 'other',
    },
]


def generate_bbox(mask):
    coors = np.where(np.array(mask) == 1)

    if len(coors[0]) == 0:
        return None
    else:
        x_min = np.min(coors[1])
        x_max = np.max(coors[1])
        y_min = np.min(coors[0])
        y_max = np.max(coors[0])
        w = x_max - x_min
        h = y_max - y_min

        return np.array([x_min, y_min, w, h])


def get_mask_images(mask_dir, mask_file):
    def is_annot_from_image(file):
        return mask_file.split(".")[0] in file

    if not os.path.exists(mask_dir):
        return []
    else:
        base, extension = mask_file.split(".")

        masks = glob(os.path.join(mask_dir, "{}*".format(base)))

        for file in masks:
            r_im = Image.open(file).convert('L')
            r_im = np.array(r_im)
            kernel = np.ones((3, 3))
            # kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])

            im, count = label(r_im, structure=kernel)
            print("masks:", count, os.path.join(mask_dir, file))
            # split_images_masks(file)
        return filter(is_annot_from_image, os.listdir(mask_dir))


def get_image_coco_info(image_id, filename, size, license="", cocourl="", flickurl="", datacaptured=""):
    info = {
        "id": image_id,
        "license": license,
        "coco_url": cocourl,
        "flickr_url": flickurl,
        "width": size[0],
        "height": size[1],
        "file_name": filename,
        "date_captured": datacaptured
    }
    return info


def get_annotations_coco_image(segmentation_id, image_id, category_info, binary_mask):
    bbox = generate_bbox(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    segmentation = list()
    for seg in contours:
        el_seg = []
        for dot in seg:
            dot = dot.astype(int)
            el_seg += dot[0].astype(int).tolist()
        segmentation.append(el_seg)

    area = mask.area(mask.encode(np.asfortranarray(binary_mask.astype(np.uint8))))
    iscrowd = 1 if category_info["is_crowd"] else 0

    if bbox is not None and area > 1 and segmentation is not None:
        bbox = bbox.tolist()

        annot = {
            # id, category id, iscrowd, segmentation, image_id, area, bbox(x,y,w,h)

            "segmentation": segmentation,
            "iscrowd": iscrowd,
            "area": float(area),
            "image_id": image_id,
            "bbox": bbox,
            "category_id": category_info['id'],
            "id": segmentation_id
        }
        print(annot)

        return annot

    else:
        return None


def dataset_to_coco(root_folder: str, info, licenses, categories, seqs: list or None, out_name: str,
                    has_annotations: bool = True) -> None:
    coco_output = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    images_folder = os.path.join(root_folder, "images")
    masks_folder = os.path.join(root_folder, "masks")
    image_id = 1
    segmentation_id = 1

    if not has_annotations:
        coco_output.pop("annotations")

    folder_ims = os.listdir(images_folder)

    for file in folder_ims:

        if seqs is not None:
            seq_name = file[0:3]
            if seq_name not in seqs:
                continue

        im_file = os.path.join(images_folder, file)
        im = Image.open(im_file)

        im_info = get_image_coco_info(image_id, os.path.basename(file), im.size)
        coco_output["images"].append(im_info)

        # if dataset or image have zero masks, return empty list
        masks = get_mask_images(masks_folder, file)

        for mask in masks:
            mask_file = os.path.join(masks_folder, mask)
            print(mask_file)
            class_id = [category['id'] for category in categories if category['name'] in mask.split("_")][0]

            # TODO should change is_crowd (need to update data.csv adding is_crowd data)
            category_info = {'id': class_id, 'is_crowd': 'crowd' in mask_file}
            binary_mask = np.asarray(Image.open(mask_file).convert('1')).astype(np.uint8)

            annot_info = get_annotations_coco_image(segmentation_id, image_id, category_info, binary_mask)

            if annot_info is not None:
                coco_output["annotations"].append(annot_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open(os.path.join(root_folder, "annotations", out_name), "w") as f:
        json.dump(coco_output, f)


def rename_images(root_dir):
    # TODO Refactor per dataset

    images_dir = os.path.join(root_dir, "images")
    MASKS_DIR = os.path.join(root_dir, "masks")

    for im in os.listdir(MASKS_DIR):
        # CVC train-val
        # seq = int(im.split("-")[0])
        # im_number = int(im.split(".")[0].split("-")[1])
        # new_name = "{:03d}-{:04d}{}.{}".format(seq, im_number, MASK_EXTENSION, extension)

        # CVC test
        # NO HAY MASCARAS

        # COLONDB
        im_number, extension = im.split(".")
        im_number = int(im_number)
        new_name = "{:03d}{}.{}".format(im_number, MASK_EXTENSION, extension)

        # ETIS-Larib
        # im_number = int(im.split(".")[0][1:])
        # new_name = "{:03d}{}.{}".format(im_number, MASK_EXTENSION, extension)
        #

        os.rename(os.path.join(MASKS_DIR, im), os.path.join(MASKS_DIR, new_name))

    # for im in os.listdir(images_dir):
    #     extension = im.split(".")[1]
    #
    #     # CVC train-val
    #     # CVC test
    #     # seq = int(im.split("-")[0])
    #     # im_number = int(im.split(".")[0].split("-")[1])
    #     # new_name = "{:03d}-{:04d}.{}".format(seq, im_number, extension)
    #
    #     # ETIS-Larib
    #     # im_number = int(im.split(".")[0])
    #     # new_name = "{:03d}.{}".format(im_number, extension)
    #
    #     # COLONDB
    #     im_number, extension = im.split(".")
    #     im_number = int(im_number)
    #     new_name = "{:03d}.{}".format(im_number, extension)
    #
    #     os.rename(os.path.join(images_dir, im), os.path.join(images_dir, new_name))
    #
    if "colondb" in root_dir:
        REFLEX_DIR = os.path.join(root_dir, "specular")

        for im in os.listdir(REFLEX_DIR):
            im_number, extension = im.split(".")

            im_number = int(im_number)
            new_name = "{:03d}{}.{}".format(im_number, SPECULAR_EXTENSION, extension)

            os.rename(os.path.join(REFLEX_DIR, im), os.path.join(REFLEX_DIR, new_name))


def split_images_masks(image):
    name, ext = os.path.basename(image).split(".")
    ext = "tif"
    path = os.path.dirname(image)

    r_im = Image.open(image).convert('L')
    r_im = np.array(r_im)
    kernel = np.ones((3,3))
    # kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    if r_im.sum() != 0:
        im, count = label(r_im, structure=kernel)
        if count <= 1:
            imsave(os.path.join(path, "{}_{}.{}".format(name, 1, ext)), r_im)
        else:
            for cont in range(count):
                print(np.where(im == cont + 1, r_im, 0).sum() / 255)
                if np.where(im == cont + 1, r_im, 0).sum() / 255 >= 125.0:
                    imsave(os.path.join(path, "{}_{}.{}".format(name, cont + 1, ext)), np.where(im == cont + 1, r_im, 0))

    os.remove(image)


def _gen_csv_for_dataset():
    datasets_folders = [ROOT_DIR_CVC_CLASSIFICATION]
    file_extension = "png"
    mask_extension = "_Polyp"
    for dataset in datasets_folders:
        ims_folder = os.path.join(dataset, "images")
        mask_folder = os.path.join(dataset, "masks")

        df = pd.DataFrame(columns=['image', 'mask', 'has_polyp', 'classification'])
        gt = pd.read_csv(os.path.join(dataset, "annotations", "processed.csv"))

        for im in os.listdir(ims_folder):
            im_file = os.path.basename(im)

            mask, ext = im_file.split(".")
            mask_file = mask + mask_extension + ".{}".format(file_extension)

            mask_im = cv2.imread(os.path.join(mask_folder, mask_file))
            if mask_im.sum() > 0:
                index = gt.loc[gt["IMAGES"] == im_file, "Histologia"].index
                if len(gt.iloc[index].Histologia) > 0:
                    df.loc[-1] = [im_file, mask_file, 1, gt.iloc[index].Histologia.tolist()[0]]
                else:
                    continue

            else:
                df.loc[-1] = [im_file, mask_file, 0, 0]

            df.index += 1
            df.sort_index()
            df.reset_index(inplace=True, drop=True)

        df.sort_values(by='image')
        df.to_csv(os.path.join(dataset, "data.csv"), index=False)


if __name__ == '__main__':
    train_seq = ["{:03d}".format(x) for x in range(1, 16)]
    val_seq = ["{:03d}".format(x) for x in range(16, 19)]

    # dataset_to_coco(ROOT_DIR_CVC_TRAIN_VAL, INFO_CVC, LICENSES, CATEGORIES, train_seq, "train.json",
    #                 has_annotations=True)
    #
    # dataset_to_coco(ROOT_DIR_CVC_TRAIN_VAL, INFO_CVC, LICENSES, CATEGORIES, val_seq, "val.json",
    #                 has_annotations=True)

    # dataset_to_coco(ROOT_DIR_CVC_TEST, INFO_CVC, LICENSES, CATEGORIES, None, "test.json",
    #                 has_annotations=False)
    #
    # dataset_to_coco(ROOT_DIR_CVC_CLASSIFICATION, INFO_CVC, LICENSES, CATEGORIES, None, "train.json",
    #                 has_annotations=True)
    #
    # dataset_to_coco(ROOT_DIR_ETIS, INFO_ETIS, LICENSES, CATEGORIES, None, "train.json",
    #                 has_annotations=True)
    #
    dataset_to_coco(ROOT_DIR_CVC_300, INFO_CVC, LICENSES, CATEGORIES, None, "train.json",
                    has_annotations=True)
    #
    # dataset_to_coco(ROOT_DIR_CVC_612, INFO_CVC, LICENSES, CATEGORIES, None, "train.json",
    #                 has_annotations=True)
