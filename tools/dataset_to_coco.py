import os
from PIL import Image
import numpy as np

import datetime
import json

from cv2 import RETR_LIST
from pycocotools import mask
from skimage import measure
import cv2

import matplotlib.pyplot as plt


# ROOT_DIR = "C:\\Users\\Yael\\Desktop\\CVC-VideoClinicDBtrain_valid"
# ROOT_DIR = "/home/devsodin/Downloads/MaskRCNN/CVC-VideoClinicDBtrain_valid"
from torch._C import dtype

#ROOT_DIR = "datasets/CVC-VideoClinicDBtrain_valid"
ROOT_DIR = "datasets/cvcvideoclinicdbtest"
MASK_EXTENSION = "_polyp"
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
MASKS_DIR = os.path.join(ROOT_DIR, "masks")

INFO = {
    "description": "CVC-CLINIC Dataset - Test",
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

CATEGORIES = [
    {
        'id': 1,
        'name': 'polyp',
        'supercategory': 'polyp',
    },

]


def generate_bbox(mask):
    """

    :rtype: list
    """
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


def get_mask_images(mask_file):
    def is_annot_from_image(file):
        return mask_file.split(".")[0] in file

    if not os.path.exists(MASKS_DIR):
        return []
    else:
        return filter(is_annot_from_image, os.listdir(MASKS_DIR))


def get_image_coco_info(image_id, filename, size, license="", cocourl="", flickurl="", datacaptured=""):
    info =  {
        "id": image_id,
        "license": license,
        "coco_url": cocourl,
        "flickr_url": flickurl,
        "width": size[0],
        "height": size[1],
        "file_name": filename,
        "date_captured": datacaptured
    }
    print(info)
    return info


def get_annotations_coco_image(segmentation_id, image_id, category_info, binary_mask):
    bbox = generate_bbox(binary_mask)
    contours,_ = cv2.findContours(binary_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    segmentation = []
    for seg in contours:
        el_seg = []
        seg = np.unique(seg,axis=0)
        for dot in seg:
            dot = dot.astype(int)
            el_seg += dot.tolist()[0]
        segmentation.append(el_seg)


    area = mask.area(mask.encode(np.asfortranarray(binary_mask.astype(np.uint8))))
    iscrowd = 1 if category_info["is_crowd"] else 0

    if bbox is not None and area > 1 and segmentation is not None:
        bbox = bbox.tolist()

        annot =  {
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


def dataset_to_coco(root_folder, info, licenses, categories, seqs, out_name, has_annotations=True):


    coco_output = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    if not has_annotations:
        coco_output.pop("annotations")

    images_folder = os.path.join(root_folder, "images")
    masks_folder = os.path.join(root_folder, "masks")
    image_id = 1
    segmentation_id = 1

    for file in os.listdir(images_folder):
        if seqs is not None:
            if file[0:3] not in seqs:
                continue
        im_file = os.path.join(images_folder, file)
        im = Image.open(im_file)

        im_info = get_image_coco_info(image_id, os.path.basename(file), im.size)
        coco_output["images"].append(im_info)

        masks = get_mask_images(file)

        for mask in masks:
            mask_file = os.path.join(masks_folder, mask)

            class_id = [x['id'] for x in CATEGORIES if x['name'] in mask_file][0]
            category_info = {'id': class_id, 'is_crowd': 'crowd' in mask_file}
            binary_mask = np.asarray(Image.open(mask_file).convert('1')).astype(np.uint8)

            annot_info = get_annotations_coco_image(segmentation_id, image_id, category_info, binary_mask)

            if annot_info is not None:

                coco_output["annotations"].append(annot_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open(os.path.join(ROOT_DIR,"annotations", out_name), "w") as f:
        json.dump(coco_output, f)


def rename_images():
    for im in os.listdir(MASKS_DIR):
        extension = im.split(".")[1]

        # CVC train-val
        # seq = int(im.split("-")[0])
        # im_number = int(im.split(".")[0].split("-")[1])
        # new_name = "{:03d}-{:04d}{}.{}".format(seq,im_number,MASK_EXTENSION,extension)

        # CVC test
        # NO HAY MASCARAS

        # ETIS-Larib
        # im_number = int(im.split(".")[0][1:])
        # new_name = "{:03d}{}.{}".format(im_number,MASK_EXTENSION, extension)

        # os.rename(os.path.join(ANNOTATIONS_DIR, im),os.path.join(ANNOTATIONS_DIR, new_name))

    for im in os.listdir(IMAGES_DIR):
        extension = im.split(".")[1]

        # CVC train-val
        # CVC test
        seq = int(im.split("-")[0])
        im_number = int(im.split(".")[0].split("-")[1])
        new_name = "{:03d}-{:04d}.{}".format(seq, im_number, extension)

        # ETIS-Larib
        # im_number = int(im.split(".")[0])
        # new_name = "{:03d}.{}".format(im_number,extension)

        # os.rename(os.path.join(IMAGES_DIR, im), os.path.join(IMAGES_DIR, new_name))


if __name__ == '__main__':
    train_seq = ["{:03d}".format(x) for x in range(1,16)]
    val_seq= ["{:03d}".format(x) for x in range(16, 19)]
    #dataset_to_coco(ROOT_DIR, INFO, LICENSES, CATEGORIES, train_seq, "train.json")
    #dataset_to_coco(ROOT_DIR, INFO, LICENSES, CATEGORIES, val_seq, "val.json")
    dataset_to_coco(ROOT_DIR, INFO, LICENSES, CATEGORIES, None, "test.json", has_annotations=False)