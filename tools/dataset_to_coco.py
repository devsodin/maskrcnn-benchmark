import os
from PIL import Image
import numpy as np
import pycococreatortools
import datetime


#ROOT_DIR = "C:\\Users\\Yael\\Desktop\\CVC-VideoClinicDBtrain_valid"
ROOT_DIR = "C:\\Users\\Yael\\Desktop\\ETIS-LaribPolypDB"
IMAGES_DIR = os.path.join(ROOT_DIR,"images")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR,"masks")
MASK_EXTENSION = "_polyp"


INFO = {
    "description": "CVC-CLINIC Dataset",
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

CATEGORIES = [
    {
        'id': 1,
        'name': 'polyp',
        'supercategory': 'polyp',
    },

]

def bbox_from_mask(mask_file):
    pass

def generate_bboxes(mask_folder):
    for file in (os.path.join(mask_folder, file) for file in os.listdir(mask_folder)):

        im = Image.open(file)


        coors = np.where(np.array(im) == 1)

        if len(coors[0]) == 0:
            pass
        else:
            x_min = np.min(coors[0])
            x_max = np.max(coors[0])
            y_min = np.min(coors[1])
            y_max = np.max(coors[1])
            print(x_min,x_max, y_min,y_max)


def get_mask_images(mask_file):

    def is_annot_from_image(file):
        return mask_file.split(".")[0] in file

    return filter(is_annot_from_image, os.listdir(ANNOTATIONS_DIR))


def cocoize_dataset():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1


    for file in os.listdir(IMAGES_DIR):
        im_file = os.path.join(IMAGES_DIR, file)
        im = Image.open(im_file)

        im_info = pycococreatortools.create_image_info(image_id, os.path.basename(file), im.size)
        coco_output["images"].append(im_info)

        masks = get_mask_images(file)

        for mask in masks:
            mask_file = os.path.join(ANNOTATIONS_DIR, mask)

            class_id = [x['id'] for x in CATEGORIES if x['name'] in mask_file][0]
            category_info = {'id': class_id, 'is_crowd': 'crowd' in mask_file}
            binary_mask = np.asarray(Image.open(mask_file).convert('1')).astype(np.uint8)

            annot_info = pycococreatortools.create_annotation_info(segmentation_id, image_id, category_info, binary_mask, im.size, tolerance=1)

            if annot_info is not None:
                coco_output["annotations"].append(annot_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    print(coco_output)


def rename_images():
    for im in os.listdir(ANNOTATIONS_DIR):
        extension = im.split(".")[1]

        # CVC train-val
        #seq = int(im.split("-")[0])
        #im_number = int(im.split(".")[0].split("-")[1])
        #new_name = "{:03d}-{:04d}{}.{}".format(seq,im_number,MASK_EXTENSION,extension)

        # CVC test
        # NO HAY MASCARAS

        # ETIS-Larib
        #im_number = int(im.split(".")[0][1:])
        #new_name = "{:03d}{}.{}".format(im_number,MASK_EXTENSION, extension)

        #os.rename(os.path.join(ANNOTATIONS_DIR, im),os.path.join(ANNOTATIONS_DIR, new_name))

    for im in os.listdir(IMAGES_DIR):
        extension = im.split(".")[1]

        # CVC train-val
        # CVC test
        seq = int(im.split("-")[0])
        im_number = int(im.split(".")[0].split("-")[1])
        new_name = "{:03d}-{:04d}.{}".format(seq,im_number,extension)



        # ETIS-Larib
        #im_number = int(im.split(".")[0])
        #new_name = "{:03d}.{}".format(im_number,extension)

        os.rename(os.path.join(IMAGES_DIR, im),os.path.join(IMAGES_DIR, new_name))



if __name__ == '__main__':

    dataset_folder = "C:\\Users\\Yael\\Desktop\\CVC-VideoClinicDBtrain_valid"

    mask_folder = os.path.join(dataset_folder, "Masks")

    rename_images()

    #cocoize_dataset()

