import os
import cv2
import numpy as np

def bbox_from_mask(mask_file):
    pass

def generate_bboxes(mask_folder):
    for file in (os.path.join(mask_folder, file) for file in os.listdir(mask_folder)):

        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        print(im.shape)
        coors = np.where(im == 255)
        print(coors)
        pass



if __name__ == '__main__':

    dataset_folder = "/home/devsodin/Downloads/CVC-VideoClinicDBtrain_valid"
    mask_folder = os.path.join(dataset_folder,"Masks")

    generate_bboxes(mask_folder)