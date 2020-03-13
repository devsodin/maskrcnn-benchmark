# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomGaussianBlur(object):
    def __init__(self, kernel_size=[], prob=0.5):
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, image, target):
        if random.random() < self.prob:
            kernel_size = random.choice(self.kernel_size)
            image = image.filter(ImageFilter.GaussianBlur(kernel_size))

        return image, target


class RandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, prob=0.5):
        self.prob = prob
        if isinstance(degrees, int):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, int):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                       (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] / 100 * img_size[0]
            max_dy = translate[1] / 100 * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, target):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img = np.array(img)

        if random.random() < self.prob:
            angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear,
                                                                img.shape)

            R = np.eye(3)
            center_y, center_x = img.shape[0] / 2, img.shape[1] / 2
            R[:2] = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)

            T = np.eye(3)
            T[0, 2] = img.shape[0] * translations[0]
            T[1, 2] = img.shape[1] * translations[1]

            S = np.eye(3)
            import math
            S[0, 1] = math.tan(shear[0] * math.pi / 180)
            S[1, 0] = math.tan(shear[1] * math.pi / 180)

            M = S @ T @ R
            if (M != np.eye(3)).any():
                # assuming that the first pixel is always part of endoscopy mask
                bg_color_image = img[0, 0, :].tolist()
                img = cv2.warpAffine(img, M[:2], img.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=bg_color_image)


            transformed_points = []
            for box in target.bbox:
                x, y, x2, y2 = box.numpy()
                point = np.array(((x, y), (x2, y2), (x, y2)))[None, ...]
                transformed_points.append(cv2.perspectiveTransform(point, M))

            new_boxes = []
            for box in transformed_points:
                box = box.squeeze()
                xy = box[0]
                x2y2 = box[1]

                new_boxes.append([xy[0], xy[1], x2y2[0], x2y2[1]])

            target.bbox = torch.tensor(new_boxes)
            target.clip_to_image()

        return Image.fromarray(img), target


class RandomGaussianNoise(object):
    def __init__(self, prob=0.5, sigma=0.2):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image + torch.rand_like(image) * self.sigma
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic, target):
        return F.to_pil_image(pic, self.mode), target


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from maskrcnn_benchmark.data.datasets import CVCClinicDataset


    def scatter_points(bbox):
        bbox = bbox
        plt.plot([bbox[0], bbox[0], bbox[2], bbox[2]],
                 [bbox[1], bbox[3], bbox[1], bbox[3]])


    ds = CVCClinicDataset("datasets/CVC-VideoClinicDBtrain_valid/annotations/train.json",
                          "datasets/CVC-VideoClinicDBtrain_valid/images/",
                          False, "colon612",
                          transforms=Compose(
                              [RandomAffine(prob=1., degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1),
                                            shear=(2, 2))]))

    ds_no = CVCClinicDataset("datasets/cvc-colondb-612/annotations/train.json", "datasets/cvc-colondb-612/images/",
                             False, "colon612")

    # sample = 261
    # b = ds[sample]
    for i in range(len(ds)):
        b = ds[i]
        print(b)

        for box in b[1].bbox:
            scatter_points(box)

        print(b[1].bbox)
        plt.imshow(b[0])
        plt.show()

    # a = ds[sample]
    #
    # for box in a[1].bbox:
    #     scatter_points(box)
    #
    # print(a[1].bbox)
    # plt.imshow(a[0])
    # plt.show()
