import os

import torch
from PIL import Image
from torchvision.datasets.coco import CocoDetection

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False
    return True


class CVCClinicDataset(CocoDetection):
    def __init__(self, annotation_file, root, remove_images_without_annotations, name, transforms=None):
        super(CVCClinicDataset, self).__init__(root, annotation_file)
        self.root = root
        self.name = name
        self.annotation_file = annotation_file

        files = sorted([v['file_name'] for key, v in self.coco.imgs.items()])
        self.ids = []
        for f in files:
            for k, fn in self.coco.imgs.items():
                if fn['file_name'] == f:
                    self.ids.append(k)

        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        orig_image = image
        # orig_image = image.convert("L")

        # filter crowd annotations
        # TODO might be better to add an extra field
        annotation = [obj for obj in target if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in annotation]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in annotation]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if annotation and "segmentation" in annotation[0]:
            masks = [obj["segmentation"] for obj in annotation]
            masks = SegmentationMask(masks, image.size, mode='poly')
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target, idx, orig_image

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__ == '__main__':
    test = CVCClinicDataset("../../../datasets/CVC-classification/annotations/train.json",
                            "../../../datasets/CVC-classification/images", False, None)
    print(test[0])

    batch = test[532]

    print(batch)
    import numpy as np
    from matplotlib import pyplot as plt

    plt.imshow(batch[3])
    print(np.array(batch[3]).std())
    plt.show()

    a = np.array(batch[3].convert("YCbCr"))[:,:,0]
    plt.imshow(a)
    plt.title("YCbCr- Y")
    plt.show()

    print("aaa")
