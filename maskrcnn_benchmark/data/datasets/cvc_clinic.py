import os

import torch
from PIL import Image
from torchvision.datasets.coco import CocoDetection

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class CVCClinicDataset(CocoDetection):
    def __init__(self, annotation_file, root, name, transforms=None):
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

    batch = test[1152]

    print(batch)
    import numpy as np
    from matplotlib import pyplot as plt

    plt.imshow(batch[3])
    print(np.array(batch[3]).std())
    plt.show()
