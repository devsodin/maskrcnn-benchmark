import torch
from torchvision.datasets.coco import CocoDetection

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class CVCClinicDataset(CocoDetection):
    def __init__(self, annotation_file, root, name, transforms=None):
        super(CVCClinicDataset, self).__init__(root, annotation_file)
        self.root = root
        self.name = name
        self.annotation_file = annotation_file

        self.ids = sorted(self.ids)

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
        image, annotation = super(CVCClinicDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        annotation = [obj for obj in annotation if obj["iscrowd"] == 0]

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

        return image, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__ == '__main__':
    test = CVCClinicDataset("../../../datasets/CVC-classification/annotations/train.json",
                            "../../../datasets/CVC-classification/images", False, None)
    print(test[0][1])
