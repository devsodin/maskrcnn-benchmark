import torch
from torchvision.datasets.coco import CocoDetection

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

min_keypoints_per_image = 10


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False

    return False


class CVCClinicDataset(CocoDetection):
    def __init__(self, annotation_file, root, remove_unannotated_images, transforms=None):
        super(CVCClinicDataset, self).__init__(root, annotation_file)

        self.ids = sorted(self.ids)

        if remove_unannotated_images:
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