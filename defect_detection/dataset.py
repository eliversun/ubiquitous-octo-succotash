import os
import json
from typing import List, Dict
from PIL import Image
import torch
from torchvision.transforms import functional as F

class CeramicDefectDataset(torch.utils.data.Dataset):
    """Dataset for ceramic defect detection in COCO format."""
    def __init__(self, images_dir: str, annotations_path: str, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = list(self.images.keys())[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        anns = self.image_to_anns.get(image_id, [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(ann['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([image_id])}
        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)
        return img, target
