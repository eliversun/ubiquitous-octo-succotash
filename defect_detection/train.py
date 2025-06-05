import argparse
import os
from dataset import CeramicDefectDataset
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CeramicDefectDataset(args.images, args.annotations)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = get_model(num_classes=args.num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {losses.item():.4f}")
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train defect detector')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--annotations', required=True, help='Path to COCO annotations JSON')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes including background')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', default='model.pth', help='Output model path')
    args = parser.parse_args()
    train(args)
