import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import CeramicDefectDataset


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_knowledge_base(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kb = load_knowledge_base(args.kb)
    num_classes = len(kb['categories']) + 1
    model = get_model(num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    img = Image.open(args.image).convert('RGB')
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]
    draw = ImageDraw.Draw(img)
    results = []
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score < args.confidence:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        cat = kb['categories'].get(str(label.item()), f'class_{label.item()}')
        info = kb['repairs'].get(cat, {})
        results.append({'category': cat, 'score': float(score), 'repairable': info.get('repairable', False), 'how_to_repair': info.get('method', '')})
    output_image = Path(args.output)
    img.save(output_image)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict defects on an image')
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--weights', default='model.pth', help='Model weights path')
    parser.add_argument('--kb', default='knowledge_base.json', help='Knowledge base JSON')
    parser.add_argument('--confidence', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--output', default='result.jpg', help='Path to save annotated image')
    args = parser.parse_args()
    predict(args)
