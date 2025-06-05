# Ceramic Defect Detection

This repository provides a simple pipeline for training and running a vision model to detect defects in sanitary ceramics. It uses PyTorch and Torchvision with a Faster R-CNN model. The code is designed to be deployable via Docker and can be integrated into other systems (such as Dify) by exposing command line or API calls.

## Features

- Dataset loader for COCO-format annotations (`defect_detection/dataset.py`).
- Training script (`defect_detection/train.py`).
- Prediction script with knowledge base lookup (`defect_detection/predict.py`).
- Example knowledge base mapping defect categories to repair instructions (`defect_detection/knowledge_base.json`).
- Minimal Dockerfile for running the scripts.

## Quick Start

Build the docker image:

```bash
docker build -t ceramic-defect .
```

Train a model (assuming images and annotations are mounted in the container):

```bash
docker run -v /path/to/data:/data ceramic-defect \
    python defect_detection/train.py \
    --images /data/images \
    --annotations /data/annotations.json \
    --num-classes 4 \
    --output /data/model.pth
```

Run prediction on a single image:

```bash
docker run -v /path/to/data:/data ceramic-defect \
    python defect_detection/predict.py \
    --image /data/test.jpg \
    --weights /data/model.pth \
    --kb defect_detection/knowledge_base.json \
    --output /data/result.jpg
```

The prediction script prints a JSON list of detected defects with repair suggestions and saves an annotated image.

## Integration with Labeling Tools

The training script expects annotations in COCO format. Popular tools such as [CVAT](https://github.com/opencv/cvat) or [LabelMe](https://github.com/wkentaro/labelme) can export to this format, enabling manual data labeling.

## Model Flexibility

`train.py` uses Faster R-CNN, but the `get_model` function can be replaced with any Torchvision or custom model. This design allows swapping models without rewriting the training loop.

## Using with Dify

The Docker image runs standard Python commands. To integrate with Dify's Docker deployment, include this repository or a built image in your Dify `docker-compose` setup and call the prediction script from within your workflow.
