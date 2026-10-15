# Desk Object Detection with YOLOv8n and Faster R-CNN

**Master's Course Project – Machine Vision (Maschinelles Sehen)**  
**M.Eng. Computer Engineering – Embedded Systems · BHT Berlin · 2026**  
**Course project completed with Mert Satici**

## Overview

This project investigates custom object detection for cluttered desk environments using two different detection architectures:

- **YOLOv8n** – one-stage object detection
- **Faster R-CNN** – two-stage region-based object detection

A custom dataset of desk scenes was created and manually annotated for five object classes:

- bottle
- cell phone
- cup
- laptop
- mouse

The project covers the complete workflow from dataset preparation and transfer learning to model evaluation and visual comparison.

## Project Pipeline

1. Collect and annotate custom desk images
2. Train multiple YOLOv8n versions using transfer learning
3. Convert YOLO-format annotations for Faster R-CNN
4. Train Faster R-CNN on the final dataset
5. Evaluate both approaches on the same task
6. Compare quantitative metrics and visual predictions

## Models

### YOLOv8n

YOLOv8n was fine-tuned using three dataset versions.

The final version used the clean 120-image dataset without brightness augmentation.

**Final YOLOv8n v3 results:**

- Precision: **0.890**
- Recall: **0.906**
- mAP50: **0.965**
- mAP50-95: **0.870**

### Faster R-CNN

Faster R-CNN was trained using the same final dataset to provide a two-stage detector for comparison.

YOLO annotations were converted from normalized bounding-box coordinates to absolute `xyxy` coordinates required by the PyTorch Faster R-CNN implementation.

## Key Technologies

- Python
- PyTorch
- Torchvision
- Ultralytics YOLOv8
- Faster R-CNN
- OpenCV
- NumPy
- Matplotlib
- Roboflow
- Transfer Learning
- Object Detection

## Key Findings

Dataset quality had a significant impact on detection performance.

Additional targeted images improved detection of smaller and more difficult objects such as mobile phones and mice. Removing brightness augmentation also reduced false positives in the final YOLO dataset.

YOLOv8n provided the stronger real-time candidate, while Faster R-CNN produced clean region-based predictions and served as a useful architectural comparison.

## Repository Structure

```text
data/        Dataset and sample images
docs/        Project documentation
models/      Model-related files
notebooks/   Experiments and analysis
results/     Evaluation outputs and prediction images
src/         Training, evaluation and comparison scripts
