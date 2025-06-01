# Nebula_CADOT_Challenge
IEEE ICIP 2025 CADOT Challenge Submission

## Overview
YOLOv5m-based solution for the IEEE ICIP 2025 CADOT Challenge, using copy-paste augmentation and focal loss.

## Requirements
- Hardware: NVIDIA GPU (16–24 GB VRAM), 64 GB RAM
- OS: Ubuntu 20.04
- Python: 3.9
- CUDA: 11.8
- Dependencies: `requirements.txt`

## Setup
```bash
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
!pip install torch==2.1.0 pycocotools==2.0.7 pandas==2.2.2 numpy==1.26.4 opencv-python==4.10.0.84
