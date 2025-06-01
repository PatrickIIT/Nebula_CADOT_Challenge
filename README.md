# Nebula_CADOT_Challenge
IEEE ICIP 2025 CADOT Challenge Submission

## Overview
This repository contains our solution for the IEEE ICIP 2025 CADOT Challenge, detecting 14 urban object classes in aerial images. We implemented a YOLOv11n model trained on the CADOT dataset, achieving an mAP50 of 0.335 on the validation set. The pipeline includes data preprocessing, training, inference, and evaluation scripts.

## Dataset
- **CADOT Dataset**: Aerial images with 14 classes (basketball field, building, crosswalk, football field, graveyard, large vehicle, medium vehicle, playground, roundabout, ship, small vehicle, swimming pool, tennis court, train).
- **Splits**: 3,234 training images, 929 validation images, ~465 test images.
- **Annotations**: COCO format, converted to YOLO format for training.

## Requirements
- **Hardware**: CPU (Intel Xeon 2.00GHz tested), 16–32 GB RAM
- **OS**: Ubuntu 20.04 or compatible
- **Python**: 3.11
- **Dependencies**: See `requirements.txt`



## Results
- YOLOv11n: mAP50 = [NEW_mAP50] (use the former weight and train for more 10 epochs, CPU)
- YOLOv5x: mAP50 = 0.03 (trained for 8 epochs)
- Test score: mAP50 = [NEW_TEST_mAP50]
- See `reports/report.pdf`.

## Submission
- Submit `/kaggle/working/submission_new.json` to CADOTools.
- Repository: https://github.com/PatrickIIT/Nebula_CADOT_Challenge

```bash
pip install -r requirements.txt
