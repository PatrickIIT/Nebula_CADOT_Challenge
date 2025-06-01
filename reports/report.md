# CADOT Challenge Report

## Introduction
This report details our approach for the IEEE ICIP 2025 CADOT Challenge, detecting 14 urban object classes in aerial images using YOLOv11n.

## Methodology
- **Data Preprocessing**: Converted COCO annotations to YOLO format (3,234 train, 929 valid images).
- **Model**: YOLOv11n, trained for 8 epochs on CPU.
- **Training**: Batch=8, imgsz=512, patience=5.
- **Evaluation**: mAP50=0.335 on validation set.

## Results
- **mAP50**: 0.335
- **Per-Class AP**:
  - Crosswalk: 0.814
  - Football Field: 0.870
  - Swimming Pool: 0.857
  - Playground: 0.654
  - Building, Roundabout, Ship, Train: 0.000
- **YOLOv5x Baseline**: mAP50=0.000 (pre-trained, no training).

## Analysis
- Strong performance on frequent classes (e.g., crosswalk, swimming pool).
- Poor performance on rare classes (e.g., building, ship) due to limited instances and epochs.
- CPU training limited convergence.

## Future Work
- Train for 20–50 epochs.
- Use YOLOv11x or YOLOv8x.
- Apply copy-paste augmentation for rare classes.
- Use GPU for faster training.

## Conclusion
Our YOLOv11n model achieved mAP50: 0.335, competitive but below the top 10 (>55.51). Future improvements will focus on rare classes and GPU training." > /kaggle/working/Nebula_CADOT_Challenge/reports/report.md
pandoc reports/report.md -o reports/report.pdf
git add reports/report.md reports/report.pdf
git commit -m "Update report.md and add report.pdf"
git push origin main
