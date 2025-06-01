import torch
from ultralytics import YOLO
import os
import json
import argparse
import pandas as pd

def load_image_ids(image_ids_file):
    """Load image IDs from provided CSV."""
    df = pd.read_csv(image_ids_file)
    return dict(zip(df['image_name'], df['image_id']))

def run_inference(model_path, test_images_dir, image_ids_file, output_json, img_size=640):
    """Generate COCO-format predictions for test set."""
    model = YOLO(model_path)
    image_ids = load_image_ids(image_ids_file)
    
    predictions = []
    for img_name in sorted(os.listdir(test_images_dir)):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(test_images_dir, img_name)
            results = model.predict(img_path, imgsz=img_size, conf=0.25, iou=0.45, augment=True)  # TTA
            image_id = image_ids.get(img_name)
            if image_id is None:
                print(f"Warning: No image_id for {img_name}")
                continue
            for det in results[0].boxes:
                x, y, w, h = det.xywh[0].tolist()
                score = det.conf.item()
                category_id = int(det.cls.item()) + 1  # 1-based for COCO
                predictions.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x - w/2, y - h/2, w, h],  # [x_min, y_min, w, h]
                    'score': score
                })
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=2)
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_images_dir', type=str, default='/kaggle/input/cadot-paris/CADOT_Dataset/test/images', help='Path to test images')
    parser.add_argument('--image_ids_file', type=str, default='/kaggle/input/cadot-paris/image_ids.csv', help='Path to image IDs CSV')
    parser.add_argument('--output_json', type=str, default='results/predictions.json', help='Output JSON')
    parser.add_argument('--img_size', type=int, default=640, help='Inference image size')
    args = parser.parse_args()

    predictions = run_inference(args.model_path, args.test_images_dir, args.image_ids_file, args.output_json, args.img_size)
