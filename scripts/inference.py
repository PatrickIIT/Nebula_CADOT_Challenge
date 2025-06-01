%%writefile /kaggle/working/scripts/inference.py
#!/usr/bin/env python3

import torch
import os
import json
import sys
import argparse
from pathlib import Path

# Add YOLOv5 directory to sys.path
sys.path.append('/kaggle/working/yolov5')

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes

try:
    from utils.datasets import LoadImages
except ImportError:
    from utils.dataloaders import LoadImages

def load_image_ids(image_ids_file, dataset_dir=None):
    """Load image IDs from JSON or annotations file."""
    with open(image_ids_file, 'r') as f:
        data = json.load(f)
    
    print("JSON data structure:", data)
    
    # Case 1: Dictionary with 'images' key (e.g., image_ids.json)
    if isinstance(data, dict) and 'images' in data:
        if all(isinstance(item, dict) and 'file_name' in item and 'id' in item for item in data['images']):
            return {item['file_name']: item['id'] for item in data['images']}
        else:
            raise ValueError("JSON 'images' list must contain dictionaries with 'file_name' and 'id' keys")
    
    # Case 2: List of dictionaries with 'image_name' and 'image_id'
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if all('image_name' in item and 'image_id' in item for item in data):
            return {item['image_name']: item['image_id'] for item in data}
        else:
            raise ValueError("JSON items must contain 'image_name' and 'image_id' keys")
    
    # Case 3: List of strings (image names)
    elif isinstance(data, list) and all(isinstance(item, str) for item in data):
        return {image_name: idx + 1 for idx, image_name in enumerate(data)}
    
    # Case 4: Fallback to annotations file if dataset_dir is provided
    elif dataset_dir and os.path.exists(os.path.join(dataset_dir, '_annotations.coco.json')):
        print(f"image_ids.json does not match dataset, trying {dataset_dir}/_annotations.coco.json")
        with open(os.path.join(dataset_dir, '_annotations.coco.json'), 'r') as f:
            ann_data = json.load(f)
        if 'images' in ann_data:
            return {item['file_name']: item['id'] for item in ann_data['images']}
        else:
            raise ValueError("Annotations file must contain 'images' key with 'file_name' and 'id'")
    
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}. Expected a dictionary with 'images' key, a list of dictionaries, or a list of strings.")

def run_inference(model_path, test_images_dir, image_ids_file, output_json, 
                 img_size=640, conf_thres=0.25, iou_thres=0.45):
    """Generate COCO-format predictions for test set."""
    
    # Setup device
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = attempt_load(model_path, device=device)
    model.eval()
    
    # Load image IDs mapping
    print(f"Loading image IDs from: {image_ids_file}")
    image_ids = load_image_ids(image_ids_file, test_images_dir)
    
    # Create dataset
    print(f"Loading images from: {test_images_dir}")
    dataset = LoadImages(test_images_dir, img_size=img_size)
    
    predictions = []
    processed_count = 0
    
    print("Starting inference...")
    for data in dataset:
        print(f"Dataset output: {len(data)} elements - {data}")
        if len(data) == 4:
            path, img, im0s, vid_cap = data
        elif len(data) == 5:
            path, img, im0s, vid_cap, _ = data
        else:
            raise ValueError(f"Unexpected number of values returned by LoadImages: {len(data)}")
        
        img_name = os.path.basename(path)
        image_id = image_ids.get(img_name)
        
        if image_id is None:
            print(f"Warning: No image_id found for {img_name}")
            continue
        
        # Prepare image tensor
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = model(img, augment=True)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    x_min, y_min, x_max, y_max = [x.item() for x in xyxy]
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    predictions.append({
                        'image_id': image_id,
                        'category_id': int(cls.item()) + 1,
                        'bbox': [x_min, y_min, w, h],
                        'score': conf.item()
                    })
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
    
    # Save predictions
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Inference complete! Processed {processed_count} images")
    print(f"Generated {len(predictions)} predictions")
    print(f"Results saved to: {output_json}")
    
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 Inference for COCO Detection')
    parser.add_argument('--model_path', type=str, 
                       default='/kaggle/input/yolov5x/yolov5x.pt',
                       help='Path to YOLOv5 model')
    parser.add_argument('--test_images_dir', type=str,
                       default='/kaggle/input/cadot-paris/CADOT_Dataset/test',
                       help='Directory containing test images')
    parser.add_argument('--image_ids_file', type=str,
                       default='/kaggle/input/cadot-paris/image_ids.json',
                       help='JSON file with image ID mappings')
    parser.add_argument('--output_json', type=str,
                       default='/kaggle/working/results/predictions.json',
                       help='Output JSON file for predictions')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Inference image size')
    parser.add_argument('--conf_thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        test_images_dir=args.test_images_dir,
        image_ids_file=args.image_ids_file,
        output_json=args.output_json,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
