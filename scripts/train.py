import torch
import yaml
import os
import argparse
from ultralytics import YOLO

def create_data_yaml(data_path):
    """Create data.yaml for CADOT dataset."""
    data = {
        'train': os.path.join(data_path, 'train'),
        'val': os.path.join(data_path, 'valid'),
        'test': os.path.join(data_path, 'test'),
        'nc': 14,
        'names': [
            'Basketball Field', 'Building', 'Crosswalk', 'Football Field', 'Graveyard',
            'Large Vehicle', 'Medium Vehicle', 'Playground', 'Roundabout', 'Ship',
            'Small Vehicle', 'Swimming Pool', 'Tennis Court', 'Train'
        ]
    }
    yaml_path = '/kaggle/working/data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def train_model(data_yaml, model_variant='/kaggle/input/yolov5x/yolov5x.pt', epochs=20, img_size=640, batch_size=16):
    """Train YOLOv5x on CADOT dataset."""
    model = YOLO(model_variant)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch_size=batch_size,
        device=0,
        workers=4,
        project='/kaggle/working/runs/train',
        name='cadot_exp',
        exist_ok=True,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        focal_loss=True,
        multi_scale=True,
        seed=42
    )
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/kaggle/input/cadot-paris/CADOT_Dataset/', help='Path to CADOT dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    data_yaml = create_data_yaml(args.data_path)
    model = train_model(data_yaml, epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size)
    torch.save(model.state_dict(), '/kaggle/working/runs/train/cadot_exp/weights/best.pt')
