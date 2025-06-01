# CADOT Dataset Structure

The CADOT dataset is organized as follows for the IEEE ICIP 2025 Grand Challenge:

- **train/**: Training images and annotations
  - **images/**: JPG images of urban scenes (e.g., `75-2021-0640-6860-LA93-0M20-E080-1175_jpeg.rf.a995e128310478d56dcf9fbb2a943d78.jpg`)
  - **labels.json**: COCO-format annotations with bounding boxes for 14 categories
- **val/**: Validation images and annotations
  - **images/**: JPG images
  - **labels.json**: COCO-format annotations
- **test/**: Test images
  - **images/**: JPG images for inference (no annotations provided)
- **image_ids.csv**: Maps test image filenames to `image_id` for submission
- **data.yaml**: Dataset configuration for YOLOv5 training
  ```yaml
  train: /kaggle/input/cadot-paris/CADOT_Dataset/train/images
  val: /kaggle/input/cadot-paris/CADOT_Dataset/val/images
  test: /kaggle/input/cadot-paris/CADOT_Dataset/test/images
  nc: 14
  names: ['Basketball Field', 'Building', 'Crosswalk', 'Football Field', 'Graveyard', 'Large Vehicle', 'Medium Vehicle', 'Playground', 'Roundabout', 'Ship', 'Small Vehicle', 'Swimming Pool', 'Tennis Court', 'Train']
