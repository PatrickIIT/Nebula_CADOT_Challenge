%%writefile /kaggle/working/scripts/augment.py
import cv2
import numpy as np
import random

def apply_copy_paste_augmentation(images, annotations, prob=0.5):
    """
    Apply copy-paste augmentation to images and annotations.
    Args:
        images: List of image arrays.
        annotations: List of COCO-format annotations.
        prob: Probability of applying augmentation.
    Returns:
        Augmented images and annotations.
    """
    if random.random() > prob:
        return images, annotations
    
    # Randomly select source and target images
    src_idx = random.randint(0, len(images) - 1)
    tgt_idx = random.randint(0, len(images) - 1)
    src_img = images[src_idx].copy()
    src_anns = [ann for ann in annotations if ann['image_id'] == src_idx]
    tgt_img = images[tgt_idx].copy()
    tgt_anns = [ann for ann in annotations if ann['image_id'] == tgt_idx]
    
    # Copy objects from source to target
    for ann in src_anns:
        bbox = ann['bbox']  # [x_min, y_min, w, h]
        x, y, w, h = [int(v) for v in bbox]
        obj = src_img[y:y+h, x:x+w]
        # Paste at random location in target image
        tgt_h, tgt_w = tgt_img.shape[:2]
        x_new = random.randint(0, tgt_w - w)
        y_new = random.randint(0, tgt_h - h)
        tgt_img[y_new:y_new+h, x_new:x_new+w] = obj
        # Update annotations
        new_ann = ann.copy()
        new_ann['bbox'] = [x_new, y_new, w, h]
        new_ann['image_id'] = tgt_idx
        annotations.append(new_ann)
    
    images[tgt_idx] = tgt_img
    return images, annotations
