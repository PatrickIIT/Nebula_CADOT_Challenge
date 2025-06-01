import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import os
import argparse

def evaluate_predictions(gt_json, pred_json, output_csv):
    """Evaluate predictions and save per-class metrics."""
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    class_names = [
        'Basketball Field', 'Building', 'Crosswalk', 'Football Field', 'Graveyard',
        'Large Vehicle', 'Medium Vehicle', 'Playground', 'Roundabout', 'Ship',
        'Small Vehicle', 'Swimming Pool', 'Tennis Court', 'Train'
    ]
    metrics = {'Class': [], 'AP@50': [], 'AR@50': []}
    
    for i, cat_id in enumerate(coco_gt.getCatIds()):
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap = coco_eval.stats[1]
        ar = coco_eval.stats[8]
        metrics['Class'].append(class_names[i])
        metrics['AP@50'].append(ap)
        metrics['AR@50'].append(ar)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', type=str, default='/kaggle/input/cadot-paris/CADOT_Dataset/valid/_annotations.coco.json', help='Ground truth COCO JSON')
    parser.add_argument('--pred_json', type=str, default='/kaggle/working/results/predictions.json', help='Predictions COCO JSON')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/results/metrics.csv', help='Output CSV')
    args = parser.parse_args()
    
    evaluate_predictions(args.gt_json, args.pred_json, args.output_csv)
