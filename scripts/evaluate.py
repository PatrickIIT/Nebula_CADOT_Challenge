%%writefile /kaggle/working/scripts/evaluate.py
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import argparse

def evaluate_predictions(gt_file, pred_file, output_csv):
    try:
        # Load ground truth and predictions
        coco_gt = COCO(gt_file)
        with open(pred_file, 'r') as f:
            coco_dt = coco_gt.loadRes(json.load(f))
        
        # Initialize evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract per-class AP
        class_aps = {}
        for i, cat_id in enumerate(coco_gt.getCatIds()):
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap = coco_eval.stats[0]  # mAP@50
            class_aps[coco_gt.cats[cat_id]['name']] = ap
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Class': list(class_aps.keys()),
            'AP@50': list(class_aps.values())
        })
        metrics_df.to_csv(output_csv, index=False)
        print(f"Metrics saved to {output_csv}")
        print(f"Overall mAP@50: {coco_eval.stats[0]:.2f}")
    except Exception as e:
        print(f"Evaluation error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CADOT predictions')
    parser.add_argument('--gt_file', type=str, default='/kaggle/input/cadot-paris/CADOT_Dataset/valid/_annotations.coco.json',
                        help='Path to ground truth annotations')
    parser.add_argument('--pred_file', type=str, default='/kaggle/working/results/val_predictions.json',
                        help='Path to predictions JSON')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/results/val_metrics.csv',
                        help='Output CSV for metrics')
    
    # Parse arguments, ignoring unknown ones
    args, _ = parser.parse_known_args()
    
    evaluate_predictions(args.gt_file, args.pred_file, args.output_csv)
