import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU,ConfusionMatrixMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from sklearn.metrics import precision_score, recall_score
import argparse

def main(args):
    if not args.output_csv.endswith('.csv'):
        raise ValueError("Output file must have a .csv extension.")
    # Define paths
    segmentation_path = args.segmentation_folder
    ground_truth_path = args.groudtruth_folder
    names = os.listdir(segmentation_path)

    # Define the number of classes
    num_classes = 4  # Update this based on the actual number of classes
    labels = list(range(1, num_classes))  # Exclude background (0)

    # Initialize metrics (excluding background)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)
    assd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
    confusion = ConfusionMatrixMetric(include_background=False,reduction='mean_batch',metric_name = ["sensitivity","precision"])

    # Post-processing
    post_pred = AsDiscrete(argmax=True)  # Convert to discrete predictions if needed
    post_label = AsDiscrete(to_onehot=num_classes)  # Convert labels to one-hot

    metrics_list = []

    # Evaluation Loop
    for name in tqdm(names, desc="Evaluating Segmentation Results"):
        try:
            seg_img = nib.load(os.path.join(segmentation_path, name))
            gt_img = nib.load(os.path.join(ground_truth_path, name))
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
        seg_array = torch.tensor(seg_img.get_fdata(), dtype=torch.int32).unsqueeze(0)  # (1, H, W, D)
        gt_array = torch.tensor(gt_img.get_fdata(), dtype=torch.int32).unsqueeze(0)

        if seg_array.shape != gt_array.shape:
            print(f"Skipping {name}: Shape mismatch (Seg: {seg_array.shape}, GT: {gt_array.shape})")
            continue
        
        # Convert to one-hot representation
        seg_onehot = post_label(seg_array)
        gt_onehot = post_label(gt_array)
        # # Compute Metrics
        dice_metric(y_pred=[seg_onehot], y=[gt_onehot])
        hd95_metric(y_pred=[seg_onehot], y=[gt_onehot])
        assd_metric(y_pred=[seg_onehot], y=[gt_onehot])
        iou_metric(y_pred=[seg_onehot], y=[gt_onehot])
        confusion(y_pred=[seg_onehot], y=[gt_onehot])
        
        # # Aggregate and store results
        dice_scores = dice_metric.aggregate().tolist()
        hd95_scores = hd95_metric.aggregate().tolist()
        assd_scores = assd_metric.aggregate().tolist()
        iou_scores = iou_metric.aggregate().tolist()
        sensitivity, precision = confusion.aggregate()[0].tolist(),confusion.aggregate()[1].tolist()
        
        dice_metric.reset()
        hd95_metric.reset()
        assd_metric.reset()
        iou_metric.reset()
        confusion.reset()
        # Store results in dictionary
        metrics_dict = {"name": name}
        for i, label in enumerate(labels):
            metrics_dict[f"dice_class_{label}"] = dice_scores[i]
            metrics_dict[f"hd95_class_{label}"] = hd95_scores[i]
            metrics_dict[f"assd_class_{label}"] = assd_scores[i]
            metrics_dict[f"iou_class_{label}"] = iou_scores[i]
            metrics_dict[f"precision_class_{label}"] = precision[i]
            metrics_dict[f"recall_class_{label}"] = sensitivity[i]
            
        metrics_dict[f"dice_avg"] = np.array(dice_scores).mean()
        metrics_dict[f"hd95_avg"] = np.array(hd95_scores).mean()
        metrics_dict[f"assd_avg"] = np.array(assd_scores).mean()
        metrics_dict[f"iou_avg"] = np.array(iou_scores).mean()
        metrics_dict[f"precision_avg"] = np.array(precision).mean()
        metrics_dict[f"recall_avg"] = np.array(sensitivity).mean()
        
        metrics_list.append(metrics_dict)

    # Save results
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(args.output_csv, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract radiomic features from all images in a folder and save them to a CSV.")
    parser.add_argument('-s', '--segmentation_folder', required=True, help="Folder containing image files.")
    parser.add_argument('-g', '--groudtruth_folder', required=True, help="Folder containing mask files corresponding to images.")
    parser.add_argument('-o', '--output_csv', default='radiomics_features.csv', help="Output path for the CSV file (default: radiomics_features.csv).")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)