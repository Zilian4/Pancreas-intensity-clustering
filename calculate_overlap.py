import nibabel as nib
import numpy as np
import os
import pandas as pd

def calculate_overlap(pancreas_mask, cyst_mask):
    # Calculate intersection (overlap)
    intersection = np.logical_and(pancreas_mask, cyst_mask)
    
    # Calculate pancreas area
    pancreas_area = np.sum(pancreas_mask)
    
    # Calculate cyst area
    cyst_area = np.sum(cyst_mask)
    if cyst_area == 0:
        return 0.0
    
    overlap_ratio = np.sum(intersection) / cyst_area
    return overlap_ratio

def main():
    # Define paths
    pancreas_mask_dir = r'E:\IPMN_images_masks\t2\masks'
    cyst_mask_dir = r'D:\Data\IPMNT2_Clustering\cystsMask_all'
    
    # Create lists to store results
    results = []
    
    # Process each file
    for filename in os.listdir(pancreas_mask_dir):
        if filename.endswith('.nii.gz'):
            print(f"Processing {filename}...")
            
            # Load masks
            pancreas_path = os.path.join(pancreas_mask_dir, filename)
            cyst_path = os.path.join(cyst_mask_dir, filename)
            
            # Check if cyst mask exists
            if not os.path.exists(cyst_path):
                print(f"Warning: No cyst mask found for {filename}")
                continue
            
            # Load the masks
            pancreas_img = nib.load(pancreas_path)
            cyst_img = nib.load(cyst_path)
            
            pancreas_mask = pancreas_img.get_fdata()
            cyst_mask = cyst_img.get_fdata()
            
            # Calculate overlap ratio
            overlap_ratio = calculate_overlap(pancreas_mask, cyst_mask)
            
            # Store results
            results.append({
                'filename': filename,
                'overlap_ratio': overlap_ratio,
                'pancreas_voxels': np.sum(pancreas_mask),
                'cyst_voxels': np.sum(cyst_mask),
                'overlap_voxels': np.sum(np.logical_and(pancreas_mask, cyst_mask))
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_path = 'pancreas_cyst_overlap.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total cases processed: {len(results)}")
    print(f"Mean overlap ratio: {df['overlap_ratio'].mean():.4f}")
    print(f"Median overlap ratio: {df['overlap_ratio'].median():.4f}")
    print(f"Min overlap ratio: {df['overlap_ratio'].min():.4f}")
    print(f"Max overlap ratio: {df['overlap_ratio'].max():.4f}")

if __name__ == "__main__":
    main() 