import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def clean(img_data, seg_data):
    try:
        result_data = img_data * seg_data
        return result_data
    except Exception as e:
        print(f"An error occurred while processing: {e}")

def process_slice(slice_data, mask_slice, grid_size, spatial_weight):
    # Only process if there are non-zero pixels in the mask
    if np.sum(mask_slice) == 0:
        return np.zeros_like(slice_data)
    
    # Convert to SimpleITK image
    sitk_slice = sitk.GetImageFromArray(slice_data.astype(np.float32))
    
    # SLIC parameters
    slic_filter = sitk.SLICImageFilter()
    slic_filter.SetMaximumNumberOfIterations(100)
    slic_filter.SetSuperGridSize([grid_size, grid_size])  # Control granularity here
    slic_filter.SetSpatialProximityWeight(spatial_weight)  # Control spatial weight
    slic_filter.SetEnforceConnectivity(True)
    slic_filter.SetInitializationPerturbation(True)
    
    # Apply SLIC
    slic_img = slic_filter.Execute(sitk_slice)
    supervoxels = sitk.GetArrayFromImage(slic_img)
    
    # Only keep supervoxels within the mask
    supervoxels = supervoxels * mask_slice
    
    return supervoxels

def main():
    parser = argparse.ArgumentParser(description="SLIC Superpixel Segmentation for Pancreas.")
    parser.add_argument('-i', "--image-dir", required=True, type=str, help="images path")
    parser.add_argument('-o', "--output-dir", required=True, type=str, help="path to save outputs")
    parser.add_argument('-m', "--mask-dir", required=True, type=str, help="segmentation mask path")
    parser.add_argument('-g', "--grid-size", type=int, default=5, help="Size of superpixels (smaller = finer granularity)")
    parser.add_argument('-w', "--spatial-weight", type=float, default=0.1, help="Weight of spatial proximity (higher = more compact superpixels)")
    args = parser.parse_args()
    
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    grid_size = args.grid_size
    spatial_weight = args.spatial_weight
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name in os.listdir(image_dir):
        if name.endswith('.nii.gz'):
            print(f"Processing {name}...")
            
            # Load image and mask
            image_path = os.path.join(image_dir, name)
            mask_path = os.path.join(mask_dir, name)
            
            img = nib.load(image_path)
            img_data = img.get_fdata()
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            
            # Clean data to get pancreas region
            clean_data = clean(img_data, mask_data)
            
            # Process slice by slice
            result_data = np.zeros_like(clean_data)
            for z in range(clean_data.shape[2]):
                slice_data = clean_data[:, :, z]
                mask_slice = mask_data[:, :, z]
                result_data[:, :, z] = process_slice(slice_data, mask_slice, grid_size, spatial_weight)
            
            # Save results
            output_path = os.path.join(output_dir, name)
            output_nifti = nib.Nifti1Image(result_data, img.affine, img.header)
            nib.save(output_nifti, output_path)
            print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()

