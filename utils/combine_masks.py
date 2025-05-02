import argparse
import SimpleITK as sitk
import numpy as np
import os

def combine_masks(mask1_data, mask2_data):
    """
    Combine two binary masks using union operation
    Args:
        mask1_data: First mask as numpy array
        mask2_data: Second mask as numpy array
    Returns:
        Combined mask as numpy array
    """
    # Ensure masks are binary
    mask1_binary = (mask1_data > 0).astype(np.uint8)
    mask2_binary = (mask2_data > 0).astype(np.uint8)
    
    # Perform union operation (logical OR)
    combined_mask = np.logical_or(mask1_binary, mask2_binary).astype(np.uint8)
    
    return combined_mask

def process_directories(mask1_dir, mask2_dir, output_dir):
    """
    Process all matching files in two directories
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of files in both directories
    mask1_files = {f for f in os.listdir(mask1_dir) if f.endswith('.nii.gz')}
    mask2_files = {f for f in os.listdir(mask2_dir) if f.endswith('.nii.gz')}
    
    # Find common files
    common_files = mask1_files.intersection(mask2_files)
    
    if not common_files:
        print("No matching .nii.gz files found in both directories!")
        return
    
    print(f"Found {len(common_files)} matching files to process")
    
    # Process each pair of files
    for filename in common_files:
        try:
            print(f"\nProcessing: {filename}")
            
            # Read masks
            mask1_path = os.path.join(mask1_dir, filename)
            mask2_path = os.path.join(mask2_dir, filename)
            
            mask1 = sitk.ReadImage(mask1_path)
            mask2 = sitk.ReadImage(mask2_path)
            
            # Convert to numpy arrays
            mask1_array = sitk.GetArrayFromImage(mask1)
            mask2_array = sitk.GetArrayFromImage(mask2)
            
            # Check if masks have same dimensions
            if mask1_array.shape != mask2_array.shape:
                print(f"Warning: Dimensions do not match for {filename}: {mask1_array.shape} vs {mask2_array.shape}")
                continue
            
            # Combine masks
            print("Combining masks...")
            combined_array = combine_masks(mask1_array, mask2_array)
            
            # Create output image
            output_image = sitk.GetImageFromArray(combined_array)
            output_image.CopyInformation(mask1)  # Copy metadata from first mask
            
            # Save combined mask
            output_path = os.path.join(output_dir, filename)
            print(f"Saving to: {output_path}")
            sitk.WriteImage(output_image, output_path)
            
            # Print statistics
            total_voxels = np.prod(combined_array.shape)
            mask1_voxels = np.sum(mask1_array > 0)
            mask2_voxels = np.sum(mask2_array > 0)
            combined_voxels = np.sum(combined_array > 0)
            
            print(f"Mask Statistics for {filename}:")
            print(f"Total voxels: {total_voxels}")
            print(f"Mask 1 voxels: {mask1_voxels} ({mask1_voxels/total_voxels*100:.2f}%)")
            print(f"Mask 2 voxels: {mask2_voxels} ({mask2_voxels/total_voxels*100:.2f}%)")
            print(f"Combined mask voxels: {combined_voxels} ({combined_voxels/total_voxels*100:.2f}%)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two segmentation masks using union operation")
    parser.add_argument('-m1', "--mask1", required=True, type=str, help="Path to first mask directory or file (.nii.gz)")
    parser.add_argument('-m2', "--mask2", required=True, type=str, help="Path to second mask directory or file (.nii.gz)")
    parser.add_argument('-o', "--output", required=True, type=str, help="Output directory or file path")
    
    args = parser.parse_args()
    
    # Check if inputs are directories
    if os.path.isdir(args.mask1) and os.path.isdir(args.mask2):
        print("Processing directories...")
        process_directories(args.mask1, args.mask2, args.output)
    else:
        # Process single file pair
        try:
            print("Processing single file pair...")
            mask1 = sitk.ReadImage(args.mask1)
            mask2 = sitk.ReadImage(args.mask2)
            
            mask1_array = sitk.GetArrayFromImage(mask1)
            mask2_array = sitk.GetArrayFromImage(mask2)
            
            if mask1_array.shape != mask2_array.shape:
                raise ValueError(f"Mask dimensions do not match: {mask1_array.shape} vs {mask2_array.shape}")
            
            combined_array = combine_masks(mask1_array, mask2_array)
            
            output_image = sitk.GetImageFromArray(combined_array)
            output_image.CopyInformation(mask1)
            
            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            sitk.WriteImage(output_image, args.output)
            print(f"Saved combined mask to: {args.output}")
            
        except Exception as e:
            print(f"Error occurred: {e}") 