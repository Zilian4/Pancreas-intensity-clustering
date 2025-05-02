import argparse
import SimpleITK as sitk
import numpy as np
import os
import cv2

def apply_clahe_3d(image_data, clip_limit=2.0, grid_size=(8,8)):
    """
    Apply CLAHE to 3D image using OpenCV
    Args:
        image_data: 3D numpy array
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
    Returns:
        Processed 3D numpy array
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    # Process each slice
    processed_slices = []
    for slice_idx in range(image_data.shape[0]):
        # Get slice and convert to uint8
        slice_2d = image_data[slice_idx]
        slice_2d = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE
        processed_slice = clahe.apply(slice_2d)
        
        # Convert back to original range
        processed_slice = (processed_slice.astype(float) / 255.0 * 
                         (image_data[slice_idx].max() - image_data[slice_idx].min()) + 
                         image_data[slice_idx].min())
        
        processed_slices.append(processed_slice)
    
    return np.stack(processed_slices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply CLAHE to 3D medical images")
    parser.add_argument('-i', "--image-dir", required=True, type=str, help="Input image directory")
    parser.add_argument('-o', "--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument('-c', "--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default 2.0)")
    parser.add_argument('-g', "--grid-size", type=int, default=8, help="CLAHE grid size (default 8)")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process all images in directory
    for name in os.listdir(args.image_dir):
        if name.endswith('.nii.gz'):
            try:
                print(f"\nProcessing image: {name}")
                
                # Read image
                image_path = os.path.join(args.image_dir, name)
                image = sitk.ReadImage(image_path)
                image_array = sitk.GetArrayFromImage(image)
                
                # Apply CLAHE
                print("Applying CLAHE...")
                processed_array = apply_clahe_3d(
                    image_array,
                    clip_limit=args.clip_limit,
                    grid_size=(args.grid_size, args.grid_size)
                )
                
                # Save processed image
                output_image = sitk.GetImageFromArray(processed_array)
                output_image.CopyInformation(image)  # Copy original image metadata
                
                output_path = os.path.join(args.output_dir, name)
                sitk.WriteImage(output_image, output_path)
                print(f"Saved to: {output_path}")
                
            except Exception as e:
                print(f"Error occurred when processing {name}: {e}") 