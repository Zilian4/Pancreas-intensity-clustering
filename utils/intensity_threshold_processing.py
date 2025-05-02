import argparse
import SimpleITK as sitk
import numpy as np
import os

def process_image_by_intensity_threshold(image_data, percentile=90):
    """
    Process 3D image based on intensity distribution
    Args:
        image_data: 3D numpy array
        percentile: percentage of pixels to consider (default 90%)
    """
    # Get non-zero pixels (ignore background)
    valid_mask = image_data > 0
    valid_intensities = image_data[valid_mask]
    
    # Calculate threshold (get max value covering 90% of pixels)
    threshold = np.percentile(valid_intensities, percentile)
    
    # Create new image array
    processed_image = np.copy(image_data)
    
    # Set values above threshold to threshold value
    processed_image[processed_image > threshold] = threshold
    
    print(f"Original intensity range: [{np.min(valid_intensities)}, {np.max(valid_intensities)}]")
    print(f"{percentile}th percentile threshold: {threshold}")
    print(f"Processed intensity range: [{np.min(processed_image)}, {np.max(processed_image)}]")
    
    return processed_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intensity distribution based image processing")
    parser.add_argument('-i', "--image-dir", required=True, type=str, help="Input image directory")
    parser.add_argument('-o', "--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument('-p', "--percentile", type=float, default=90, help="Percentage of pixels to consider (default 90)")
    
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
                
                # Process image
                processed_array = process_image_by_intensity_threshold(
                    image_array, 
                    percentile=args.percentile
                )
                
                # Save processed image
                output_image = sitk.GetImageFromArray(processed_array)
                output_image.CopyInformation(image)  # Copy original image metadata
                
                output_path = os.path.join(args.output_dir, name)
                sitk.WriteImage(output_image, output_path)
                print(f"Saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {name}: {e}") 