import argparse
import SimpleITK as sitk
import os
from Preprocessor import Preprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="bilateral_denoising Filter for Medical Images.")
    parser.add_argument('-i', "--image-dir", default=None, required=True, type=str, help="image dir path")
    parser.add_argument('-o', "--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument('-s', "--sigma-s", type=float, default=8, help="Spatial sigma for Gaussian blur")
    parser.add_argument('-r', "--sigma-r", type=float, default=0.1, help="Range sigma for edge preservation")
    args = parser.parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir
    sigma_s = args.sigma_s
    sigma_r = args.sigma_r

    preprocessor = Preprocessor()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for name in os.listdir(image_dir):
        if name.endswith('.nii.gz') and not name.startswith('.'):
            try:
                save_path = os.path.join(output_dir, name)
                print(save_path)
                if os.path.exists(save_path):
                    continue
                    
                image_path = os.path.join(image_dir, name)
                raw_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                
                # Store original image information
                original_spacing = raw_image.GetSpacing()
                original_origin = raw_image.GetOrigin()
                original_direction = raw_image.GetDirection()
                original_metadata = {k: raw_image.GetMetaData(k) for k in raw_image.GetMetaDataKeys()}
                
                preprocessor.set_image(raw_image)
                
                # Apply rolling guidance filter
                print('Applying rolling guidance filter...')
                processed_image = preprocessor.bilateral_denoising(sigma_s=sigma_s, sigma_r=sigma_r)
                
                # Restore original image information
                processed_image.SetSpacing(original_spacing)
                processed_image.SetOrigin(original_origin)
                processed_image.SetDirection(original_direction)
                for k, v in original_metadata.items():
                    processed_image.SetMetaData(k, v)
                
                # Save the processed image
                sitk.WriteImage(processed_image, save_path)
                print('done')
                print(name, 'Rolling guidance filter finished, image saved to', save_path)
                
            except Exception as e:
                print(f'Error occurred when processing {name}: {e}') 