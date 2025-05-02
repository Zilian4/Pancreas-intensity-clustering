import argparse
import SimpleITK as sitk
import os
from utils.Preprocessor import Preprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unsharp Mask Processing for Medical Images.")
    parser.add_argument('-i', "--image-dir", default=None, required=True, type=str, help="image dir path")
    parser.add_argument('-o', "--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument('-a', "--amount", type=float, default=5, help="Amount of sharpening")
    parser.add_argument('-r', "--radius", type=float, default=1, help="Radius for Gaussian blur")
    
    args = parser.parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir
    amount = args.amount
    radius = args.radius

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
                preprocessor.set_image(raw_image)
                
                # Apply unsharp mask
                print('Applying unsharp mask...')
                preprocessor.unsharp_mask(amount=amount, radius=radius)
                
                # Save the processed image
                image_pre = preprocessor.get_image()
                sitk.WriteImage(image_pre, save_path)
                print(name, 'Unsharp mask processing finished')
                
            except Exception as e:
                print(f'Error occurred when processing {name}: {e}') 