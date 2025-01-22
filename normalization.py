
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import argparse

def intensity_normalization(img_data):
    return ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--image-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-o',"--output-dir", default=None, type=str, help="path to save outputs")

    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f'Folder created:{output_dir}')
    
    for name in os.listdir(image_dir):
        if name.endswith('.nii.gz'):
            image_path = os.path.join(image_dir,name)
            
            img = nib.load(image_path)
            img_data = img.get_fdata()  
            
            img_data = intensity_normalization(img_data)
            new_img = nib.Nifti1Image(img_data.astype(np.int16), img.affine, img.header)
            output_path = os.path.join(output_dir,name)
            
            nib.save(new_img, output_path)
            print(f"Normalized result saved as {output_path}")
            