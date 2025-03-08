import nibabel as nib
import os 
from intensity_normalization.normalize.nyul import NyulNormalize
import SimpleITK as sitk
import argparse
import gc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--image-dir", default=None, required=True,type=str, help="image dir path")
    parser.add_argument('-o',"--output-dir", default=None, type=str, help="path to save outputs")

    args = parser.parse_args()
    img_dir = args.image_dir
    save_dir = args.output_dir
    image_paths = [os.path.join(img_dir,name) for name in os.listdir(img_dir)]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    images_np = {}
    for image_path in image_paths:
        images_np[image_path] = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))  

    print("Images read")
    # normalize the images and save the standard histogram
    nyul_normalizer = NyulNormalize()
    nyul_normalizer.fit(list(images_np.values()))
    print("Fitting completed")
    
    normalized_images = {}
    for image_path in image_paths:
        normalized_images[image_path] = nyul_normalizer(images_np[image_path])
        del images_np[image_path]  
        gc.collect()
    print("Normalization completed")
    
    for image_path in image_paths:
        image_name = image_path.split('/')[-1]
        save_path = os.path.join(save_dir,image_name)
        if os.path.exists(save_path):
            print(image_name,'skipped')
            continue
        # Convert to array
        array = normalized_images[image_path]
        
        # Standarization
        standardized_image = sitk.GetImageFromArray(array)
        sitk.WriteImage(standardized_image, save_path)
        # Load original image
        original_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        
        # Copy info from original image
        standardized_image.CopyInformation(original_image) 
        
        sitk.WriteImage(standardized_image, save_path)
        print(f"Image :{image_name} standardization completed, saved to {save_path}")
    nyul_normalizer.save_standard_histogram(os.path.join(save_dir,"standard_histogram.npy"))