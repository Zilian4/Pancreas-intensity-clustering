import SimpleITK as sitk
import numpy as np
import copy
import cv2
from scipy.ndimage import uniform_filter
import cv2.ximgproc as xip
def simple_guided_filter(I, p, radius, eps):
    mean_I = uniform_filter(I, size=radius)
    mean_p = uniform_filter(p, size=radius)
    corr_I = uniform_filter(I * I, size=radius)
    corr_Ip = uniform_filter(I * p, size=radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=radius)
    mean_b = uniform_filter(b, size=radius)

    q = mean_a * I + mean_b
    return q

# Rolling Guidance Filter
def rolling_guidance_filter_simple(I, sigma_s=8, sigma_r=0.5):
    I = np.float32(I)
    max_value = np.max(I)
    min_value = np.min(I)
    I = (I - min_value) / (max_value - min_value)  
    I = cv2.bilateralFilter(I, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)

    # guide = np.clip(guide, 0, 1)
    I = I*(max_value - min_value) + min_value
    return I
    # return (guide * 255).astype(np.uint8)


def unsharp_mask(image, amount=1.0, radius=1):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)




class Preprocessor():
    def __init__(self):
        self.image = None
        self.original_image = None

    def set_image(self,image):
        self.image = image
        self.original_image = copy.deepcopy(image)

    def get_image(self):
        return self.image

    def clache(self,):
        clahe_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
        clahe_filter.SetAlpha(0.3)  # Controls how much histogram is modified (0.0 - 1.0)
        clahe_filter.SetBeta(0.3)   # Controls how much enhancement is applied (0.0 - 1.0)

        # Apply CLAHE to the entire 3D volume
        self.image = clahe_filter.Execute(self.image)

    def intensity_normalization(input_array:np.array):
        return ((input_array - input_array.min()) / (input_array.max() - input_array.min()))*255

    def n4_bias_correction(self):
        """
        Perform N4 Bias Field Correction on a medical image.
        
        Args:
            input_image_path (str): Path to the input .nii.gz image.
            output_image_path (str): Path to save the corrected .nii.gz image.
        
        Returns:
            sitk.Image: Bias-corrected image.
        """
        n4_corrector = sitk.N4BiasFieldCorrectionImageFilter()
        self.image = n4_corrector.Execute(self.image)
        
        return self.image

    def bilateral_denoising(self, sigma_s=8, sigma_r=1):
        """Apply rolling guidance filter to the image slice by slice."""
        if self.image is None:
            raise ValueError("No image set")
        
        # Convert to numpy array for processing
        img_array = sitk.GetArrayFromImage(self.image)
        result = np.zeros_like(img_array)
        
        # Process each slice
        for z in range(img_array.shape[0]):
            slice_data = img_array[z]
            # Apply rolling guidance filter to the slice
            result[z] = rolling_guidance_filter_simple(slice_data, sigma_s=sigma_s, sigma_r=sigma_r)
            # Print progress
            if z % 10 == 0:
                print(f"Processing slice {z}/{img_array.shape[0]}")
        
        # Convert back to SimpleITK image
        self.image = sitk.GetImageFromArray(result)
        return self.image

    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply CLAHE to the image."""
        if self.image is None:
            raise ValueError("No image set")
        
        # Convert to numpy array for processing
        img_array = sitk.GetArrayFromImage(self.image)
        result = np.zeros_like(img_array)
        
        # Process each slice
        for z in range(img_array.shape[0]):
            slice_data = img_array[z]
            # Apply CLAHE
            clahe_filter = sitk.CLAHEImageFilter()
            clahe_filter.SetClipLimit(clip_limit)
            clahe_filter.SetTilesGridSize(tile_grid_size)
            result[z] = sitk.GetArrayFromImage(
                clahe_filter.Execute(sitk.GetImageFromArray(slice_data))
            )
        
        # Convert back to SimpleITK image
        self.image = sitk.GetImageFromArray(result)

    def unsharp_mask(self, amount=5, radius=1):
        """Apply unsharp mask to the image."""
        if self.image is None:
            raise ValueError("No image set")
        
        # Convert to numpy array for processing
        img_array = sitk.GetArrayFromImage(self.image)
        result = np.zeros_like(img_array)
        
        # Process each slice
        for z in range(img_array.shape[0]):
            slice_data = img_array[z]
            # Apply unsharp mask
            blurred = sitk.GetArrayFromImage(
                sitk.DiscreteGaussian(sitk.GetImageFromArray(slice_data), radius)
            )
            result[z] = slice_data + amount * (slice_data - blurred)
        
        # Convert back to SimpleITK image
        self.image = sitk.GetImageFromArray(result)

