
import SimpleITK as sitk
import numpy as np
import copy
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
    
    def bilateral_denoising(self):

        domain_sigma = 8  # Spatial domain standard deviation
        range_sigma = 0.5  # Intensity range standard deviation
        bilateral_filter = sitk.BilateralImageFilter()
        bilateral_filter.SetDomainSigma(domain_sigma)
        bilateral_filter.SetRangeSigma(range_sigma)
        self.image = bilateral_filter.Execute(self.image)
        return self.image

    # def gaussian_denoising():
    #     """
    #     Apply Gaussian denoising to the input image.
    #     Args:
    #         input_image (sitk.Image): Input image for denoising.
    #         sigma (float): Standard deviation for Gaussian filter.
    #     Returns:
    #         sitk.Image: Denoised image.
    #     """
    #     # Convert SimpleITK image to NumPy array for filtering
    #     sigma=0.5
    #     input_image = 
    #     input_array = sitk.GetArrayFromImage(input_image)
    #     # Apply Gaussian filter
    #     denoised_array = gaussian_filter(input_array, sigma=sigma)
    #     # Convert back to SimpleITK image
    #     denoised_image = sitk.GetImageFromArray(denoised_array)
    #     denoised_image.CopyInformation(input_image)  # Preserve original image metadata
    #     return denoised_image



    # def unsharp_masking(input_image):

    #     # Convert SimpleITK image to NumPy array for filtering
    #     input_array = sitk.GetArrayFromImage(input_image)
        
    #     # Normalization 
    #     normalized_image = intensity_normalization(input_array)
    #     input_array = normalized_image/255

    #     # Apply Gaussian filter
    #     sharpened_array = filters.unsharp_mask(input_array, radius=1, amount=2)
        
    #     # Convert back to SimpleITK image
    #     sharpened_image = sitk.GetImageFromArray(intensity_normalization(sharpened_array))
    #     sharpened_image.CopyInformation(input_image)  # Preserve original image metadata
    #     return sharpened_image