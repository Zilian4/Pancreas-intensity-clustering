import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import os

def intensity_clustering(input_path, output_path, n_clusters=3):

    img = nib.load(input_path)
    img_data = img.get_fdata()


    flat_data = img_data.reshape(-1, 1)
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flat_data)
    

    clustered_data = kmeans.labels_.reshape(img_data.shape)
    

    new_img = nib.Nifti1Image(clustered_data.astype(np.int16), img.affine, img.header)
    
    nib.save(new_img, output_path)

    print(f"Clustering result saved as {output_path}")


input_file = 'path/to/input_image.nii.gz'
output_file = 'path/to/output_clustered_image.nii.gz'
intensity_clustering(input_file, output_file, n_clusters=3)
