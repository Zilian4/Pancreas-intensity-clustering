# %%
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import argparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances

# %%
def intensity_clustering(input_path, output_path,algorithms, n_clusters=3):
    
    img = nib.load(input_path)
    img_data = img.get_fdata()
    flat_data = img_data.reshape(-1, 1)
    
    tissue_mask = flat_data != 0
    tissue_part = flat_data[tissue_mask].reshape(-1, 1)
    
    if algorithms == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tissue_part) 
        tissue_clustered_label= kmeans.labels_ # get the clustering results of tissue
        
    elif algorithms == 'spectral':
        # print(np.std(tissue_part))
        # similarity_matrix = np.exp(-euclidean_distances(tissue_part, tissue_part) ** 2 / (2.0 * np.std(tissue_part) ** 2))
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors', 
            random_state=42
        )
        tissue_clustered_label = spectral.fit_predict(tissue_part)
        # print(tissue_clustered_label)

    image_clustered = np.zeros_like(flat_data) # zero array to store results
    image_clustered[tissue_mask] = tissue_clustered_label+1 # insert results in the zore array
    
    image_clustered = image_clustered.reshape(img_data.shape) # reshape
    new_img = nib.Nifti1Image(image_clustered.astype(np.int16), img.affine, img.header)
    # nib.save(new_img, output_path)
    
    print(f"Clustering result saved as {output_path}")
    return image_clustered

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--image-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-o',"--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument('-n',"--n_clusters",default=3,type=int,help='Number of cluster centers')
    parser.add_argument('-a','--algorithms',default='kmeans',type=str,help='Has to be ')
    args = parser.parse_args()
    
    n = args.n_clusters
    input_dir = args.image_dir
    algorithms = args.algorithms
    output_dir = os.path.join(args.output_dir,f'{algorithms} Clusters_{n}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for name in os.listdir(input_dir):
        if name.endswith('.nii.gz'):
            input_path = os.path.join(input_dir,name)
            output_path = os.path.join(output_dir,name)
            c_data = intensity_clustering(input_path=input_path,output_path=output_path,n_clusters=n,algorithms=algorithms)
            print(f"Clustering result saved as {output_path}")