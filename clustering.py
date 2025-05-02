# %%
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import os
import argparse
from sklearn.cluster import SpectralClustering,HDBSCAN
from sklearn.mixture import GaussianMixture

def intensity_normalization(img_data):
    return ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)

def remap_cluster_labels(image, labels):
    # Get unique labels
    unique_labels = np.unique(labels)
    # Compute the mean intensity of pixels for each cluster label
    mean_intensities = []
    for label in unique_labels:
        mean_intensity = np.mean(image[labels == label])
        mean_intensities.append((label, mean_intensity))

    # Sort labels by mean intensity
    sorted_labels = sorted(mean_intensities, key=lambda x: x[1])
    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label + 1 for new_label, (old_label, _) in enumerate(sorted_labels)}
    # Remap the labels
    remapped_labels = np.vectorize(label_mapping.get)(labels)

    return remapped_labels


def rbf_kernel(X, gamma=1.0):
    pairwise_sq_dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
    return np.exp(-gamma * pairwise_sq_dists)


def clean(img_data, seg_data):
    try:
        result_data = img_data * seg_data
        return result_data
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

def intensity_clustering(image_path, mask_path,algorithms, n_clusters=3):
    img = nib.load(image_path)
    img_data = img.get_fdata()  
    
    seg_img = nib.load(mask_path)
    seg_data = seg_img.get_fdata()
    
    # Multiply image with the mask to get the pancreas region
    clean_data = clean(img_data,seg_data)
    
    flat_data = clean_data.reshape(-1, 1)
    
    tissue_mask = flat_data != 0
    tissue_part = flat_data[tissue_mask].reshape(-1, 1)
    
    tissue_part = intensity_normalization(tissue_part)
    
    if algorithms == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(tissue_part) 
        tissue_clustered_label= kmeans.labels_ # get the clustering results of tissue
        
    elif algorithms == 'spectral':

        similarity_matrix = np.exp(-euclidean_distances(tissue_part, tissue_part) ** 2 / (2.0 * np.std(tissue_part) ** 2))
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors', 
            random_state=42,
        )

        tissue_clustered_label = spectral.fit_predict(tissue_part)
        # print(tissue_clustered_label)
        
    elif algorithms =='gmm':
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        gmm.fit(tissue_part)
        tissue_clustered_label = gmm.predict(tissue_part)

    image_clustered = np.zeros_like(flat_data) # zero array to store results

    tissue_clustered_label = remap_cluster_labels(tissue_part,tissue_clustered_label)

    image_clustered[tissue_mask] = tissue_clustered_label # insert results in the zore array
    
    image_clustered = image_clustered.reshape(img_data.shape) # reshape
    new_img = nib.Nifti1Image(image_clustered.astype(np.int16), img.affine, img.header)
    return new_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--image-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-o',"--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument('-n',"--n_clusters",default=3,type=int,help='Number of cluster centers')
    parser.add_argument('-a','--algorithms',default='kmeans',type=str,help='Has to be the name of algorithm')
    parser.add_argument('-m','--mask-dir',default='None',type=str,help='segmentation mask')
    args = parser.parse_args()
    
    n = args.n_clusters
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    algorithms = args.algorithms
    output_dir = os.path.join(args.output_dir,f'{algorithms} Clusters_{n}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for name in os.listdir(image_dir):
        if name.endswith('.nii.gz') and not name.startswith('.'):
            try:
                image_path = os.path.join(image_dir,name)
                mask_path =os.path.join(mask_dir,name)
                output_path = os.path.join(output_dir,name)
                cluster_label = intensity_clustering(image_path=image_path,mask_path = mask_path,n_clusters=n,algorithms=algorithms)
                nib.save(cluster_label, output_path)
                print(f"Clustering result saved as {output_path}")
            except Exception as e:
                print(f"An error occurred while processing {image_path}: {e}")
            