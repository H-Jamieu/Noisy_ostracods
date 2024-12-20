import torch
import pandas as pd
import numpy as np
from time import time

"""
Read the image embedding file form csv, the calculate the NN for each image
"""
def compute_knn(distance_matrix, k=7):
    """
    Compute the k nearest neighbors for each image
    :params: distance_matrix: The distance matrix
    :params: k: The number of nearest neighbors
    """
    # get the k nearest neighbors
    knn = np.argsort(distance_matrix, axis=1)[:, :k]
    return knn

def compute_distance_matrix(embedding_array, device, batch_sz=768, k=31):
    """
        Compute the distance matrix between the images using cosine similarity
        :params: embedding_array: The array of embeddings
    """
    # convert the embedding array to tensor
    embedding_tensor = torch.tensor(embedding_array).to(device)
    # normalize the tensor
    embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
    # comvert the tensor, scaling up the tensor then convert to fp16
    embedding_tensor = embedding_tensor * 1e2
    embedding_tensor = embedding_tensor.half()
    # print which device embedding tensor in
    print(embedding_tensor.device)
    all_knn = torch.tensor(()).to(device)
    for i in range(0, len(embedding_tensor), batch_sz):
        # get the batch
        batch = embedding_tensor[i:i+batch_sz]
        # calculate the distance matrix
        batch_distance_matrix = torch.matmul(batch, embedding_tensor.T)
        # faster
        batch_knn = torch.topk(batch_distance_matrix, k=k, dim=1, largest=False)[1]
        #batch_distance_matrix = batch_distance_matrix.cpu().numpy()
        #batch_knn = compute_knn(batch_distance_matrix)
        del batch_distance_matrix
        del batch
        all_knn = torch.cat((all_knn, batch_knn), 0)
        # release cache
        torch.cuda.empty_cache()
    # distance_matrix = torch.matmul(embedding_tensor, embedding_tensor.T)/1e8
    # # convert distance matrix to numpy
    # distance_matrix = distance_matrix.cpu().numpy()
    return all_knn.cpu().numpy()

def compute_nn(embedding_file, output_file, base_str,  device):
    """
    Compute the nearest neighbors for each image
    :params: embedding_file: The file containing the embeddings
    :params: output_file: The file to save the nearest neighbors
    :params: device: The device to run the computation
    """
    # read the embedding file
    t0 = time()
    embeddings = pd.read_csv(embedding_file, header=None)
    print(f"Read the embeddings in {time()-t0} seconds")
    # get the embeddings, the 2nd to 513rd columns
    embedding_array = embeddings.iloc[:, 1:].values
    # compute the distance matrix
    t1 = time()
    knn = compute_distance_matrix(embedding_array, device)
    print(f"Computed the knn in {time()-t1} seconds")
    # convert knn to array of int
    knn = knn.astype(int)
    # compute the nearest neighbors
    # t2 = time()
    # knn = compute_knn(distance_matrix)
    #print(f"Computed the nearest neighbors in {time()-t2} seconds")
    image_names = embeddings[0].values
    image_names = [name.replace(base_str, "") for name in image_names]
    # save the nearest neighbors
    t3 = time()
    with open(output_file, 'w') as f:
        for i in range(len(embeddings)):
            line = [image_names[i]]
            for j in range(len(knn[i])):
                line.append(image_names[knn[i][j]])
            f.write(", ".join(line) + "\n")
            # f.write(f"{image_names[i]}, {image_names[knn[i][0]]}, \
            #         {image_names[knn[i][1]]}, {image_names[knn[i][2]]}, {image_names[knn[i][3]]}, \
            #         {image_names[knn[i][4]]}, {image_names[knn[i][5]]}, {image_names[knn[i][6]]}\n")
    print(f"Saved the nearest neighbors in {time()-t3} seconds")

if __name__ == "__main__":
    embedding_file = "../datasets/embeddings/image_embeddings_DINOv2_g14_full.csv"
    output_file = "../datasets/embeddings/dinov2_g14_full_nn_31.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #base_str = "/mnt/c/Users/hjmfun/working_dir/ostracods_data/class_images/"
    base_str = "/mnt/x/class_images/"
    compute_nn(embedding_file, output_file, base_str,device)
