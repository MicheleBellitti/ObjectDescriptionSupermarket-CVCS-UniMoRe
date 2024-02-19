import os
import math
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import ToTensor, Resize, Compose
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from datasets import FreiburgDataset
# Configuration parameters
DATA_DIR = "images/"
MODEL_PATH = "freiburg/single/classifier.pth"
NUM_CLASSES = 25
N_COMPONENTS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 25
BATCH_SIZE = 32

def load_model(model_path=MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE):
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Linear(256, num_classes)
        )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# Feature extraction model setup
class DenseNetEmbedding(nn.Module):
    def __init__(self, model):
        super(DenseNetEmbedding, self).__init__()
        self.features = model.features

    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(out, 1)

# Load the model
# model = load_model()
# embedding_model = DenseNetEmbedding(model).to(device)

"""## Feature Extraction and PCA Cell: Extracting and Processing Embeddings, Including PCA

"""

def extract_embeddings(model, dataloader, device=DEVICE):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            emb = model(inputs)
            embeddings.append(emb)

    # Concatenate all embeddings and then transfer to CPU
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    return embeddings

def apply_pca(embeddings, n_components=N_COMPONENTS):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

# Example of use
# test_dataset = FreiburgDataset(split='test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# embeddings = extract_embeddings(embedding_model, test_loader)
# pca_embeddings = apply_pca(embeddings)

# Similarity Calculation

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

def calculate_cosine_similarity(query_embedding, all_embeddings):
    return cosine_similarity(query_embedding.reshape(1, -1), all_embeddings).flatten()

def calculate_mahalanobis_distances(query_embedding, all_embeddings, covariance_matrix):
    """
    Calculate Mahalanobis distances between the query embedding and all other embeddings.
    """
    distances = [distance.mahalanobis(query_embedding, emb, covariance_matrix) for emb in all_embeddings]
    return np.array(distances)

def compute_euclidean_distances(query_embedding, all_embeddings):
    """
    Calculate Euclidean distances between the query embedding and all other embeddings.
    """
    distances = [np.linalg.norm(query_embedding - emb) for emb in all_embeddings]
    return np.array(distances)




# Compute the inverse of the covariance matrix
#covariance_matrix = np.cov(embeddings.T)
#inv_covariance_matrix = np.linalg.inv(covariance_matrix)

# Example of use:
# mahalanobis_distances = calculate_mahalanobis_distances(query_embedding, pca_embeddings, inv_covariance_matrix)

# Example of use
# query_embedding = ...
# cosine_scores = calculate_cosine_similarity(query_embedding, pca_embeddings)

# Display Images and Results

"""

def show_images(images, titles, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Example of use
import random
rand = random.randint(0,len(test_dataset))
sample_images, _ = zip(*[test_dataset[i] for i in range(rand, rand+5)])
show_images(sample_images, ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'])
"""

# Mai


# Load model and data
model = load_model()
embedding_model = DenseNetEmbedding(model).to(device)
test_dataset = FreiburgDataset(split='test', transform=Compose([Resize(256), ToTensor()])), index=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Extract embeddings and apply PCA
embeddings = extract_embeddings(embedding_model, test_loader)
# pca_embeddings = apply_pca(embeddings)

# Calculate the inverse of the covariance matrix for Mahalanobis Distance
covariance_matrix = np.cov(embeddings.T)
inv_covariance_matrix = np.linalg.inv(covariance_matrix)

# Calculate similarity for a query image
query_images_dir = "/content/gdrive/MyDrive/CVCS_Project_23/retrieval_images"
image_path = f"{query_images_dir}/candy.jpg"
query_image = Image.open(image_path).convert('RGB')
query_vector = transform(query_image).to(device).unsqueeze(0)
query_embedding = embedding_model(query_vector).detach().cpu().numpy()

# pca_query_embedding = apply_pca(query_embedding)
query_embedding = query_embedding.reshape((query_embedding.shape[1],)) # Calculate Cosine Similarity and Mahalanobis Distance

cosine_scores = calculate_cosine_similarity(query_embedding, embeddings)
mahalanobis_distances = calculate_mahalanobis_distances(query_embedding, embeddings, inv_covariance_matrix)
euclidean_distances = compute_euclidean_distances(query_embedding, embeddings)

print(max(cosine_scores))
print(min(euclidean_distances))
# Show results for Cosine Similarity
top_cosine_indices = np.argsort(cosine_scores)[::-1][:5]
top_cosine_images, _ = zip(*[test_dataset[i] for i in top_cosine_indices])
show_images(top_cosine_images, ['Cosine Top 1', 'Cosine Top 2', 'Cosine Top 3', 'Cosine Top 4', 'Cosine Top 5'])

# Show results for Mahalanobis Distance
top_mahalanobis_indices = np.argsort(mahalanobis_distances)[:5]  # lower distances are better
top_mahalanobis_images, _ = zip(*[test_dataset[i] for i in top_mahalanobis_indices])
show_images(top_mahalanobis_images, ['Mahalanobis Top 1', 'Mahalanobis Top 2', 'Mahalanobis Top 3', 'Mahalanobis Top 4', 'Mahalanobis Top 5'])

# Show results for Euclidian Distance
top_euclidean_indices = np.argsort(euclidean_distances)[:5]  # lower distances are better
top_euclidean_images, _ = zip(*[test_dataset[i] for i in top_euclidean_indices])
show_images(top_euclidean_images, ['Euclidean Top 1', 'Euclidean Top 2', 'Euclidean Top 3', 'Euclidean Top 4', 'Euclidean Top 5'])

