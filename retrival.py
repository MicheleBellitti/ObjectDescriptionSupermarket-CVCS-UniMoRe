import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis, euclidean
from matplotlib import pyplot as plt
from datasets import FreiburgDataset

image_path = "retrival_files/images/pasta_1.jpg"  # Update this path

# Configuration settings for the image retrieval system
class Config:
    DATA_DIR = "/work/cvcs_2023_group23/images/"  # Directory for dataset images
    MODEL_PATH = "freiburg/single/classifier.pth"  # Path to the trained model
    NUM_CLASSES = 25  # Number of classes in the dataset
    N_COMPONENTS = 50  # Number of components for PCA
    BATCH_SIZE = 32  # Batch size for data loading
    #DEVICE = torch.device('cpu')  # Device configuration
    print(torch.cuda.is_available())
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration
    print(DEVICE)

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Linear(256, self.config.NUM_CLASSES)
        )
        if os.path.exists(self.config.MODEL_PATH):
            model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE))
        model.to(self.config.DEVICE)
        return model

    def get_embedding_model(self):
        model = self.model
        class DenseNetEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = model.features

            def forward(self, x):
                features = self.features(x)
                out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                return torch.flatten(out, 1)
        return DenseNetEmbedding().to(self.config.DEVICE)

class Visualizer:
    @staticmethod
    def save_images(images, titles, prefix, directory="retrival_files/results/", figsize=(15, 10)):
        os.makedirs(directory, exist_ok=True)
        for i, image in enumerate(images):
            plt.figure(figsize=figsize)
            if torch.is_tensor(image):
                image = transforms.ToPILImage()(image)
            plt.imshow(image)
            plt.title(titles[i])
            plt.axis('off')
            filename = os.path.join(directory, f"{prefix}_{titles[i].replace(' ', '_').lower()}.png")
            plt.savefig(filename)
            plt.close()

def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            emb = model(images)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

def apply_pca(embeddings):
    pca = PCA(n_components=Config.N_COMPONENTS)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings, pca

def calculate_cosine_similarity(query_embedding, embeddings):
    return cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()

def calculate_mahalanobis_distances(query_embedding, embeddings, inv_covariance_matrix):
    distances = [mahalanobis(query_embedding, emb, inv_covariance_matrix) for emb in embeddings]
    return np.array(distances)

def compute_euclidean_distances(query_embedding, embeddings):
    distances = [euclidean(query_embedding, emb) for emb in embeddings]
    return np.array(distances)

def main(image_path):
    config = Config()
    model_handler = ModelHandler(config)
    embedding_model = model_handler.get_embedding_model()
    dataset = FreiburgDataset(split='test', transform=transforms.Compose([
                           transforms.Resize((256, 256)),
                           transforms.ToTensor()]), data_dir=config.DATA_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    embeddings = extract_embeddings(dataloader, embedding_model)
    #pca_embeddings, pca = apply_pca(embeddings)
    pca_embeddings = embeddings   
    query_image = Image.open(image_path).convert('RGB')
    query_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    query_vector = query_transform(query_image).unsqueeze(0).to(config.DEVICE)
    query_embedding = embedding_model(query_vector).detach().cpu().numpy()
    #query_embedding_pca = pca.transform(query_embedding)  # Apply PCA to the query embedding if using PCA embeddings
    query_embedding_pca = query_embedding
    cosine_scores = calculate_cosine_similarity(query_embedding_pca, pca_embeddings)
    covariance_matrix = np.cov(pca_embeddings.T)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    mahalanobis_distances = calculate_mahalanobis_distances(query_embedding_pca.flatten(), pca_embeddings, inv_covariance_matrix)
    euclidean_distances = compute_euclidean_distances(query_embedding_pca.flatten(), pca_embeddings)
    print("Max Cosine Score:")
    print(max(cosine_scores))
    print("Min euclidean_distances:")
    print(min(euclidean_distances))
    print("Min mahalanobis_distances:")
    print(min(mahalanobis_distances))

    for distance_name, distances in zip(["cosine", "mahalanobis", "euclidean"], 
                                        [cosine_scores, mahalanobis_distances, euclidean_distances]):
        if distance_name == "cosine":
            top_indices = np.argsort(distances)[-5:]  # For cosine similarity, higher scores are better
        else:
            top_indices = np.argsort(distances)[:5]  # For other distances, lower scores are better
        top_images = [dataset[idx][0] for idx in top_indices]
        titles = [f"{distance_name.capitalize()} Top {i+1}" for i in range(5)]
        Visualizer.save_images(top_images, titles, prefix=distance_name, directory="retrival_files/results/")

if __name__ == "__main__":
    main()
