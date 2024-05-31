import os
import random
from datetime import datetime
from typing import List, Tuple, Union
import numpy as np
from scipy.linalg import LinAlgError
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)

class Config:
    SCENE_DIR = "/work/cvcs_2023_group23/SKU110K_fixed/images"
    #SCENE_DIR = "retrival_files/input_scene" #Example
    QUERY_DIR = "retrival_files/images"
    #QUERY_DIR = "retrival_files/input_query" #Example
    frcnn_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/frcnn/checkpoint_230324_AdaBelief_Transforms_75epochs.pth"
    MODEL_PATH = "checkpoints/clf_densetnet121/240325/40Epochs/last.ckpt"
    NUM_CLASSES = 43
    N_COMPONENTS = 50
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelHandler:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = models.densenet121(weights="DenseNet121_Weights.DEFAULT")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, self.config.NUM_CLASSES)
        
        if os.path.exists(self.config.MODEL_PATH):
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE)
            state_dict = self._strip_prefix_from_state_dict(checkpoint['state_dict'], "model.")
            model.load_state_dict(state_dict)
        model.to(self.config.DEVICE)
        return model

    @staticmethod
    def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
        new_state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
        return new_state_dict

    def get_embedding_model(self) -> nn.Module:
        model = self.model

        class DenseNetEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = model.features

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.features(x)
                out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                return torch.flatten(out, 1)

        return DenseNetEmbedding().to(self.config.DEVICE)

class DataProcessor:
    def __init__(self, config: Config, model: nn.Module):
        self.config = config
        self.model = model

    def extract_embeddings_from_bboxes(self, t: transforms.Compose, image_input: Union[str, Image.Image], bboxes: np.ndarray) -> List[np.ndarray]:
        image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for bbox in bboxes:
                cropped_image = self._crop_image(image, bbox)
                cropped_tensor = t(cropped_image).unsqueeze(0).to(self.config.DEVICE)
                embedding = self.model(cropped_tensor).flatten()
                embeddings.append(embedding.cpu().numpy())
        return embeddings

    def extract_embeddings_from_image(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input
        self.model.eval()
        with torch.no_grad():
            image_tensor = transforms.functional.to_tensor(image).unsqueeze(0).to(self.config.DEVICE)
            embedding = self.model(image_tensor).flatten()
        return embedding.cpu().numpy()

    @staticmethod
    def _crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        return image.crop(bbox)

def extract_embeddings(dataloader, model: nn.Module) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            emb = model(images).flatten(start_dim=1)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

def apply_pca(embeddings: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=Config.N_COMPONENTS)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings, pca

def calculate_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    # Ensure that query_embedding is a 2D array
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    return cosine_similarity(query_embedding, embeddings).flatten()

def calculate_mahalanobis_distances(query_embedding: np.ndarray, embeddings: np.ndarray, inv_covariance_matrix: np.ndarray) -> np.ndarray:
    distances = []
    for emb in embeddings:
        try:
            distance = mahalanobis(query_embedding, emb, inv_covariance_matrix)
            distances.append(distance)
        except LinAlgError:
            distances.append(np.nan)
    return np.array(distances)

def compute_euclidean_distances(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    distances = [euclidean(query_embedding, emb) for emb in embeddings]
    return np.array(distances)

def calculate_inverse_covariance_matrix(embeddings: np.ndarray) -> np.ndarray:
    cov_matrix = np.cov(embeddings.T)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except LinAlgError:
        # Regularize the covariance matrix if it's not invertible
        print("Covariance matrix is not invertible, applying regularization.")
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-5
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    return inv_cov_matrix

def select_random_image_from_folder(folder_path: str) -> str:
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not image_files:
        raise ValueError(f"No images found in folder: {folder_path}")
    return os.path.join(folder_path, random.choice(image_files))

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

    @staticmethod
    def create_combined_image(base_directory, metric_results, scene_image, query_image, scene_image_with_bboxes):
        num_metrics = len(metric_results)
        fig, axs = plt.subplots(num_metrics + 1, 5, figsize=(10, 4 * (num_metrics + 1)))
        fig.suptitle("Top 5 Similar Products for Each Metric", fontsize=16)
        
        # Add scene, query, and scene with bboxes images in the first row
        axs[0, 1].imshow(scene_image)
        axs[0, 1].set_title("Scene Image", fontsize=10)
        axs[0, 1].axis('off')
        axs[0, 2].imshow(query_image)
        axs[0, 2].set_title("Query Image", fontsize=10)
        axs[0, 2].axis('off')
        axs[0, 3].imshow(scene_image_with_bboxes)
        axs[0, 3].set_title("Scene Image with BBoxes", fontsize=10)
        axs[0, 3].axis('off')
        for j in range(0, 5):
            if j != 1 and j != 2 and j != 3:
                axs[0, j].axis('off')

        for i, (metric, (images, titles, values)) in enumerate(metric_results.items()):
            for j in range(5):
                if torch.is_tensor(images[j]):
                    images[j] = transforms.ToPILImage()(images[j])
                axs[i + 1, j].imshow(images[j])
                axs[i + 1, j].set_title(f"{titles[j]}\nValue: {values[j]:.4f}" if not np.isnan(values[j]) else f"{titles[j]}\nValue: NaN", fontsize=10)
                axs[i + 1, j].axis('off')
        
        combined_image_path = os.path.join(base_directory, "combined_results.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
        plt.savefig(combined_image_path)
        plt.close()
        return combined_image_path

def main():
    config = Config()

    # Select random images from the specified folders
    scene_image_path = select_random_image_from_folder(config.SCENE_DIR)
    query_image_path = select_random_image_from_folder(config.QUERY_DIR)

    print(f"Selected scene image: {scene_image_path}")
    print(f"Selected query image: {query_image_path}")

    detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    checkpoint = torch.load(config.frcnn_checkpoint_path)

    torch.cuda.empty_cache()
    detector = detector.to(config.DEVICE)

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v

    detector.load_state_dict(new_state_dict)
    detector.eval()

    print("FRCNN loaded correctly")
    detector_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

    model_handler = ModelHandler(config)
    embedding_model = model_handler.get_embedding_model()
    dp = DataProcessor(config, embedding_model)

    # Process the scene image to get bounding boxes
    scene_image = Image.open(scene_image_path).convert('RGB')
    scene_image_tensor = transforms.functional.to_tensor(scene_image).unsqueeze(0).to(config.DEVICE)
    output = detector(scene_image_tensor)[0]
    bboxes = output["boxes"].cpu().detach().numpy()
    print("BBoxes obtained from scene image: ", len(bboxes))

    # Draw bounding boxes on the scene image
    scene_image_with_bboxes = scene_image.copy()
    draw = ImageDraw.Draw(scene_image_with_bboxes)
    for bbox in bboxes:
        draw.rectangle(bbox.tolist(), outline="red", width=25)

    # Extract embeddings for each detected product in the scene image
    embeddings = dp.extract_embeddings_from_bboxes(detector_transform, scene_image, bboxes)
    embeddings = np.array(embeddings)
    print("Embeddings obtained for scene image: ", len(embeddings))

    # Process the query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_embedding = dp.extract_embeddings_from_image(query_image)

    # Calculate similarity metrics
    cosine_scores = calculate_cosine_similarity(query_embedding, embeddings)
    top_cosine_indices = np.argsort(cosine_scores)[-5:][::-1]

    euclidean_distances = compute_euclidean_distances(query_embedding, embeddings)
    top_euclidean_indices = np.argsort(euclidean_distances)[:5]

    inv_cov_matrix = calculate_inverse_covariance_matrix(embeddings)
    mahalanobis_distances = calculate_mahalanobis_distances(query_embedding, embeddings, inv_cov_matrix)
    top_mahalanobis_indices = np.argsort(mahalanobis_distances)[:5]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_directory = f"retrival_files/results/{timestamp}"
    os.makedirs(result_directory, exist_ok=True)

    top_images_cosine = [DataProcessor._crop_image(scene_image, bboxes[i]) for i in top_cosine_indices]
    top_titles_cosine = [f"Cosine Similarity Top {i+1}" for i in range(5)]
    top_values_cosine = cosine_scores[top_cosine_indices]
    Visualizer.save_images(top_images_cosine, top_titles_cosine, prefix="cosine_similarity", directory=result_directory)

    top_images_euclidean = [DataProcessor._crop_image(scene_image, bboxes[i]) for i in top_euclidean_indices]
    top_titles_euclidean = [f"Euclidean Distance Top {i+1}" for i in range(5)]
    top_values_euclidean = euclidean_distances[top_euclidean_indices]
    Visualizer.save_images(top_images_euclidean, top_titles_euclidean, prefix="euclidean_distance", directory=result_directory)

    top_images_mahalanobis = [DataProcessor._crop_image(scene_image, bboxes[i]) for i in top_mahalanobis_indices]
    top_titles_mahalanobis = [f"Mahalanobis Distance Top {i+1}" for i in range(5)]
    top_values_mahalanobis = mahalanobis_distances[top_mahalanobis_indices]
    Visualizer.save_images(top_images_mahalanobis, top_titles_mahalanobis, prefix="mahalanobis_distance", directory=result_directory)

    metric_results = {
        "Cosine Similarity": (top_images_cosine, top_titles_cosine, top_values_cosine),
        "Euclidean Distance": (top_images_euclidean, top_titles_euclidean, top_values_euclidean),
        "Mahalanobis Distance": (top_images_mahalanobis, top_titles_mahalanobis, top_values_mahalanobis)
    }

    combined_image_path = Visualizer.create_combined_image(result_directory, metric_results, scene_image, query_image, scene_image_with_bboxes)

    print(f"Results saved in directory: {result_directory}")
    print(f"Combined image saved at: {combined_image_path}")

if __name__ == "__main__":
    main()