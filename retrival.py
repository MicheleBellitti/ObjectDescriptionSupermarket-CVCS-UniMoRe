import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis, euclidean
from matplotlib import pyplot as plt
from datasets import FreiburgDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

NUM_TEST_IMAGES = 2939

image_path = f"/work/cvcs_2023_group23/SKU110K_fixed/images/test_{random.randint(0,NUM_TEST_IMAGES)}.jpg"  # Random Scene Image

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
# Processes data, including loading, transformations, and embedding extraction
class DataProcessor:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def extract_embeddings_from_bboxes(self, t, image_input, bboxes):
        """
        Extract embeddings for each bounding box in the image.

        :param image_input: A PIL Image or a path to an image file.
        :param bboxes: A list of bounding boxes, each defined by [x1, y1, x2, y2].
        :return: A list of embeddings.
        """
        # Load image if a path is provided, otherwise use the PIL image directly
        image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for bbox in bboxes:
                cropped_image = self._crop_image(image, bbox)
                cropped_tensor = t(cropped_image).unsqueeze(0).to(self.config.DEVICE)
                embedding = self.model(cropped_tensor).reshape((1024, ))
                embeddings.append(embedding.cpu().numpy())
        return embeddings
    def extract_embeddings_from_image(self, image_input):
      """
      Extract embeddings for each bounding box in the image.

      :param image_input: A PIL Image or a path to an image file.
      :return: A list of embeddings.
      """
      # Load image if a path is provided, otherwise use the PIL image directly
      image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input
      embeddings = []
      self.model.eval()
      with torch.no_grad():

            image_tensor = transforms.functional.to_tensor(image).unsqueeze(0).to(self.config.DEVICE)
            embedding = self.model(image_tensor)
            embeddings.append(embedding.cpu().numpy())
      return embeddings
    @staticmethod
    def _crop_image(image, bbox):
        """
        Crop the image to the bounding box.

        :param image: A PIL Image.
        :param bbox: A bounding box defined by [x1, y1, x2, y2].
        :return: Cropped PIL Image.
        """
        return image.crop(bbox)
  

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
    # Load object detector for inference
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    # Load weights
    #frcnn_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/frcnn/checkpoint_230224_AdaBelief_Transforms_100epochs.pth"
    frcnn_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/frcnn/checkpoint_230324_AdaBelief_Transforms_75epochs.pth"
    checkpoint = torch.load(frcnn_checkpoint_path)
    
    # Setting up distributed processing
    torch.cuda.empty_cache()
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    
    detector = detector.to(local_rank)
    detector = DDP(detector, device_ids=[local_rank], output_device=local_rank)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval() # inference mode
    
    print("FRCNN loaded correctly")
    # detector transform
    detector_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    
    # Load embedding model
    model_handler = ModelHandler(config)
    embedding_model = model_handler.get_embedding_model()
    dp = DataProcessor(config, embedding_model)
    dataset = FreiburgDataset(split='test', transform=transforms.Compose([
                           transforms.Resize((256, 256)),
                           transforms.ToTensor()]), data_dir=config.DATA_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    embeddings = extract_embeddings(dataloader, embedding_model)
    
    # Inference on FRCNN
    query_image = Image.open(image_path).convert('RGB')
    query_image_tensor = transforms.functional.to_tensor(query_image).unsqueeze(0).to(config.DEVICE)
    output = detector(query_image_tensor)[0]
    bboxes = output["boxes"].cpu().detach().numpy()
    
    # embeddings = dp.extract_embeddings_from_bboxes(detector_transform, query_image, bboxes)
    # pca_embeddings, pca = apply_pca(embeddings)
    print("Embeddings obtained..", len(embeddings))
    pca_embeddings = np.array(embeddings) #No PCA 
    
    query_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # random bbox from query image
    random_box = bboxes[random.randint(0, bboxes.shape[0])]
    
    input_image = DataProcessor._crop_image(query_image, random_box)
    query_vector = query_transform(input_image).unsqueeze(0).to(config.DEVICE)
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
    
    Visualizer.save_images([input_image], ["query_box"], prefix="input", directory="retrieval_files/results")

    for distance_name, distances in zip(["cosine", "mahalanobis", "euclidean"], 
                                        [cosine_scores, mahalanobis_distances, euclidean_distances]):
        if distance_name == "cosine":
            top_indices = np.argsort(distances)[-5:]  # For cosine similarity, higher scores are better
        else:
            top_indices = np.argsort(distances)[:5]  # For other distances, lower scores are better
        top_images = [DataProcessor._crop_image(query_image, bboxes[top_i]) for top_i in top_indices]
        titles = [f"{distance_name.capitalize()} Top {i+1}" for i in range(5)]
        
        Visualizer.save_images(top_images, titles, prefix=distance_name, directory="retrival_files/results/")

if __name__ == "__main__":
    main(image_path)
