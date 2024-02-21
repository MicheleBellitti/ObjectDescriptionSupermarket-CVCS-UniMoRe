import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from datasets import FreiburgDataset  # Assuming this is a custom module or replace with your dataset

# Configuration settings for the image retrieval system
class Config:
    DATA_DIR = "images/"  # Directory for dataset images
    MODEL_PATH = "freiburg/single/classifier.pth"  # Path to the trained model
    NUM_CLASSES = 25  # Number of classes in the dataset
    N_COMPONENTS = 50  # Number of components for PCA
    BATCH_SIZE = 32  # Batch size for data loading
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration

# Handles loading and modifying the model for feature extraction
class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        # Load a pre-trained DenseNet121 model and modify the classifier
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Linear(256, self.config.NUM_CLASSES)
        )
        # Load model weights if a model path is provided
        if os.path.exists(self.config.MODEL_PATH):
            model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE))
        model.to(self.config.DEVICE)
        return model

    def get_embedding_model(self):
        # Return a model that outputs embeddings from the features layer
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

    def extract_embeddings_from_bboxes(self, image_input, bboxes):
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
                cropped_tensor = transforms.functional.to_tensor(cropped_image).unsqueeze(0).to(self.config.DEVICE)
                embedding = self.model(cropped_tensor)
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

# Visualization utilities for displaying images
class Visualizer:
    @staticmethod
    def show_images(images, titles, figsize=(15, 10)):
        """
        Display a list of images with titles.

        :param images: A list of PIL Images.
        :param titles: A list of titles for each image.
        :param figsize: Figure size for the matplotlib plot.
        """
        plt.figure(figsize=figsize)
        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(image)
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

# Main workflow for the image retrieval system
def main():
    config = Config()
    model_handler = ModelHandler(config)
    embedding_model = model_handler.get_embedding_model()
    
    # Example usage with an image and bounding boxes
    image_path = "path/to/your/image.jpg"  # Or use a PIL Image directly
    bboxes = [[10, 20, 100, 200], [150, 250, 300, 400]]  # Example bounding boxes

    data_processor = DataProcessor(config, embedding_model)
    embeddings = data_processor.extract_embeddings_from_bboxes(image_path, bboxes)
    print("Extracted Embeddings:", embeddings)

    # Optionally, visualize the cropped images (for demonstration)
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    cropped_images = [data_processor._crop_image(image, bbox) for bbox in bboxes]
    Visualizer.show_images(cropped_images, [f"BBox {i+1}" for i in range(len(bboxes))])

if __name__ == "__main__":
    main()
