import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes, save_image
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import random

from sklearn.metrics import silhouette_score
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
import webcolors
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
from scipy import stats
from collections import Counter
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.color import rgb2hsv, rgb2lab, rgb2ycbcr, lab2rgb
from skimage import color
from openai import OpenAI
from text_transformer import SceneDescriptionGenerator
import clip

NUM_TEST_IMAGES = 2939

scene_image_path = f"/work/cvcs_2023_group23/SKU110K_fixed/images/test_{random.randint(0,NUM_TEST_IMAGES)}.jpg"  # Update this path
frcnn_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/frcnn/checkpoint_230324_AdaBelief_Transforms_75epochs.pth"
densenet_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/clf_densetnet121/240325/40Epochs/last.ckpt"

# Configuration settings for the image retrieval system
class Config:
    #DEVICE = torch.device('cpu')  # Device configuration
    print(torch.cuda.is_available())
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration
    print(DEVICE)

def apply_hist_equalization(image):
    """
    Apply histogram equalization to an image.

    Args:
        image (PIL.Image): Image to process.

    Returns:
        image_eq (PIL.Image): Histogram equalized image.
    """
    image_np = np.array(image)  # Convert PIL image to NumPy array
    image_ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)  # Convert to YCbCr
    y, cb, cr = cv2.split(image_ycbcr)  # Split into components
    y_eq = cv2.equalizeHist(y)  # Equalize the Y channel
    image_eq_ycbcr = cv2.merge((y_eq, cb, cr))  # Merge back the channels
    image_eq = cv2.cvtColor(image_eq_ycbcr, cv2.COLOR_YCrCb2RGB)  # Convert to RGB
    image_eq = Image.fromarray(image_eq)  # Convert back to PIL Image
    print("Image Preprocessed with Histogram Equalization")
    return image_eq

def determine_shelf_numbers(bounding_boxes, img_width, img_height, y_weight=50):
    """
    Determine shelf numbers for each detected item based on their bounding box midpoints.
    """
    midpoints = []
    for index, row in bounding_boxes.iterrows():
        # Assuming row format: index, x_min, y_min, x_max, y_max, image_width_scale, image_height_scale
        x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        
        midpoint_x = (x_min + x_max) / 2
        midpoint_y = (y_min + y_max) / 2
        midpoints.append([midpoint_x, midpoint_y])

    midpoints_array = np.array(midpoints)
    midpoints_array_scaled = midpoints_array.copy()
    midpoints_array_scaled[:, 1] *= y_weight

    silhouette_scores = []
    K = range(4, min(15, len(midpoints) + 3))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(midpoints_array_scaled)
        score = silhouette_score(midpoints_array_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    max_score_index = silhouette_scores.index(max(silhouette_scores))
    optimal_clusters = K[max_score_index]

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(midpoints_array_scaled)
    labels = kmeans.labels_

    sorted_cluster_centers = np.argsort(kmeans.cluster_centers_[:, 1])
    shelf_labels = np.zeros_like(labels)

    for shelf_number, cluster_index in enumerate(sorted_cluster_centers, start=1):
        midpoints_in_cluster = labels == cluster_index
        shelf_labels[midpoints_in_cluster] = shelf_number
    print("Shelf Number Determined")
    return shelf_labels, kmeans

class GroceryStoreDataset(Dataset):

    # directory structure:
    # - root/[train/test/val]/[vegetable/fruit/packages]/[vegetables/fruit/packages]_class/[vegetables/fruit/packages]_subclass/[vegetables/fruit/packages]_image.jpg
    # - root/classes.csv
    # - root/train.txt
    # - root/test.txt
    # - root/val.txt
    def __init__(self, split='test', transform=None):
        super(GroceryStoreDataset, self).__init__()
        self.root = "/work/cvcs_2023_group23/GroceryStoreDataset/dataset/"
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._descriptions = {}

        classes_file = os.path.join(self.root, "classes.csv")

        self.classes = {'42': 'background'}
        with open(classes_file, "r") as f:

            lines = f.readlines()

            for line in lines[1:]:
                class_name, class_id, coarse_class_name, coarse_class_id, iconic_image_path, prod_description = line.strip().split(
                    ",")
                self.classes[class_id] = class_name
                self.class_to_idx[class_name] = coarse_class_id
                self.idx_to_class[class_id] = class_name
                self._descriptions[class_name] = prod_description

        self.samples = []
        split_file = os.path.join(self.root, self.split + ".txt")
        # print(self.classes)
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_path, class_id, coarse_class_id = line.split(",")
                class_name = self.classes[class_id.strip()]
                self.samples.append(
                    (os.path.join(self.root, img_path), int(self.class_to_idx[class_name])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        # print(img.shape, label)
        return img, label

    def description(self, class_name):
        return self._descriptions[class_name]

class LitModel(pl.LightningModule):
    def __init__(self, num_classes=43, learning_rate=1e-3):
        super().__init__()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features  # Get the number of features of the last layer
        self.model.classifier = nn.Linear(num_ftrs, num_classes)  # Update classifier
                
        # Make sure the classifier parameters are set to require gradients
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss)

        # Compute top-k accuracies
        top1_acc = self.compute_topk_accuracy(outputs, labels, k=1)
        top3_acc = self.compute_topk_accuracy(outputs, labels, k=3)
        top5_acc = self.compute_topk_accuracy(outputs, labels, k=5)
        self.log_dict({'test_top1_acc': top1_acc, 'test_top3_acc': top3_acc, 'test_top5_acc': top5_acc})

        # Add preds and labels to output
        _, preds = torch.max(outputs, dim=1)
        return {'test_loss': loss, 'preds': preds, 'labels': labels, 'test_top1_acc': top1_acc, 'test_top3_acc': top3_acc, 'test_top5_acc': top5_acc}

    def compute_topk_accuracy(self, outputs, labels, k=1):
        _, top_k_predictions = outputs.topk(k, 1, True, True)
        top_k_correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
        top_k_correct_sum = top_k_correct.view(-1).float().sum(0)
        return top_k_correct_sum.mul_(100.0 / labels.size(0))

def spatial_rgb_color_clustering(image, n_clusters=3):
    """
    Perform color clustering on an image using RGB color space and spatial features (x, y coordinates).
    
    Args:
        image (PIL.Image): The cropped image of the object.
        n_clusters (int): Number of clusters to use in KMeans.
        
    Returns:
        tuple: The dominant color in RGB format.
    """
    # Convert image to NumPy array
    image_np = np.array(image)
    height, width, _ = image_np.shape
    
    # Create features using both RGB color values and position (x, y)
    X = np.zeros((width * height, 5))
    X[:, :3] = image_np.reshape((-1, 3))  # RGB color features
    X[:, 3] = np.tile(np.arange(width), height)  # x coordinate
    X[:, 4] = np.repeat(np.arange(height), width)  # y coordinate
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    dominant_cluster_index = np.argmax(np.bincount(kmeans.labels_))
    dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster_index][:3]

    # The color values are already in RGB, so we can return them directly
    return tuple(int(c) for c in dominant_color_rgb)

def create_color_tree():
    # Define CSS3 named colors directly
    css3_db = webcolors.CSS3_NAMES_TO_HEX
    names = []
    rgb_values = []
    for name, hex_value in css3_db.items():
        names.append(name)
        rgb_values.append(webcolors.hex_to_rgb(hex_value))
    # Create a KDTree for fast lookup of nearest RGB values
    tree = KDTree(rgb_values)
    return tree, names
color_tree, color_names = create_color_tree()

def closest_color(requested_color):
    # Convert requested_color to a NumPy array if it's not already
    requested_color_np = np.array(requested_color)
    # Query the KDTree to find the closest color name
    _, index = color_tree.query(requested_color_np)
    return color_names[index]

def extract_clip_keywords(image, model, preprocess, top_k=1):
    """
    Extract descriptive keywords from an image using the CLIP model,
    specifically targeting common supermarket products.
    
    Args:
        image (PIL.Image): The image to process.
        model: The pre-trained CLIP model.
        preprocess: The pre-processing function for CLIP.
        top_k (int): The number of top keywords to return.
        
    Returns:
        List[str]: A list of descriptive keywords.
    """
    image = preprocess(image).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        potential_keywords = [
        "fruit", "apple", "banana", "orange", "lemon", "pineapple", "avocado", "grapes", "kiwi", "mango", "peach", "pear", "watermelon",
        "vegetable", "asparagus", "beetroot", "bell pepper", "broccoli", "carrot", "cauliflower", "celery", "cucumber", "garlic", "kale", "lettuce", "mushroom", "onion", "potato", "spinach", "tomato", "zucchini",
        "dairy", "milk", "yogurt", "butter", "cheese", "ice cream",
        "beverages", "water", "coffee", "tea", "wine", "beer", "soda", "juice",
        "meat", "chicken breast", "ground beef", "steak", "pork chops", "bacon", "sausage", "salmon", "tuna", "shrimp", "crab",
        "rice", "quinoa", "barley", "pasta", "bread", "tortillas",
        "tomato sauce", "pickles", "olives", "jam", "peanut butter",
        "chips", "pretzels", "popcorn", "nuts", "granola bars", "cookies", "crackers",
        "frozen vegetables", "frozen meals", "frozen desserts",
        "flour", "sugar", "salt", "baking powder", "cinnamon", "nutmeg", "vanilla extract",
        "ketchup", "mustard", "mayonnaise", "vinegar", "olive oil",
        "laundry detergent", "dish soap", "paper towels", "toilet paper", "trash bags",
        "shampoo", "conditioner", "body wash", "toothpaste", "deodorant",
        "dog food", "cat food", "pet treats",
        "diapers", "baby wipes", "baby formula",
        "pain reliever", "medicine", "vitamins", "band-aids",
        "greeting cards", "batteries", "light bulbs"
        ]

        text_tokens = clip.tokenize(potential_keywords).to(Config.DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity as logits
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        # Convert tensor to full precision (float32) before calling topk
        similarity = similarity.float()
        
        # Get the top k highest values' indices
        top_probs, top_labels = similarity.cpu().topk(top_k, dim=-1)

        keywords = [potential_keywords[i] for i in top_labels[0]]
        
        return keywords

def describe_spatial_relationships(final_array):
    # Initialize a dictionary to hold descriptions for each product ID
    spatial_descriptions = {}
    # Loop through each item in the final array
    for i, item in enumerate(final_array):
        # List to hold relationships descriptions for the current item
        descriptions = []   
        # Compare with every other item in the array
        for j, other_item in enumerate(final_array):
            if i != j:  # Ensure not comparing an item with itself
                # Only consider items on the same shelf
                if item['shelf_number'] == other_item['shelf_number']:
                    # Determine the relative position (left or right)
                    direction = "left of" if item['x_center'] < other_item['x_center'] else "right of"
                    # Construct the description including color and product name
                    description = f"{direction} {other_item['color']} {other_item['product_name']} (ID: {other_item['product_id']})"
                    descriptions.append(description)
        # Join all descriptions for the current item into a single string
        spatial_descriptions[item['product_id']] = '; '.join(descriptions)
    return spatial_descriptions

def overlay(image_path, final_array, output_dir, font_size=10):
    """
    Overlays bounding boxes, scores, shelf numbers, and product IDs on the image and saves it,
    placing the text in the center of the bounding box.
    """
    # Load the image
    scene_image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(scene_image)

    # Set up the font for text overlay
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    for entry in final_array:
        # Extracting coordinates for drawing
        x1, y1, x2, y2 = entry['x1'], entry['y1'], entry['x2'], entry['y2']
        clip_keywords_str = ', '.join(entry.get('clip_keywords', []))
        # Draw the bounding box in red
        draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
        # Prepare the text to overlay
        # overlay_text = f"ID: {entry['product_id']} \nFRCNN: {entry.get('frcnn_confidence', 0):.2f} \nShelf: {entry['shelf_number']} \nName: {entry.get('product_name')} \nDenseNet: {entry.get('densenet_confidence', 0):.2f} \n{entry.get('color')} \n{clip_keywords_str}"
        # overlay_text = f"ID: {entry['product_id']} \nFRCNN: {entry.get('frcnn_confidence', 0):.2f} \nShelf: {entry['shelf_number']} \n{entry.get('color')} \n{clip_keywords_str}"
        overlay_text = f"ID: {entry['product_id']} \nFRCNN: {entry.get('frcnn_confidence', 0):.2f} \nShelf: {entry['shelf_number']} \n{clip_keywords_str}"
        # Measure text size to center it
        text_width, text_height = 0, 0
        for overlay_text_line in overlay_text.split('\n'):
            text_width += draw.textlength(overlay_text_line, font=font)
            text_height += font.size
        # Calculate the center position
        text_x = x1 + (x2 - x1 - text_width) / 2
        text_y = y1 + (y2 - y1 - text_height) / 2
        # Draw the text
        draw.text((text_x, text_y), overlay_text, fill="yellow", font=font)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output path
    base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{base_filename}_frcnn_output{file_extension}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the image
    scene_image.save(output_path)
    print(f"Overlayed Scene Image saved to {output_path}")

def main(scene_image_path, frcnn_checkpoint_path, densenet_checkpoint_path):
    print(scene_image_path)
    print("Starting FRCNN module")
    config = Config()
    # Load object detector for inference
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    # Load weights
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
    # detector transform
    detector_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    print("FRCNN loaded correctly")
    
    # Load and preprocess the image
    scene_image = Image.open(scene_image_path).convert('RGB')
    scene_image_eq = apply_hist_equalization(scene_image)  # Apply histogram equalization
    scene_image_tensor = transforms.functional.to_tensor(scene_image_eq)  # Convert equalized image to tensor
    scene_image_tensor_unsqueezed = scene_image_tensor.unsqueeze(0)  # Add batch dimension
    
    # FRCNN Inference
    with torch.no_grad():
        output = detector(scene_image_tensor_unsqueezed)[0]
    bboxes = output["boxes"].cpu().numpy()  # Convert to NumPy array and move to CPU
    scores = output['scores'].cpu().numpy()
    print("FRCNN inferred")

    # Convert detections to a DataFrame for processing
    img_width, img_height = scene_image_tensor.shape[-1], scene_image_tensor.shape[-2]
    detections_data = [{
        'x_min': bbox[0] * (img_width / 1024),  # Example scaling, adjust as necessary
        'y_min': bbox[1] * (img_height / 1024),
        'x_max': bbox[2] * (img_width / 1024),
        'y_max': bbox[3] * (img_height / 1024),
    } for bbox in bboxes]
    detections_dataframe = pd.DataFrame(detections_data)

    # Determine shelf numbers
    shelf_labels, _ = determine_shelf_numbers(detections_dataframe, img_width, img_height)

    # Final Array
    final_array = []
    for i, ((x1, y1, x2, y2), score, shelf_number) in enumerate(zip(bboxes, scores, shelf_labels)):
        # Calculate center coordinates
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        # Construct the entry for this detection
        detection_entry = {
            'product_id': i,  # Assigning a simple incremental ID
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'x_center': x_center,
            'y_center': y_center,
            'frcnn_confidence': score,  # Now including the detection score
            'shelf_number': shelf_number  # Shelf number determined earlier
        }
        # Append this entry to the final array
        final_array.append(detection_entry)
    # Sort final_array by y_center, then by x_center
    final_array.sort(key=lambda x: (x['shelf_number'], x['x_center']))
    product_id_counter = 0
    for entry in final_array:
        # Assign unique product IDs based on the counter
        entry['product_id'] = product_id_counter
        product_id_counter += 1
    
    
    # # Load the trained model
    # densenet_model = LitModel.load_from_checkpoint(densenet_checkpoint_path)
    # densenet_model.eval()
    # densenet_model.to(Config.DEVICE)  # Assuming you're using the Config class from your object detection code
    # print("DenseNet loaded correctly")

    # Define test dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    grocery_test_dataset = GroceryStoreDataset(split='test', transform=test_transform)

    # Load the CLIP model
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=Config.DEVICE)
    print("CLIP loaded correctly")

    #print("Looping through BBox: Color Detection, DenseNet121 Classifier & CLIP Zero Shot")
    print("Looping through BBox: CLIP Zero Shot")
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)  # Convert bbox coordinates to integers
        crop = scene_image_eq.crop((x1, y1, x2, y2))  # Crop using PIL
        
        # # Convert bbox coordinates to integers
        # x1, y1, x2, y2 = map(int, bbox)
        # # Calculate the midpoints of the bounding box
        # mid_x = (x1 + x2) // 2
        # mid_y = (y1 + y2) // 2
        # # Determine the size of the middle box you want to crop
        # # This will crop a box of size (box_width x box_height) centered at the midpoint
        # box_width = (x2 - x1) * 0.85
        # box_height = (y2 - y1) * 0.85
        # # Calculate the coordinates of the middle box
        # middle_box_x1 = max(x1, mid_x - box_width // 2)
        # middle_box_y1 = max(y1, mid_y - box_height // 2)
        # middle_box_x2 = min(x2, mid_x + box_width // 2)
        # middle_box_y2 = min(y2, mid_y + box_height // 2)
        # # Crop the middle box using PIL
        # middle_crop = scene_image_eq.crop((middle_box_x1, middle_box_y1, middle_box_x2, middle_box_y2)) 
        # # Get the most frequent color from the cropped image
        # dominant_color = spatial_rgb_color_clustering(middle_crop, n_clusters=3)
        # # Ensure dominant_color is formatted correctly (should already be, based on Step 1)
        # if not isinstance(dominant_color, tuple) or len(dominant_color) != 3:
        #     raise ValueError("Dominant color must be a tuple of 3 elements.")
        # # Now call closest_color with the correctly formatted dominant_color
        # closest_color_name = closest_color(dominant_color)

        # # Prepare crop for DenseNet121 inference
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize to match DenseNet121 input
        #     transforms.ToTensor(),
        # ])
        # crop_tensor = transform(crop).unsqueeze(0).to(Config.DEVICE)  # Add batch dimension and move to device
        # # DenseNet121 inference for class name with confidence
        # with torch.no_grad():
        #     output = densenet_model(crop_tensor)
        #     probabilities = torch.nn.functional.softmax(output, dim=1)
        #     _, predicted = torch.max(output, 1)  # Using raw output for prediction as before
        #     predicted_class_name = grocery_test_dataset.idx_to_class[str(predicted.item())]

        #     # Now, extracting confidence from probabilities using the predicted class index
        #     confidence = probabilities[0, predicted.item()].item()
        
        # Crop the image for CLIP keyword extraction
        crop_for_clip = scene_image_eq.crop((x1, y1, x2, y2))

        # Extract keywords with CLIP
        clip_keywords = extract_clip_keywords(crop_for_clip, clip_model, clip_preprocess, top_k=1)

        # Update the final_array entry for this detection with classifier information
        #final_array[i]['product_name'] = predicted_class_name
        final_array[i]['product_name'] = "None"
        # final_array[i]['densenet_confidence'] = confidence
        #final_array[i]['color'] = closest_color_name
        final_array[i]['color'] = "None"
        final_array[i]['clip_keywords'] = clip_keywords

    print("Colors and Products Identified\n")
    print(final_array)

    # Generate spatial descriptions
    print("Generating Templated Spatial Descriptions")
    spatial_descriptions = describe_spatial_relationships(final_array)
    combined_descriptions = "\n".join(f"Product ID {pid}: {desc}" for pid, desc in spatial_descriptions.items())
    
    # Print templated spatial descriptions
    print("\nTemplated Spatial Descriptions:\n", combined_descriptions)
    
    # Generate a summary of the spatial descriptions
    print("\nGenerating Concise Scene Description with GPT")
    generator = SceneDescriptionGenerator(model_name="gpt-3.5-turbo")
    # concise_summary = generate_gpt_summary(combined_descriptions, final_array)
    try:
        concise_summary = generator.generate_description(combined_descriptions, final_array)
       
    except Exception as e:
        print(e)
        concise_summary = "An issue occurred with API call..\n"
    # Print the concise scene description generated by GPT-3.5/4
    print("\nConcise Scene Description Generated by GPT:\n", concise_summary)

    # Overlay boxes, scores, and shelf numbers
    output_dir = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/inference/"
    overlay(scene_image_path, final_array, output_dir)
    print("Scene Image Overlayed")

if __name__ == "__main__":
    main(scene_image_path, frcnn_checkpoint_path, densenet_checkpoint_path)