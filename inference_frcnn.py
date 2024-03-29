import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageDraw, ImageFont
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

NUM_TEST_IMAGES = 2939

scene_image_path = f"/work/cvcs_2023_group23/SKU110K_fixed/images/test_{random.randint(0,NUM_TEST_IMAGES)}.jpg"  # Update this path
frcnn_checkpoint_path = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/frcnn/checkpoint_230324_AdaBelief_Transforms_75epochs.pth"

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

def overlay(image_path, final_array, output_dir, font_size=15):
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
        
        # Draw the bounding box in red
        draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
        
        # Prepare the text to overlay
        score_text = f"Score: {entry['score']:.2f}" if 'score' in entry else ''
        overlay_text = f"ID: {entry['product_id']} \n{score_text} \nShelf: {entry['shelf_number']}"

        # Measure text size to center it
        text_width, text_height = draw.textsize(overlay_text, font=font)
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
    print(f"Image with boxes, scores, shelf numbers, and product IDs saved to {output_path}")

def main(scene_image_path, frcnn_checkpoint_path):
    
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
    print(scores)
    print("FRCNN inferred correctly")

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
            'score': score,  # Now including the detection score
            'shelf_number': shelf_number  # Shelf number determined earlier
        }
        # Append this entry to the final array
        final_array.append(detection_entry)
    # Sort final_array by y_center, then by x_center
    final_array.sort(key=lambda x: (x['shelf_number'], x['x_center']))
    print("Final detections sorted from top-left to bottom-right")
    product_id_counter = 0
    for entry in final_array:
        # Assign unique product IDs based on the counter
        entry['product_id'] = product_id_counter
        product_id_counter += 1
    print(final_array)

    # Overlay boxes, scores, and shelf numbers
    output_dir = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/inference/"
    overlay(scene_image_path, final_array, output_dir)

if __name__ == "__main__":
    main(scene_image_path, frcnn_checkpoint_path)