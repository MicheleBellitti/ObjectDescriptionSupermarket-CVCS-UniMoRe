## --objectdetect -> bbox inference (bboxes on input scene photo)
#SceneImage -> FRCNN(Infer) -> List(id, bboxes)

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def object_detection(image_path):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Extracting boxes and labels
    bboxes = prediction[0]['boxes'].tolist()
    labels = prediction[0]['labels'].tolist()
    return list(zip(labels, bboxes))

## –shelf -> no bboxes according to shelves (numbered bboxes on input scene photo)
# SceneImage -> clustering (KNN+silhouette_scores -> List (id, bboxes, shelf number)

# Define the weight for the Y positions (e.g., 2 for double weight)
#y_weight = 3.5
y_weight = 50

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Path to the annotations file and images folder (update these paths if needed)
annotations_file = '/content/SKU110K_fixed/annotations/annotations_val.csv'
images_folder = '/content/SKU110K_fixed/images/'

# Load annotations
annotations_df = pd.read_csv(annotations_file)

# Assuming the first column contains the image file names
image_column = annotations_df.columns[0]

# Get a list of unique image file names from the annotations file
unique_image_files = annotations_df[image_column].unique()

# Select a random image from the annotations
random_image_name = random.choice(unique_image_files)
#random_image_name = "val_25.jpg" #example
#random_image_name = "val_554.jpg" #example
#random_image_name = "val_365.jpg" #example

print(random_image_name)

image_path = os.path.join(images_folder, random_image_name)

# Get bounding boxes for the selected image
bounding_boxes = annotations_df[annotations_df[image_column] == random_image_name]

# Open the image
image = Image.open(image_path)
img_width, img_height = image.size

# Collecting all midpoints
midpoints = []

# Add bounding boxes and their midpoints
for index, row in bounding_boxes.iterrows():
    # Scaling the bounding box coordinates
    x_min = row[1] * (img_width / row[6])
    y_min = row[2] * (img_height / row[7])
    x_max = row[3] * (img_width / row[6])
    y_max = row[4] * (img_height / row[7])

    # Calculate midpoint
    midpoint_x = (x_min + x_max) / 2
    midpoint_y = (y_min + y_max) / 2
    midpoints.append((midpoint_x, midpoint_y))

# Convert midpoints to a NumPy array for clustering
midpoints_array = np.array(midpoints)

# Create a copy of midpoints for clustering with weighted Y positions
midpoints_array_scaled = midpoints_array.copy()
midpoints_array_scaled[:, 1] *= y_weight  # Apply the weight to Y positions only

silhouette_scores = []
K = range(4, 15)  # Silhouette score is only defined if the number of labels is 2 <= n_labels <= n_samples - 1.
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(midpoints_array_scaled)
    score = silhouette_score(midpoints_array_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.title('Silhouette Analysis For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Find the index of the maximum silhouette score
max_score_index = silhouette_scores.index(max(silhouette_scores))
# The optimal k is the index of the maximum score plus 2 (since range starts at 2)
optimal_clusters = K[max_score_index]
# Print the optimal number of clusters
print("Optimal Number of Clusters:", optimal_clusters)

# Assume the optimal number of clusters is found to be 3 for this example
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10).fit(midpoints_array_scaled)
labels = kmeans.labels_

# Prepare plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting original image with bounding boxes and midpoints
axs[0].imshow(image)
axs[0].set_title('Image with Bounding Boxes and Midpoints')

# Plotting only midpoints on a white background
axs[1].set_xlim([0, img_width])
axs[1].set_ylim([img_height, 0])  # Inverted Y-axis for correct orientation
axs[1].set_title('Midpoints on White Background')
axs[1].set_facecolor('white')

# Add bounding boxes, their midpoints to the first plot, and only midpoints to the second plot
for index, row in bounding_boxes.iterrows():
    # Scaling the bounding box coordinates
    x_min = row[1] * (img_width / row[6])
    y_min = row[2] * (img_height / row[7])
    x_max = row[3] * (img_width / row[6])
    y_max = row[4] * (img_height / row[7])

    # Calculate midpoint
    midpoint_x = (x_min + x_max) / 2
    midpoint_y = (y_min + y_max) / 2

    # Create a Rectangle patch for the first plot
    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)

    # Plot the midpoint on both plots
    axs[0].plot(midpoint_x, midpoint_y, 'bo')  # Blue circle marker
    axs[1].plot(midpoint_x, midpoint_y, 'bo')  # Blue circle marker

plt.show()

sorted_cluster_centers = np.argsort(kmeans.cluster_centers_[:, 1])

# Initialize an array to hold the shelf label for each midpoint
shelf_labels = np.zeros_like(labels)

# Assign shelf numbers based on the sorted cluster indices
for shelf_number, cluster_index in enumerate(sorted_cluster_centers, start=1):
    # Find all midpoints belonging to the current cluster
    midpoints_in_cluster = labels == cluster_index
    # Assign the shelf number to those midpoints
    shelf_labels[midpoints_in_cluster] = shelf_number
# Now `shelf_labels` contains the shelf number for each midpoint

# Create a final array containing product_id, x1, x2, y1, y2, x_center, y_center, and shelf numbers
final_array = []

# Initialize product_id to start from 1
product_id = 1

# Loop through bounding boxes and midpoints
for label, (_, row), midpoint in zip(shelf_labels, bounding_boxes.iterrows(), midpoints):
    x_min = row[1] * (img_width / row[6])
    y_min = row[2] * (img_height / row[7])
    x_max = row[3] * (img_width / row[6])
    y_max = row[4] * (img_height / row[7])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    final_array.append([product_id, x_min, x_max, y_min, y_max, x_center, y_center, label])

    # Increment product_id for the next bounding box
    product_id += 1

# Now `final_array` contains [product_id, x1, x2, y1, y2, x_center, y_center, label] for each bounding box

print(final_array)

def plot_overlay_with_shelf_numbers(image_path, final_array):
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Create a large figure
    plt.figure(figsize=(18, 12))

    # Display the image
    plt.imshow(image)
    plt.title('Image with Overlayed Shelf Numbers', fontsize=20)

    # Iterate over items in final_array
    for item in final_array:
        product_id, x_min, x_max, y_min, y_max, x_center, y_center, shelf = item

        # Draw bounding boxes
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        # Draw a blue dot at the center point
        plt.plot(x_center, y_center, 'bo', markersize=8)

        # Prepare product information text
        #product_info = f"{int(product_id)}\n{int(shelf)}\n"
        product_info = f"{int(product_id)}"
        #product_info = f"{int(shelf)}"
        # Overlay shelf numbers and product information on the image
        plt.text(x_min, y_min - 10, product_info, color='red', fontsize=7, weight='bold')
        plt.text(x_min, y_min + 50, str(int(shelf)), color='red', fontsize=16, weight='bold')

    # Remove the axis and show the image
    plt.axis('off')
    plt.show()


# Call the function to plot the image with overlayed shelf numbers, blue dots, and product information
plot_overlay_with_shelf_numbers(image_path, final_array)

## --product -> product id mentioned on boxes (id-ed bboxes on input scene photo)
# SceneImage -> Crop on every id -> DenseNet121(Infer) -> List (id, bboxes, shelf number, product name, color(?)) (maybe the color)???
# **: Different dataset

from torchvision import models, transforms
import torch.nn as nn
import cv2

def product_identification(image_path, bboxes):
    # Load DenseNet121 with a custom final layer for your product dataset
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_product_classes) # num_product_classes needs to be defined
    model.eval()
    
    image = cv2.imread(image_path)
    products = []
    for idx, (label, box) in enumerate(bboxes):
        crop_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # Transform and classify cropped image
        # Assuming you have a transform composed for your model
        crop_tensor = transform(crop_img).unsqueeze(0) # transform needs to be defined
        
        with torch.no_grad():
            outputs = model(crop_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Assuming you have a function to map predicted class to product name and possibly detect color
            product_name = map_class_to_product_name(predicted.item())
            color = detect_color(crop_img) # detect_color needs to be implemented
            
            products.append((idx, box, label, product_name, color))
    return products

# –captioning -> caption it (Scene description from the input scene photo)
# List (id, bboxes, shelf number, product name, color(?) => Magic (LLM/Rule-based templating) => Scene Description

def generate_caption(products):
    # Simple rule-based example
    descriptions = []
    for product in products:
        descriptions.append(f"Product {product[3]} of color {product[4]} on shelf {product[2]}.")
    return " ".join(descriptions)

# –retrival -> find query image inside the scene image (give probable matches)
# Query + Scene (Crop on every id) => Embeddings Matching => Probable Suspects
# Query Image => DenseNet121(Infer) => Match with scene’s List (id, bboxes, shelf number, product name, color(?)) 
# Use both to remove false positives

def image_retrieval(scene_products, query_image_path):
    # Load the query image, process it with DenseNet121, and get its embedding
    query_embedding = get_image_embedding(query_image_path) # get_image_embedding needs to be implemented
    
    # Assuming you have a way to get embeddings for each product in the scene
    probable_matches = []
    for product in scene_products:
        product_embedding = get_product_embedding(product) # get_product_embedding needs to be implemented
        similarity = compute_similarity(query_embedding, product_embedding) # compute_similarity needs to be implemented
        
        if similarity > some_threshold: # some_threshold needs to be defined
            probable_matches.append(product)
    return probable_matches

import argparse
from pathlib import Path

# Assuming all the previously defined functions are available in this script or imported

def main():
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('--objectdetect', action='store_true',
                        help='perform bbox inference')
    parser.add_argument('--shelf', action='store_true',
                        help='no bboxes according to shelves')
    parser.add_argument('--product', action='store_true',
                        help='product id mentioned on boxes')
    parser.add_argument('--captioning', action='store_true',
                        help='caption it')
    parser.add_argument('--retrieval', action='store_true',
                        help='find query image inside the scene image')
    parser.add_argument('image_path', type=str,
                        help='path to the scene image')
    parser.add_argument('--query_image_path', type=str, default='',
                        help='path to the query image for retrieval')
    
    args = parser.parse_args()

    # Load the image and preprocess if necessary
    # For the sake of the example, we're assuming image_path is directly usable for each function
    if args.objectdetect:
        results = object_detection(args.image_path)
        print("Object Detection Results:", results)
    elif args.shelf:
        # Shelf detection would require object detection to run first to get bboxes
        object_detection_results = object_detection(args.image_path)
        results = shelf_detection([bbox for _, bbox in object_detection_results])
        print("Shelf Detection Results:", results)
    elif args.product:
        # Product identification would also typically require object detection results
        object_detection_results = object_detection(args.image_path)
        results = product_identification(args.image_path, object_detection_results)
        print("Product Identification Results:", results)
    elif args.captioning:
        # Assuming captioning needs product information
        object_detection_results = object_detection(args.image_path)
        product_results = product_identification(args.image_path, object_detection_results)
        caption = generate_caption(product_results)
        print("Caption:", caption)
    elif args.retrieval:
        if not args.query_image_path:
            raise ValueError("Query image path is required for retrieval.")
        object_detection_results = object_detection(args.image_path)
        product_results = product_identification(args.image_path, object_detection_results)
        matches = image_retrieval(product_results, args.query_image_path)
        print("Retrieval Matches:", matches)
    else:
        print("Please specify an operation to perform.")

if __name__ == '__main__':
    main()