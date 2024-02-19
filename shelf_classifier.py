import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ShelfClassifier:
    def __init__(self, annotations_file, images_folder):
        self.annotations_file = annotations_file
        self.images_folder = images_folder
        self.annotations_df = pd.read_csv(self.annotations_file)
        self.image_column = self.annotations_df.columns[0]
        self.unique_image_files = self.annotations_df[self.image_column].unique()

    def select_random_image(self):
        random_image_name = random.choice(self.unique_image_files)
        print(random_image_name)
        return random_image_name

    def get_bounding_boxes(self, image_name):
        image_path = os.path.join(self.images_folder, image_name)
        bounding_boxes = self.annotations_df[self.annotations_df[self.image_column] == image_name]
        return image_path, bounding_boxes

    def process_image(self, image_path, bounding_boxes):
        image = Image.open(image_path)
        img_width, img_height = image.size

        midpoints = []
        for index, row in bounding_boxes.iterrows():
            x_min = row[1] * (img_width / row[6])
            y_min = row[2] * (img_height / row[7])
            x_max = row[3] * (img_width / row[6])
            y_max = row[4] * (img_height / row[7])

            midpoint_x = (x_min + x_max) / 2
            midpoint_y = (y_min + y_max) / 2
            midpoints.append((midpoint_x, midpoint_y))

        return np.array(midpoints), img_width, img_height

    def find_optimal_shelf(self, midpoints):
        y_weight = 50
        midpoints_scaled = midpoints.copy()
        midpoints_scaled[:, 1] *= y_weight

        silhouette_scores = []
        K = range(4, 15)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(midpoints_scaled)
            score = silhouette_score(midpoints_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        max_score_index = silhouette_scores.index(max(silhouette_scores))
        optimal_clusters = K[max_score_index]
        print("Optimal Number of Clusters:", optimal_clusters)

        return KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10).fit(midpoints_scaled), optimal_clusters

    def assign_shelf_labels(self, kmeans, midpoints):
        labels = kmeans.labels_
        sorted_cluster_centers = np.argsort(kmeans.cluster_centers_[:, 1])

        shelf_labels = np.zeros_like(labels)
        for shelf_number, cluster_index in enumerate(sorted_cluster_centers, start=1):
            midpoints_in_cluster = labels == cluster_index
            shelf_labels[midpoints_in_cluster] = shelf_number

        return shelf_labels

    def create_final_array(self, bounding_boxes, midpoints, shelf_labels, img_width, img_height):
        final_array = []
        product_id = 1
        for label, (_, row), midpoint in zip(shelf_labels, bounding_boxes.iterrows(), midpoints):
            x_min = row[1] * (img_width / row[6])
            y_min = row[2] * (img_height / row[7])
            x_max = row[3] * (img_width / row[6])
            y_max = row[4] * (img_height / row[7])

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            final_array.append([product_id, x_min, x_max, y_min, y_max, x_center, y_center, label])
            product_id += 1

        return final_array

    def optimize_shelf_placement(self, image_name):
        image_path, bounding_boxes = self.get_bounding_boxes(image_name)
        midpoints, img_width, img_height = self.process_image(image_path, bounding_boxes)
        kmeans, optimal_clusters = self.find_optimal_shelf(midpoints)
        shelf_labels = self.assign_shelf_labels(kmeans, midpoints)
        final_array = self.create_final_array(bounding_boxes, midpoints, shelf_labels, img_width, img_height)
        return optimal_clusters, final_array