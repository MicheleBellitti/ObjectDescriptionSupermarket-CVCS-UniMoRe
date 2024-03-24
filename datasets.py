# create a custom dataset class for each dataset
import ast
import json
import os
import yaml
import numpy as np
from PIL import Image
import threading
import torch
from PIL import ImageDraw
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torchvision
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd

# create a custom dataset class for each dataset
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Define transforms
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize the image
    # transforms.RandomResizedCrop(800, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip **
    # transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip (optional)
    # squeutransforms.RandomRotation(15),  # Random rotation by +/- 15 degrees **
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Color jitter **
    # transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Random perspective **
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Gaussian blur with variable sigma
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((1920, 1080)),  # Resize the image
    # transforms.CenterCrop(1024),  # Center crop
    transforms.ToTensor(),  # Convert to tensor
    # transforms.Normalize(mean=mean, std=std)  # Normalize
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize the image
    # transforms.CenterCrop(1024),  # Center crop
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize
])


class GroceryStoreDataset(Dataset):

    # directory structure:
    # - root/[train/test/val]/[vegetable/fruit/packages]/[vegetables/fruit/packages]_class/[vegetables/fruit/packages]_subclass/[vegetables/fruit/packages]_image.jpg
    # - root/classes.csv
    # - root/train.txt
    # - root/test.txt
    # - root/val.txt
    def __init__(self, split='train', transform=None):
        super(GroceryStoreDataset, self).__init__()
        self.root = "Datasets/GroceryStoreDataset-1/dataset"
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes_file = os.path.join(self.root, "classes.csv")

        self.classes = {'81': 'background'}
        with open(classes_file, "r") as f:

            lines = f.readlines()

            for line in lines[1:]:
                class_name, class_id, coarse_class_name, coarse_class_id, iconic_image_path, prod_description = line.strip().split(
                    ",")
                self.classes[class_id] = class_name
                self.class_to_idx[class_name] = coarse_class_id
                self.idx_to_class[class_id] = class_name
                self._description = prod_description

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

    def description(self):
        return self._description


def collate_fn(batch):
    """
    Collate function that pads sequences to the same length.
    """

    inputs = [torch.clone(item[0]).detach() for item in batch]
    targets = [torch.clone(item[1]).detach() for item in batch]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    # print(inputs_padded.shape, targets_padded.shape)
    return inputs_padded, targets_padded


class FreiburgDataset(Dataset):
    def __init__(self, split, data_dir="/work/cvcs_2023_group23/images/", index=1, transform=None):
        self.data_dir = data_dir
        # split can be 'train', 'test', 'val' or 'full_<training or testing>' for the entire training set
        self.split = split
        self.index = index
        self.transform = transform
        self.targets = []
        self.image_labels = []
        self._load_data()

    def _load_data(self):
        # Iterate over the split files
        split_file = f"data/{self.split}{self.index}.txt" if 'full' not in self.split else f"data/{self.split}.txt"

        with open(split_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split()
                full_path = os.path.join(self.data_dir, image_path)
                self.image_labels.append((full_path, int(label)))
                self.targets.append(label)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path, label = self.image_labels[idx]
        image = Image.open(image_path).convert("RGB")
        tensor_image = ToTensor()(image)
        return tensor_image, label


class ShelvesDataset(Dataset):
    """class for loading the Shelves dataset for object detection"""

    # structure: root/[images/annotations]/[file_name].jpg/json

    def __init__(self, transform=None, max_num_boxes=10):
        super(ShelvesDataset, self).__init__()

        self.root = os.path.join(
            "Datasets", "Supermarket+shelves", "Supermarket shelves", "Supermarket shelves")
        self.transform = transform
        self.num_files = len(os.listdir(os.path.join(self.root, "images")))
        self.max_num_boxes = max_num_boxes
        # self.setup_labels()
        # self.normalize_annotations()

    def setup_labels(self):
        # for each image name, get the corresponding json file and generate a txt file with the following format:
        # class_id(0 or 1), x1, y1, x2, y2
        # where x1, y1, x2, y2 are the coordinates of the bounding box

        # get the list of image names
        img_path = os.path.join(self.root, "images")
        img_filenames = os.listdir(img_path)
        # get the list of json filenames
        annotation_path = os.path.join(self.root, "annotations")
        annotation_filenames = os.listdir(annotation_path)

        for img_filename in img_filenames:
            # get the corresponding json filename
            annotation_filename = img_filename + ".json"
            # open the json file
            with open(os.path.join(annotation_path, annotation_filename)) as f:
                annotations = json.load(f)

            # create a new txt file
            txt_filename = img_filename[:-3] + "txt"
            with open(os.path.join(annotation_path.replace("annotations", "labels"), txt_filename), "w") as f:
                # for each object in the image
                # get width and height of the image
                width = int(annotations["size"]["width"])
                height = int(annotations["size"]["height"])
                for obj in annotations["objects"]:
                    # get the class id
                    class_id = 1 if obj["classTitle"] == "product" else 0
                    # get the bounding box coordinates
                    x1 = float(obj["points"]["exterior"][0][0])
                    y1 = float(obj["points"]["exterior"][0][1])
                    x2 = float(obj["points"]["exterior"][1][0]) - x1
                    y2 = float(obj["points"]["exterior"][1][1]) - y1
                    # write the line in the txt file
                    f.write(
                        f"{class_id} {x1/width} {y1/height} {x2/width} {y2/height}\n")

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        torch.random.manual_seed(0)
        img_path = os.path.join(self.root, "images")
        img_filename = os.listdir(img_path)[idx]
        ret = [{'boxes': None, 'labels': [], 'image_id': torch.tensor([int(img_filename[:3])]),
                'area': torch.tensor([4000 * 4000]),
                'iscrowd': torch.tensor([[0, 0]])}]
        annotation_path = os.path.join(self.root, "annotations")
        annotation_filename = os.listdir(annotation_path)[idx]

        # read the image
        img = Image.open(os.path.join(img_path, img_filename)).convert('RGB')
        labels = []
        # Load the JSON annotation file
        with open(os.path.join(annotation_path, annotation_filename)) as f:
            annotatations = json.load(f)
        class_id_to_label = {}
        # Create an empty list to hold the padded bounding boxes
        padded_boxes = []
        # Iterate over the objects list
        for i, obj in enumerate(annotatations['objects']):
            # Extract the classId and the bounding box coordinates
            class_id = obj['classId']
            x1, y1 = obj['points']['exterior'][0]
            x2, y2 = obj['points']['exterior'][1]
            box = [x1, y1, x2, y2]
            padded_boxes.append(box)
            labels.append(0 if class_id == 10213293 else 1)

        # Pad the bounding boxes and labels to a fixed length
        padded_boxes = pad_boxes(padded_boxes, 100)
        labels = pad_labels(labels, 100)

        # return the image and the correspondent padded bounding boxes and labels
        if self.transform:
            img = self.transform(img)

        ret[0]['boxes'] = torch.tensor(padded_boxes)
        ret[0]['labels'] = torch.tensor(labels)
        return img, ret

    def normalize_annotations(self):
        # divide the xyxy coordinates by the image width and height
        annotation_path = os.path.join(self.root, "labels")
        annotation_filenames = os.listdir(annotation_path)

        for annotation_filename in annotation_filenames:
            with open(os.path.join(annotation_path, annotation_filename)) as f:
                lines = f.readlines()
            with open(os.path.join(annotation_path, annotation_filename), "w") as f:
                height, width = Image.open(os.path.join(
                    self.root, "images", annotation_filename[:-3] + "jpg")).size
                for line in lines:
                    class_id, x1, y1, x2, y2 = line.strip().split()
                    '''x1 = float(x1) / width
                    y1 = float(y1) / height
                    x2 = float(abs(x1 - int(x2))) / width
                    y2 = float(abs(y1 - int(y2))) / height
                    f.write(f"{class_id} {x1} {y1} {x2} {y2}\n")'''
                    print(x1, y1, x2, y2)


def pad_boxes(boxes_list, pad_length):
    # pad the list with zeros if its length is less than the pad_length
    if len(boxes_list) < pad_length:
        boxes_list += [[0, 0, 1, 1]] * (pad_length - len(boxes_list))
    # truncate the list if its length is greater than the pad_length
    elif len(boxes_list) > pad_length:
        boxes_list = boxes_list[:pad_length]
    return boxes_list


def pad_labels(labels, max_num_boxes):
    # Add zeros to the labels list until it has a length of max_num_boxes
    while len(labels) < max_num_boxes:
        labels.append(0)
    return labels


class SKUDataset(Dataset):

    def __init__(self, split='train', transform=None):
        self.root_dir = '/work/cvcs_2023_group23/SKU110K_fixed'
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.annotations_dir = os.path.join(self.root_dir, 'annotations')
        self.split = split
        self.transform = transform

        # Load annotations CSV file
        annotations_file = os.path.join(
            self.annotations_dir, f'annotations_{self.split}.csv')
        self.annotations_df = (pd.read_csv(
            annotations_file) if self.split == 'train' else pd.read_csv(annotations_file)[:50000])
        self.image_names = self.annotations_df.image_name.unique()
        # print(len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_annotations = []
        image = None

        # Get all rows for the specific image
        image_name = self.image_names[idx]
        image_rows = self.annotations_df[self.annotations_df['image_name'] == image_name]

        img_path = os.path.join(self.images_dir, image_name)

        rows = list(image_rows.iterrows())
        width, height = rows[0][1]["image_width"], rows[0][1]["image_height"]
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, IOError) as e:
            # print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (width, height), (0, 0, 0))
            error_log_path = 'corrupted_images.log'
            with open(error_log_path, 'a') as f:
                f.write(f"Corrupted image: {img_path}\n")

        for _, row in image_rows.iterrows():
            x1 = row['x1']
            y1 = row['y1']
            x2 = row['x2']
            y2 = row['y2']
            class_id = row['class']
            image_width = row['image_width']
            image_height = row['image_height']

            x1, x2 = min(x1, x2), max(x1, x2)

            y1, y2 = min(y1, y2), max(y1, y2)

            class_id = 1 if class_id == "object" else 0

            # Scale annotations according to the resized image
            x1 = x1 / image_width * 1024
            y1 = y1 / image_height * 1024
            x2 = x2 / image_width * 1024
            y2 = y2 / image_height * 1024

            # Append annotation to the list
            image_annotations.append(
                [x1, y1, x2, y2, class_id, image_width, image_height])

        # Convert annotations to tensors
        x1 = torch.tensor([annot[0] for annot in image_annotations])
        y1 = torch.tensor([annot[1] for annot in image_annotations])
        x2 = torch.tensor([annot[2] for annot in image_annotations])
        y2 = torch.tensor([annot[3] for annot in image_annotations])
        class_ids = torch.tensor([annot[4] for annot in image_annotations])
        image_widths = torch.tensor([annot[5] for annot in image_annotations])
        image_heights = torch.tensor([annot[6] for annot in image_annotations])

        # Apply transformation if available to the image
        if self.transform and image:
            image = self.transform(image)

        return image, x1, y1, x2, y2, class_ids, image_widths, image_heights


def custom_collate_fn(batch):
    """
    Collate function for SKUDataset.

    Args:
        batch: A batch of data from the SKUDataset.

    Returns:
        A batch of data that is ready to be passed to the model.
    """

    images, x1_tuple, y1_tuple, x2_tuple, y2_tuple, class_ids, image_widths, image_heights = zip(
        *batch)

    # Pad sequences to the maximum length in the batch
    x1_padded = pad_sequence(x1_tuple, batch_first=True, padding_value=0)
    y1_padded = pad_sequence(y1_tuple, batch_first=True, padding_value=0)
    x2_padded = pad_sequence(x2_tuple, batch_first=True, padding_value=1)
    y2_padded = pad_sequence(y2_tuple, batch_first=True, padding_value=1)
    class_ids = pad_sequence(class_ids, batch_first=True, padding_value=0)
    image_widths = pad_sequence(
        image_widths, batch_first=True, padding_value=320)
    image_heights = pad_sequence(
        image_heights, batch_first=True, padding_value=320)

    # Convert tensors to a torch.Tensor
    images = torch.stack(images)

    return images, x1_padded, y1_padded, x2_padded, y2_padded, class_ids, image_widths, image_heights


class SKUDatasetGPU(Dataset):

    def __init__(self, split, transform=None):
        self.root_dir = '/work/cvcs_2023_group23/SKU110K_fixed'
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.annotations_dir = os.path.join(self.root_dir, 'annotations')
        self.split = split
        self.transform = transform

        # Load annotations CSV file
        annotations_file = os.path.join(
            self.annotations_dir, f'annotations_{self.split}.csv')
        self.annotations_df = pd.read_csv(annotations_file, header=None)

        self.image_lock = threading.Lock()  # Add the lock

    def __len__(self):
        return len(self.annotations_df)

    def classes(self):
        return self.annotations_df.iloc[:, 5].unique()

    def num_classes(self):
        return self.classes().shape[0]

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        img_name = row[0]
        x1 = row[1]
        y1 = row[2]
        x2 = row[3]
        y2 = row[4]
        class_id = row[5]
        image_width = row[6]
        image_height = row[7]

        try:
            class_id = int(class_id)  # Convert class_id to an integer
        except ValueError:
            class_id = -1  # Set a default value for class_id

        # Load image
        img_path = os.path.join(self.images_dir, img_name)

        with self.image_lock:  # Use the lock around image loading
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a placeholder image and targets
                # Create a placeholder image
                image = Image.new('RGB', (256, 256))
                targets = {'boxes': torch.tensor(
                    [], dtype=torch.float32), 'labels': torch.tensor([], dtype=torch.int64)}
                return image, targets

        # Apply transformation if available
        if self.transform:
            image = self.transform(image)

        boxes = torch.tensor(
            [[x1, y1, x2, y2]], dtype=torch.float32) if class_id != -1 else torch.tensor([])
        labels = torch.tensor(
            [class_id], dtype=torch.int64) if class_id != -1 else torch.tensor([])

        targets = {'boxes': boxes, 'labels': labels}

        # print(f"Image shape: {image.shape}")
        # print(f"Targets: {targets}")

        return image, targets
