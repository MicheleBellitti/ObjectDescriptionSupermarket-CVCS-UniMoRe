import torch
import cv2
import numpy as np
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import torchvision.transforms as transforms
import torchvision.ops as ops
from PIL import Image
from homography import HomographyTransform
from colors import Color
from models import CustomRetinaNet


class ObjectCounter:
    """
    A class to count objects in an image using different detection models.

    Attributes:
        device (torch.device): The device to run the model on (CPU/GPU).
        trained_model (torch.nn.Module): The loaded trained model.
        model_type (str): Type of the model to be used ('ssd', 'frcnn', 'retinanet').

    Methods:
        count_objects_and_relations(image_path): Counts objects and identifies their spatial relationships.
        _load_models(trained_model_path): Loads the specified model from the given path.
        _preprocess_image(image_pil): Preprocesses the image for the model.
        _identify_spatial_relationships(bounding_boxes): Identifies spatial relationships between objects.
        _get_spatial_relationship(bbox1, bbox2): Calculates the spatial relationship between two bounding boxes.
    """

    def __init__(self, trained_model_path, model_type='ssd'):
        """
        Initializes the ObjectCounter with the specified model.

        Args:
            trained_model_path (str): Path to the trained model file.
            model_type (str): Type of the model to be used ('ssd', 'frcnn', 'CustomRetinaNet').
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self._load_models(trained_model_path)

    def _load_models(self, trained_model_path):
        """
        Loads the specified model from the given path.

        Args:
            trained_model_path (str): Path to the trained model file.
        """
        if self.model_type == 'ssd':
            self.trained_model = ssdlite320_mobilenet_v3_large(pretrained=True)
            in_channels = det_utils.retrieve_out_channels(
                self.trained_model.backbone, (320, 320))
            num_anchors = self.trained_model.anchor_generator.num_anchors_per_location()
            def norm_layer(x): return torch.nn.BatchNorm2d(
                x, eps=0.001, momentum=0.03)
            dropout = torch.nn.Dropout(p=0.3)
            self.trained_model.head.classification_head = torch.nn.Sequential(
                SSDLiteClassificationHead(
                    in_channels, num_anchors, 2, norm_layer),
                dropout
            )
        elif self.model_type == 'frcnn':
            self.trained_model = fasterrcnn_resnet50_fpn(pretrained=True)
            # Additional setup for Faster R-CNN if needed
        elif self.model_type == 'retinanet':
            # Load CustomRetinaNet model
            # Replace with actual model initialization
            self.trained_model = CustomRetinaNet(num_classes=2, pretrained_backbone=True)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        checkpoint = torch.load(trained_model_path)
        self.trained_model.load_state_dict(checkpoint["model_state_dict"])
        self.trained_model.to(self.device).eval()
        print(f"{self.model_type} model correctly loaded!")

    def count_objects_and_relations(self, image_path):
        """
        Counts objects in the given image and identifies their spatial relationships.

        Args:
            image_path (str): Path to the image file.

        Returns:
            int: Number of detected objects.
            list: List of spatial relationships between detected objects.
            list: List of colors of detected objects.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_path}")

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self._preprocess_image(image_pil).to(self.device)

        with torch.no_grad():
            prediction = self.trained_model(image_tensor)
            bounding_boxes = prediction[0]['boxes'].cpu().detach().numpy()
            scores = prediction[0]['scores'].cpu().detach().numpy()

        keep_boxes = ops.nms(torch.from_numpy(
            bounding_boxes), torch.from_numpy(scores), 0.9)
        bounding_boxes = bounding_boxes[keep_boxes]
        scores = scores[keep_boxes]
        bounding_boxes = bounding_boxes[scores >= 0.6]

        relationships = self._identify_spatial_relationships(bounding_boxes)
        color = Color()
        colors = [color.find_closest_color(color.get_dominant_color(
            image[y_min:y_max, x_min:x_max])) for x_min, y_min, x_max, y_max in bounding_boxes]

        return len(bounding_boxes), relationships, colors

    def _preprocess_image(self, image_pil):
        """
        Preprocesses the image for the model.

        Args:
            image_pil (PIL.Image): The image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop((320, 320)),
            transforms.ToTensor()
        ])
        return transform(image_pil).unsqueeze(0)

    def _identify_spatial_relationships(self, bounding_boxes):
        """
        Identifies spatial relationships between objects.

        Args:
            bounding_boxes (list): List of bounding boxes of detected objects.

        Returns:
            list: List of spatial relationships between objects.
        """
        relationships = []
        num_objects = len(bounding_boxes)

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                relationship = self._get_spatial_relationship(
                    bounding_boxes[i], bounding_boxes[j])
                relationships.append(
                    (f"Object {i + 1}", relationship, f"Object {j + 1}"))

        return relationships

    def _get_spatial_relationship(self, bbox1, bbox2):
        """
        Calculates the spatial relationship between two bounding boxes.

        Args:
            bbox1 (list): The first bounding box.
            bbox2 (list): The second bounding box.

        Returns:
            str: The spatial relationship between the two bounding boxes.
        """
        center1 = np.array(
            [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array(
            [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        vector = center2 - center1
        angle = np.degrees(np.arctan2(vector[1], vector[0]))

        if angle >= -45 and angle < 45:
            relationship = "to the right of"
        elif angle >= 45 and angle < 135:
            relationship = "below"
        elif angle >= -135 and angle < -45:
            relationship = "above"
        else:
            relationship = "to the left of"

        return relationship


# Example usage
if __name__ == "__main__":
    object_counter = ObjectCounter(
        "checkpoints/ssd/checkpoint.pth", model_type='ssd')
    count, relationships, colors = object_counter.count_objects_and_relations(
        "/work/cvcs_2023_group23/SKU110K_fixed/images/test_78.jpg")
    print(f"Detected {count} objects.")
    for rel in relationships:
        print(f"{rel[0]} is {rel[1]} {rel[2]}")
    for color in colors:
        print(color)
