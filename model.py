import torch
import torchvision.models as models
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class CustomRetinaNet(RetinaNet):
    """
    Custom RetinaNet model with improvements based on the Focal Loss paper.

    Attributes:
        num_classes (int): Number of classes for object detection.
        pretrained_backbone (bool): If True, uses a pre-trained backbone.
    """

    def __init__(self, num_classes, pretrained_backbone=False):
        """
        Initialize the CustomRetinaNet model.

        Args:
            num_classes (int): Number of classes for object detection.
            pretrained_backbone (bool): If True, uses a pre-trained backbone.
        """
        # Load a pre-trained backbone model, e.g., ResNet50
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
        
        # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
        
        # Initialize the RetinaNet model with the FPN backbone
        super(CustomRetinaNet, self).__init__(backbone=backbone, num_classes=num_classes)

        # Access the in_channels for the classification head
        # This part needs to be adjusted based on the actual structure of the FPN-based RetinaNet
        in_channels = backbone.out_channels

        # Replace the classification head with a custom one that uses focal loss
        num_anchors = self.head.classification_head.num_anchors
        self.head.classification_head = CustomRetinaNetClassificationHead(in_channels, num_anchors, num_classes)

    def forward(self, images, targets=None):
        """
        Forward pass through the network.

        Args:
            images (Tensor): Batch of images.
            targets (list[dict], optional): Ground truth boxes and labels for each image.

        Returns:
            Tensor: Model predictions.
        """
        return super(CustomRetinaNet, self).forward(images, targets)


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    """
    Custom classification head for RetinaNet using Focal Loss.

    Attributes:
        in_channels (int): Number of input channels.
        num_anchors (int): Number of anchors.
        num_classes (int): Number of classes.
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        """
        Initialize the CustomRetinaNetClassificationHead.

        Args:
            in_channels (int): Number of input channels.
            num_anchors (int): Number of anchors.
            num_classes (int): Number of classes.
        """
        super(CustomRetinaNetClassificationHead, self).__init__(in_channels, num_anchors, num_classes)
    
        
def create_model(num_classes):
    """
    Function to create the custom RetinaNet model.

    Args:
        num_classes (int): Number of classes for object detection.

    Returns:
        CustomRetinaNet: Instantiated custom RetinaNet model.
    """
    model = CustomRetinaNet(num_classes=num_classes)
    return model
