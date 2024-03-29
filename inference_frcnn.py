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

def overlay_boxes_and_scores_and_save(image_path, image_tensor, bboxes, scores, output_dir, font_size=30):
    """
    Overlays bounding boxes and scores on the image and saves it using torchvision utilities.
    
    Parameters:
    - image_path: Path to the original image file.
    - image_tensor: Tensor of the original image.
    - bboxes: Tensor of bounding boxes in the format (x1, y1, x2, y2).
    - scores: Numpy array or tensor of detection scores.
    - output_dir: Directory where the output image will be saved.
    - font_size: Size of the font for drawing text.
    """
    # Convert image tensor to uint8 if it's not already
    image_tensor = (image_tensor * 255).to(torch.uint8)
    
    # Ensure bboxes is a PyTorch tensor
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.tensor(bboxes, dtype=torch.float).to(image_tensor.device)
    
    # Draw bounding boxes on the image tensor
    drawn_image = draw_bounding_boxes(image_tensor, bboxes.int(), colors="red", width=10)
    
    # Convert the tensor back to PIL Image for drawing text
    drawn_pil = transforms.ToPILImage()(drawn_image).convert("RGB")
    draw = ImageDraw.Draw(drawn_pil)
    
    # Set font for drawing text. Adjust the path to the font file as needed.
    # For basic usage, you can use a default PIL font by not specifying the path.
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Custom font not found, using default.")

    # Ensure scores is a PyTorch tensor for consistency
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Draw scores next to the bounding boxes
    for box, score in zip(bboxes, scores):
        x, y, _, _ = box.tolist()
        score_text = f"{score:.2f}"
        draw.text((x, y), score_text, fill="yellow", font=font)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Modify here to add '_frcnn_output' postfix to the filename
    base_filename, file_extension = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{base_filename}_frcnn_output{file_extension}"
    output_path = os.path.join(output_dir, output_filename)
    drawn_pil.save(output_path)
    print(f"Image with boxes and scores saved to {output_path}")

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
    
    print("FRCNN loaded correctly")
    
    # detector transform
    detector_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    
    # Load and preprocess the image
    scene_image = Image.open(scene_image_path).convert('RGB')
    scene_image_eq = apply_hist_equalization(scene_image)  # Apply histogram equalization
    scene_image_tensor = transforms.functional.to_tensor(scene_image_eq)  # Convert equalized image to tensor
    scene_image_tensor_unsqueezed = scene_image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        output = detector(scene_image_tensor_unsqueezed)[0]
    bboxes = output["boxes"]
    scores = output['scores']

    print(scores)
    output_dir = "/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/inference/"
    overlay_boxes_and_scores_and_save(scene_image_path, scene_image_tensor, bboxes, scores, output_dir)
    print("FRCNN inferred correctly")

if __name__ == "__main__":
    main(scene_image_path, frcnn_checkpoint_path)