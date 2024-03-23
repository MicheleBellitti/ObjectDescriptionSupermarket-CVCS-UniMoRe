import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import yaml
import uuid
from datetime import date
from datasets import SKUDataset, TEST_TRANSFORM, custom_collate_fn
from colors import Color
from functools import partial
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# DistributedDataParallel wraps a model to enable distributed data parallel training.
# It synchronizes gradients across multiple GPUs/processes during the backward pass.
import torch.nn as nn
from model import CustomRetinaNet
from utils import calculate_mAP, find_jaccard_overlap

# Evaluate a trained object detection model on the test set.

# Loads the model checkpoint and runs inference on the test dataset. Computes evaluation metrics like mAP, plots distributions of IoU scores, dominant colors, etc. and visualizes predictions.

# Saves visualizations and metrics plots to the `train_results/<model_name>/<date>` directory.

# Supports SSD, Faster R-CNN and RetinaNet models. Can run distributed evaluation using multiple GPUs.

# Constants
CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.3

dominant_colors = []
color_names = []


def load_model(model_name, checkpoint_path, local_rank):
    """
    Load a model for evaluation.

    Args:
        model_name (str): The name of the model to load.
        checkpoint_path (str): The path to the checkpoint file.
        local_rank: The rank of the current process in distributed training.

    Returns:
        torch.nn.Module: The loaded model.

    Raises:
        ValueError: If the model name is unsupported.
    """

    if model_name == "ssd":
        model = _build_ssd_model(num_classes=2)
    elif model_name == "frcnn":
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    elif model_name == "retinanet":
        model = CustomRetinaNet(num_classes=2, pretrained_backbone=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    checkpoint = torch.load(checkpoint_path)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def _build_ssd_model(num_classes):
    """Build an SSD model for object detection.

    Args:
        num_classes (int): The number of classes for detection.

    Returns:
        torchvision.models.detection.SSD: The built SSD model.
    """
    result = ssdlite320_mobilenet_v3_large(
        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )
    in_channels = det_utils.retrieve_out_channels(result.backbone, (320, 320))
    num_anchors = result.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    dropout = nn.Dropout(p=0.3)
    result.head.classification_head = nn.Sequential(
        SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer),
        dropout,
    )

    return result


def visualize_results(images, outputs, writer=None, log_dir='logs/'):
    """Visualize the results of object detection.

    Args:
        images (torch.Tensor): The input images.
        outputs (List[Dict[str, torch.Tensor]]): The output predictions from the object detection model.
        writer (SummaryWriter): The SummaryWriter for TensorBoard logging.
        log_dir (str): The directory to save the visualization images.

    Returns:
        None
    """

    dominant_colors = []
    color_names = []
    color = Color()
    for i, (image, output) in enumerate(zip(images, outputs)):
        boxes = output["boxes"].cpu().detach().numpy()
        scores = output["scores"].cpu().detach().numpy()
        labels = output["labels"].cpu().detach().numpy()
        keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), NMS_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        fig, ax = plt.subplots()
        image_np = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(image_np)
        print(f'No of products detected: {boxes[scores > CONFIDENCE_THRESHOLD].shape[0]}')
        for box, score, label in zip(boxes, scores, labels):
            print(f'Label: {label}\tScore: {score}')
            
            x1, y1, x2, y2 = map(int, box)
            
            rgb = color.get_dominant_color(image_np[y1:y2, x1:x2])
            color_name = color.find_closest_color(rgb)
            
            print(f'Color name: {color_name}\tColor RGB: {rgb}')

            # Append detected color information
            dominant_colors.append(rgb)
            color_names.append(color_name)

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="cyan", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color="cyan", fontsize=8)
        fig_id = uuid.uuid4().hex
        fig.savefig(os.path.join(log_dir, f"{fig_id}.png"))
        plt.close(fig)





def test(model, test_dataloader, device, log_dir):
    model.to(device)
    model.eval()
    tb_dir = os.path.join(os.getcwd(), "logs", "test", str(date.today()))
    # writer = SummaryWriter(tb_dir)
    with torch.no_grad():
        all_outputs, all_true_boxes, all_true_labels = [], [], []
        for idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(test_dataloader):
            
            images = images.to(device)
            outputs = model(images)
            
            all_outputs.extend(outputs)
            for i in range(len(images)):
                box = torch.ones((4, x1[i].shape[0]))
                box[0] = x1[i]
                box[1] = y1[i]
                box[2] = x2[i]
                box[3] = y2[i]
                box = box.T
                all_true_boxes.append(box)
                all_true_labels.append(class_id[i])
        visualize_results(images=images, outputs=outputs, log_dir=log_dir)
        visualize_and_save_metrics(
            all_outputs, all_true_boxes, all_true_labels, log_dir
        )

    # writer.close()


def visualize_and_save_metrics(outputs, true_boxes, true_labels, log_dir):
    det_boxes, det_labels, det_scores = [], [], []
    ious = []
    tp, fp, fn = 0, 0, 0

    for output, true_box, true_label in zip(outputs, true_boxes, true_labels):
        det_boxes.append(output["boxes"].cpu().detach())
        det_labels.append(output["labels"].cpu().detach())
        det_scores.append(output["scores"].cpu().detach())

        keep = nms(det_boxes[-1], det_scores[-1], NMS_THRESHOLD)
        det_boxes[-1] = det_boxes[-1][keep]
        det_scores[-1] = det_scores[-1][keep]
        det_labels[-1] = det_labels[-1][keep]

        overlaps = find_jaccard_overlap(det_boxes[-1], true_box)
        max_overlap, _ = torch.max(overlaps, dim=1)
        ious.extend(max_overlap.tolist())

        fn += torch.sum(max_overlap < 0.5).item()
        
    average_precisions, mean_ap, tp, fp = calculate_mAP(
        det_boxes,
        det_labels,
        det_scores,
        true_boxes,
        true_labels,
        [torch.zeros(label.shape, device="cpu") for label in true_labels],
    )  # Assuming no difficulties
    print("\n==================================================================\n")

    print(f"Mean Average precision: {mean_ap}")

    # Save the metrics visualizations
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save the original mAP visualization
    sns.set_theme(style="whitegrid")
    ap_df = pd.DataFrame(
        list(average_precisions.items()), columns=["Class", "Average Precision"]
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Class", y="Average Precision", data=ap_df)
    plt.title(f"Mean Average Precision (mAP): {mean_ap:.4f}")
    plt.savefig(os.path.join(plot_dir, "mAP_visualization.png"))
    plt.close()

    # IoU Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot([iou for iou in ious if iou >= 0.5], bins=100, kde=True, cbar=True)
    plt.xlabel("IoU Score")
    plt.title("Distribution of IoU Scores")
    plt.savefig(os.path.join(plot_dir, "iou_distribution.png"))
    plt.close()

    print(f"mean iou: {sum(ious)/ len(ious)}")

    # TP, FP
    
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=["TP", "FP"], y=[tp.size(0), fp.size(0)])
    plt.title("Number of True Positives, False Positives, and False Negatives")
    plt.savefig(os.path.join(plot_dir, "tp_fp_fn.png"))
    plt.close()
    print(f"number of True positives: {tp.size(0)}!")


def main():
    parser = argparse.ArgumentParser(description="SKU Testing Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--model",
        type=str,
        default="ssd",
        choices=["ssd", "frcnn", "retinanet"],
        help="Type of model to evaluate",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    model = load_model(
        args.model, f"{os.getcwd()}/checkpoints/{args.model}/checkpoint.pth", local_rank
    )

    test_dataset = SKUDataset(split="test", transform=TEST_TRANSFORM)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=6,
        sampler=test_sampler,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    log_dir = os.path.join("train_results", args.model, str(date.today()))
    os.makedirs(log_dir, exist_ok=True)
    test(model, test_dataloader, device, log_dir)
    print("\n\n\n Evaluation ended correctly! \n\n\n")


if __name__ == "__main__":
    main()
