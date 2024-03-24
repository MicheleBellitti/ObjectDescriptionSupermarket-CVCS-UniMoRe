import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import argparse

print("Starting clf_evaluate.py \n")

# Parse command line arguments for the path to the predictions JSON file
parser = argparse.ArgumentParser(description="Evaluate the model's performance.")
parser.add_argument("--predictions_path", type=str, required=True, help="Path to the predictions JSON file.")
args = parser.parse_args()

# Load predictions and labels from the specified JSON file
with open(args.predictions_path, "r") as file:
    data = json.load(file)
preds = np.array(data["predictions"])  # Predicted class indices for each sample
labels = np.array(data["labels"])  # True class labels

# Generate and print the classification report
report = classification_report(labels, preds, digits=3)
print("Classification Report:\n", report)

# Determine the directory to save the figures
figs_dir = os.path.dirname(args.predictions_path)
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

# Generate and save the confusion matrix
conf_matrix = confusion_matrix(labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(os.path.join(figs_dir, "confusion_matrix.png"))

# Generate a larger figure size to give more space to each cell
plt.figure(figsize=(20, 20))
# Load your confusion matrix data
# Generate a mask for zero values if you want to hide them
mask = np.zeros_like(conf_matrix, dtype=bool)
mask[conf_matrix == 0] = True
# Plot using seaborn
ax = sns.heatmap(conf_matrix, mask=mask, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', annot_kws={"size": 8})
# Improve the visibility of labels by setting a rotation
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 12, rotation=90)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 12, rotation=0)
# Title and labels
plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "confusion_matrix_big.png"))

# Assuming softmax probabilities for preds_softmax
# If preds are class indices, convert to one-hot encoded form
num_classes = len(np.unique(labels))
labels_one_hot = label_binarize(labels, classes=range(num_classes))
preds_one_hot = label_binarize(preds, classes=range(num_classes))

for i in range(num_classes):  # Iterate over each class
    precision, recall, _ = precision_recall_curve(labels_one_hot[:, i], preds_one_hot[:, i])
    
    # Plot the precision-recall curve for class i
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.', label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Class {i}')
    plt.legend()
    
    # Save each figure in the same directory as the predictions JSON file
    plt.savefig(os.path.join(figs_dir, f"precision_recall_curve_class_{i}.png"))