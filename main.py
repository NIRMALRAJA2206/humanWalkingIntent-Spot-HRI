import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define paths to your folders
ground_truth_folder = "/home/karuppia/Documents/spotdata/newtrainimages/SegmentationObject"
predicted_folder = "/home/karuppia/Documents/spotdata/compare/model2/masks"

# Initialize metrics
tp_counts = []
fp_counts = []
fn_counts = []
precisions = []
recalls = []
f1_scores = []
ious = []

# Loop through the images in the folder
for filename in os.listdir(ground_truth_folder):
    if filename.endswith(".png"):  # Ensure you're processing image files
        # Load the corresponding ground truth and predicted images
        ground_truth_path = os.path.join(ground_truth_folder, filename)
        predicted_path = os.path.join(predicted_folder, filename.replace(".png", ".jpg"))

        try:
            ground_truth = Image.open(ground_truth_path).convert("L")
            predicted = Image.open(predicted_path).convert("L")
        except Exception as e:
            print(f"Error loading images for {filename}: {e}")
            continue

        # Convert images to numpy arrays
        ground_truth_array = np.array(ground_truth) > 0
        predicted_array = np.array(predicted) > 0

        # Calculate TP, FP, FN
        tp = np.logical_and(ground_truth_array, predicted_array).sum()
        fp = np.logical_and(~ground_truth_array, predicted_array).sum()
        fn = np.logical_and(ground_truth_array, ~predicted_array).sum()

        tp_counts.append(tp)
        fp_counts.append(fp)
        fn_counts.append(fn)

        # Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        # IoU
        union = np.logical_or(ground_truth_array, predicted_array).sum()
        iou = tp / union if union > 0 else 0
        ious.append(iou)

# Aggregate metrics for overall evaluation
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_iou = np.mean(ious)

# Plot 1: Outcome Counts (TP, FP, FN)
plt.figure(figsize=(15, 15))
x = range(1, len(tp_counts) + 1)
plt.bar(x, tp_counts, label="True Positives", color="green")
plt.bar(x, fp_counts, bottom=tp_counts, label="False Positives", color="red")
plt.bar(x, fn_counts, bottom=np.array(tp_counts) + np.array(fp_counts), label="False Negatives", color="orange")
plt.xlabel("Image Index", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Outcome Counts (TP, FP, FN)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Plot 2: Precision, Recall, F1-Score
plt.figure(figsize=(15, 15))
plt.plot(x, precisions, label="Precision", marker="o")
plt.plot(x, recalls, label="Recall", marker="x")
plt.plot(x, f1_scores, label="F1-Score", marker="s")
plt.xlabel("Image Index", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.title("Precision, Recall, F1-Score", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: IoU Histogram
plt.figure(figsize=(15, 15))
plt.hist(ious, bins=10, color="blue", edgecolor="black")
plt.xlabel("IoU", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("IoU Histogram", fontsize=16)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Print Summary for Paper
print("Summary Metrics for Paper:")
print(f"Mean Precision: {mean_precision:.2f}")
print(f"Mean Recall: {mean_recall:.2f}")
print(f"Mean F1-Score: {mean_f1:.2f}")
print(f"Mean IoU: {mean_iou:.2f}")
