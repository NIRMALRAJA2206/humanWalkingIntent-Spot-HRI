import os
import json
import cv2
import numpy as np
import tifffile
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
from detectron2.utils.events import EventStorage
import copy
import math
from torch.amp import autocast, GradScaler  # Correct imports for autocast and GradScaler
import warnings  # For suppressing warnings

############################################
# Suppress Specific Warnings
############################################
# Suppress FutureWarnings related to autocast and GradScaler
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*torch.meshgrid.*",
    category=UserWarning
)

############################################
# User-defined paths
############################################
original_image_dir = "/home/karuppia/Documents/spotdata/data/images"
json_dir = "/home/karuppia/Documents/spotdata/data/keypoints_output"
mask_dir = "/home/karuppia/Documents/spotdata/data/labels"
multi_channel_image_dir = "/home/karuppia/Documents/spotdata/data/multi_channel"
os.makedirs(multi_channel_image_dir, exist_ok=True)

############################################
# Functions to create additional channels
############################################
def create_bbox_mask(image_shape, bboxes):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    for (x1, y1, x2, y2) in bboxes:
        # Ensure coordinates are within image bounds
        x1, x2 = max(int(x1), 0), min(int(x2), image_shape[1] - 1)
        y1, y2 = max(int(y1), 0), min(int(y2), image_shape[0] - 1)
        mask[y1:y2, x1:x2] = 1.0
    return mask

def create_keypoint_heatmaps(image_shape, keypoints, radius=5):
    h, w = image_shape[:2]
    # Combine all persons' keypoints into one channel
    heatmap = np.zeros((h, w), dtype=np.float32)
    for person_kps in keypoints:
        for (kx, ky, vis) in person_kps:
            if vis > 0:  # Consider only visible keypoints
                px = int(kx * w)
                py = int(ky * h)
                cv2.circle(heatmap, (px, py), radius, 1.0, -1)
    return heatmap[..., None]  # shape: HxWx1

############################################
# Preprocessing: Create Multi-Channel TIFF
############################################
image_files = [f for f in os.listdir(original_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Desired dimensions (optional, adjust based on your needs)
target_height = 512
target_width = 512

for img_name in image_files:
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(original_image_dir, img_name)
    json_path = os.path.join(json_dir, base_name + ".json")
    txt_path = os.path.join(mask_dir, base_name + ".txt")

    if not os.path.exists(json_path) or not os.path.exists(txt_path):
        print(f"Skipping {img_name} due to missing annotations.")
        continue  # skip if missing annotations

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to read image {img_path}. Skipping.")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Resize image if necessary
    if h != target_height or w != target_width:
        image = cv2.resize(image, (target_width, target_height))
        h, w = image.shape[:2]

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    bboxes = data.get('bounding_boxes', [])  # [[x1,y1,x2,y2], ...]
    keypoints = data.get('keypoints', [])    # list of person keypoints

    bbox_mask = create_bbox_mask(image.shape, bboxes)
    kp_heatmap = create_keypoint_heatmaps(image.shape, keypoints)

    # Combine channels: RGB(3) + bbox(1) + keypoints(1) = 5 channels
    combined = np.concatenate([image, bbox_mask[..., None], kp_heatmap], axis=-1).astype(np.float32)

    # Normalize RGB channels to [0, 1]
    combined[..., :3] /= 255.0

    # Save TIFF
    tif_path = os.path.join(multi_channel_image_dir, base_name + ".tif")
    tifffile.imwrite(tif_path, combined)

############################################
# Function to load COCO-style annotations
############################################
def load_coco_annotation_for_image(base_name, mask_dir, width, height):
    txt_path = os.path.join(mask_dir, base_name + ".txt")
    if not os.path.exists(txt_path):
        return {"annotations": []}

    with open(txt_path, 'r') as f:
        line = f.read().strip()

    if not line:
        print(f"Skipping empty annotation file: {txt_path}")
        return {"annotations": []}  # Skip empty files

    parts = line.split()
    if len(parts) < 3:
        print(f"Skipping malformed annotation file: {txt_path}")
        return {"annotations": []}  # Skip malformed files

    try:
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
    except ValueError:
        print(f"Skipping non-integer class ID or non-float coordinates in file: {txt_path}")
        return {"annotations": []}

    # Ensure we have an even number of coordinates
    if len(coords) % 2 != 0:
        print(f"Skipping malformed polygon in file: {txt_path}")
        return {"annotations": []}

    polygon_abs = []
    for i in range(0, len(coords), 2):
        x_norm = coords[i]
        y_norm = coords[i + 1]
        x_abs = x_norm * width
        y_abs = y_norm * height
        polygon_abs.extend([x_abs, y_abs])

    xs = polygon_abs[0::2]
    ys = polygon_abs[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

    annotation = {
        "bbox": bbox,
        "bbox_mode": BoxMode.XYWH_ABS,
        "segmentation": [polygon_abs],
        "category_id": class_id
    }

    return {"annotations": [annotation]}

############################################
# Register the dataset
############################################
def get_dataset_dicts(multi_channel_dir, mask_dir):
    dataset_dicts = []
    tiff_files = [f for f in os.listdir(multi_channel_dir) if f.endswith('.tif')]

    for idx, tiff_name in enumerate(tiff_files):
        base_name = os.path.splitext(tiff_name)[0]
        tiff_path = os.path.join(multi_channel_dir, tiff_name)

        # Read TIFF for dimensions
        try:
            multi_channel_img = tifffile.imread(tiff_path)
        except Exception as e:
            print(f"Error reading TIFF file {tiff_path}: {e}. Skipping.")
            continue

        if multi_channel_img.ndim != 3 or multi_channel_img.shape[2] != 5:
            print(f"Unexpected shape {multi_channel_img.shape} for file {tiff_path}. Skipping.")
            continue

        height, width = multi_channel_img.shape[:2]

        ann = load_coco_annotation_for_image(base_name, mask_dir, width, height)

        # Check if there are annotations
        if len(ann["annotations"]) == 0:
            print(f"Warning: No annotations found for {tiff_name}, skipping this image.")
            continue  # Skip this image if it has no annotations

        record = {
            "file_name": tiff_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": ann["annotations"]
        }
        dataset_dicts.append(record)

    print(f"Loaded {len(dataset_dicts)} records from {multi_channel_dir}")
    return dataset_dicts

def register_dataset(split):
    dataset_name = "mydata_" + split
    DatasetCatalog.register(dataset_name, lambda: get_dataset_dicts(multi_channel_image_dir, mask_dir))
    MetadataCatalog.get(dataset_name).thing_classes = ["person"]
    MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id = {0: 0}  # Adjust if more classes

# Register the training dataset
for split in ["train"]:
    register_dataset(split)

# Verify registration
train_dataset = DatasetCatalog.get("mydata_train")
num_images = len(train_dataset)
batch_size = 1  # Reduced from 2 to 1 to save memory
num_batches_per_epoch = math.ceil(num_images / batch_size)
print(f"Dataset registered. Number of records: {num_images}")
mydata_metadata = MetadataCatalog.get("mydata_train")

############################################
# Custom Mapper to load multi-channel TIFF
############################################
def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)

    # Load multi-channel image (5 channels)
    image = tifffile.imread(dataset_dict["file_name"])  # HxWx5
    if image.ndim != 3 or image.shape[2] != 5:
        raise ValueError(f"Unexpected image shape {image.shape} for file {dataset_dict['file_name']}")

    # Transpose to CxHxW
    image = torch.from_numpy(image.transpose(2, 0, 1))  # 5xHxW

    annos = dataset_dict.get("annotations", [])
    if not annos:
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[1:], mask_format="bitmask")
    else:
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:], mask_format="bitmask")

    dataset_dict["image"] = image.float()  # Ensure image is float
    return dataset_dict

############################################
# Model Configuration
############################################
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # Using ResNet-50
cfg.DATASETS.TRAIN = ("mydata_train",)
cfg.DATASETS.TEST = ()  # Add validation dataset if available

# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4

# Solver (optimizer) settings
cfg.SOLVER.IMS_PER_BATCH = batch_size  # Number of images per batch
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate; adjust as needed
cfg.SOLVER.MAX_ITER = 10000  # Increase for actual training
cfg.SOLVER.STEPS = (7000, 9000)  # Learning rate decay steps
cfg.SOLVER.GAMMA = 0.1  # Decay factor

# Enable Mixed Precision Training
cfg.SOLVER.AMP.ENABLED = True

# Model architecture settings
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 'person' class

# Output directory for model checkpoints and logs
cfg.OUTPUT_DIR = "/home/karuppia/Documents/spotdata/data/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Update pixel_mean and pixel_std to accommodate 5 channels
# RGB channels followed by bbox mask and keypoints heatmap
cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406, 0.0, 0.0]
cfg.MODEL.PIXEL_STD = [0.229, 0.224, 0.225, 1.0, 1.0]

# Build the model with the updated configuration
from detectron2.modeling import build_model

model = build_model(cfg)
model.train()

# Correct Weight Initialization
with torch.no_grad():
    first_conv = model.backbone.bottom_up.stem.conv1
    existing_weight = first_conv.weight.data
    existing_bias = first_conv.bias

    # Create new weights with 5 input channels
    new_weight = torch.nn.Parameter(torch.Tensor(first_conv.out_channels, 5, first_conv.kernel_size[0], first_conv.kernel_size[1]))
    torch.nn.init.kaiming_uniform_(new_weight, a=1)

    # Optionally, preserve existing weights for RGB channels
    if existing_weight.shape[1] == 3:
        new_weight[:, :3, :, :] = existing_weight
        # The weights for the additional channels (4th and 5th) remain randomly initialized

    # Assign the new weights to the convolutional layer
    first_conv.weight = new_weight

    # If biases exist and need to be modified, handle them here
    if existing_bias is not None:
        first_conv.bias = existing_bias  # Typically unchanged

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Data Loader with custom mapper
data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)

############################################
# Initialize GradScaler
############################################
scaler = GradScaler()  # Correct GradScaler initialization

############################################
# Training Loop on GPU with EventStorage and AMP
############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to training mode
model.train()

for epoch in range(10):  # Adjust the number of epochs as needed
    epoch_loss = 0.0
    data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)  # Reinitialize at the start of each epoch
    print(f"Starting Epoch [{epoch +1}/10]")
    with EventStorage() as storage:  # Initialize EventStorage at the start of each epoch
        for batch_idx, batch_data in enumerate(data_loader, start=1):
            # Safety check to prevent exceeding expected number of batches
            if batch_idx > num_batches_per_epoch:
                print(f"Batch index {batch_idx} exceeded the expected number of batches {num_batches_per_epoch}. Breaking loop.")
                break  # Prevent stepping beyond one epoch

            # Move data to device
            for d in batch_data:
                d["image"] = d["image"].to(device)
                if "instances" in d:
                    d["instances"] = d["instances"].to(device)

            # Forward pass with autocast
            with autocast(device_type='cuda'):  # Updated autocast usage
                outputs = model(batch_data)
                loss_dict = outputs  # outputs is a dict of individual losses
                losses = sum(loss for loss in loss_dict.values())  # Sum all losses

            # Backward pass and optimization with scaler
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

            # Update EventStorage
            current_iter = epoch * num_batches_per_epoch + batch_idx
            storage.iter = current_iter  # Set iteration number

            if batch_idx % 50 == 0 or batch_idx == num_batches_per_epoch:
                print(
                    f"Epoch [{epoch + 1}/10], Step [{batch_idx}/{num_batches_per_epoch}], Loss: {losses.item():.4f}"
                )
                storage.put_scalar("loss", losses.item())

    avg_epoch_loss = epoch_loss / num_batches_per_epoch
    print(f"Epoch [{epoch + 1}/10] completed with Average Loss: {avg_epoch_loss:.4f}")
    storage.put_scalar("avg_epoch_loss", avg_epoch_loss)

    torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

print("Training Completed!")

############################################
# (Optional) Save the Trained Model
############################################
# Save the trained model's state_dict
torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
print(f"Model saved to {cfg.OUTPUT_DIR}/finalmodeltrain.pth")


# Save the configuration file
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())
print(f"Configuration saved to {cfg.OUTPUT_DIR}/config.yaml")
