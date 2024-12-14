from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm  # For progress bar

# Load YOLOv8 object detection model (for human detection)
yolo_model = YOLO('yolov8n.pt')

# Load your segmentation model
segmentation_model_path = r'/home/karuppia/Documents/spotdata/yolov8posemodel/train1/weights/best.pt'
segmentation_model = YOLO(segmentation_model_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define input and output directories
input_folder = "/home/karuppia/Documents/spotdata/newdata-20241213T211636Z-001/newdata"  # Replace with your input folder path
output_folder = "/home/karuppia/Documents/spotdata/compare/model1/masks"  # Replace with your desired output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files in the input folder
supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

# Process each image
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue  # Skip to the next image if current image is not readable

    # Resize image for consistent processing (optional)
    image = cv2.resize(image, (1280, 720))
    height, width, _ = image.shape

    # Process the image with YOLOv8 (Object Detection Model)
    yolo_results = yolo_model(image, conf=0.8)

    # Draw bounding boxes around detected humans on the original image
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Class ID for "person" in COCO dataset is 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    # Perform segmentation on the entire image (with bounding boxes) using the segmentation model
    seg_results = segmentation_model(image)  # Segmentation model inference

    # Initialize a black background mask
    mask_background = np.zeros((height, width), dtype=np.uint8)

    # Check if segmentation masks are found
    if seg_results:
        for seg_result in seg_results:
            if seg_result.masks is not None and len(seg_result.masks) > 0:
                # Loop through all masks
                for mask in seg_result.masks.data:
                    # Convert mask to numpy and resize to match the original image dimensions
                    mask_cpu = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_cpu, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Threshold to binary mask

                    # Combine masks if there are multiple
                    mask_background = cv2.bitwise_or(mask_background, mask_binary * 255)

    # Optionally, apply pose estimation (if needed for mask creation)
    # Convert the image to RGB as MediaPipe requires
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)

    if pose_results.pose_landmarks:
        # Draw pose landmarks on the original image (optional)
        mp.solutions.drawing_utils.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # You can also incorporate pose landmarks into the mask if desired

    # Create a 3-channel mask image with black background
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    mask_image[mask_background == 255] = (0, 255, 0)  # Green mask

    # Save the mask image
    output_path = os.path.join(output_folder, f"{image_file}")
    cv2.imwrite(output_path, mask_image)

print("Processing completed. Masks are saved in:", output_folder)
