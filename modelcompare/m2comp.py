from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm  # For displaying a progress bar

# ----------------------------- Configuration -----------------------------

# Paths to models
YOLO_OBJECT_DETECTION_MODEL = 'yolov8n.pt'  # Replace with your YOLOv8 object detection model path
SEGMENTATION_MODEL_PATH = '/home/karuppia/Documents/spotdata/runs/segment/train2/weights/best.pt'  # Replace with your segmentation model path

# Input and Output directories
INPUT_FOLDER = "/home/karuppia/Documents/spotdata/newdata-20241213T211636Z-001/newdata"  # Replace with your input folder path
OUTPUT_MASK_FOLDER = "/home/karuppia/Documents/spotdata/compare/model3/masks"  # Replace with your desired output folder path

# Supported image formats
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Pose Keypoint Colors (33 different colors)
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 192, 192), (255, 165, 0), (255, 99, 71),
    (0, 206, 209), (255, 105, 180), (75, 0, 130), (138, 43, 226), (255, 69, 0), (220, 20, 60),
    (255, 99, 71), (255, 140, 0), (0, 0, 0)  # Added an extra color to make it 33
]

# ----------------------------- Initialization -----------------------------

# Load YOLOv8 object detection model (for human detection)
yolo_model = YOLO(YOLO_OBJECT_DETECTION_MODEL)

# Load your segmentation model
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create output directories if they don't exist
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_FORMATS)]

# ----------------------------- Processing Loop -----------------------------

# Iterate over each image with a progress bar
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(INPUT_FOLDER, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue  # Skip to the next image if current image is not readable

    # Resize image for consistent processing (optional)
    image = cv2.resize(image, (1280, 720))
    height, width, channels = image.shape

    # Create a black background mask
    mask_background = np.zeros((height, width, 3), dtype=np.uint8)

    # --------------------- Object Detection with YOLOv8 ---------------------

    # Process the image with YOLOv8 (Object Detection Model)
    yolo_results = yolo_model(image, conf=0.8)

    # Draw bounding boxes around detected humans on the original image
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Class ID for "person" in COCO dataset is 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                # --------------------- Pose Estimation with MediaPipe ---------------------

                # Crop the image to the bounding box region
                cropped_img = image[y1:y2, x1:x2]

                # Check if the cropped image is valid
                if cropped_img.size == 0:
                    continue  # Skip if the crop is invalid

                # Convert cropped image to RGB (required by MediaPipe)
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                # Process the cropped image with MediaPipe Pose
                results_pose = pose.process(cropped_img_rgb)

                # If pose landmarks are detected, draw the keypoints
                if results_pose.pose_landmarks:
                    for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                        # Get the coordinates of each keypoint
                        x = int(landmark.x * cropped_img.shape[1])
                        y = int(landmark.y * cropped_img.shape[0])

                        # Assign a color for each keypoint
                        color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]  # Ensure color is from the list

                        # Draw a circle on the keypoints using the assigned color
                        cv2.circle(cropped_img, (x, y), 3, color, -1)

                    # Place the cropped image back in the original image (with pose keypoints)
                    image[y1:y2, x1:x2] = cropped_img

    # --------------------- Segmentation with YOLOv8 ---------------------

    # Perform segmentation on the entire image using the segmentation model
    seg_results = segmentation_model(image)  # Segmentation model inference

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

                    # Create a colored mask (e.g., green)
                    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    colored_mask[:, :] = (0, 255, 0)  # Green color

                    # Apply the binary mask to the colored mask
                    mask_applied = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_binary)

                    # Overlay the mask on the black background using alpha blending
                    alpha = 0.5  # Transparency factor
                    mask_background = cv2.addWeighted(mask_background, 1, mask_applied, alpha, 0)

    # --------------------- Saving the Results ---------------------

    # Save the mask image with black background
    mask_output_path = os.path.join(OUTPUT_MASK_FOLDER, f"{image_file}")
    cv2.imwrite(mask_output_path, mask_background)

# ----------------------------- Cleanup -----------------------------

# Release MediaPipe resources
pose.close()

print("Processing completed.")
print(f"Masks are saved in: {OUTPUT_MASK_FOLDER}")
