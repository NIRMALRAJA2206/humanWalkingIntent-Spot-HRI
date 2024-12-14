import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from ultralytics import YOLO
import mediapipe as mp
import matplotlib.pyplot as plt


# Load and initialize models (same as before)
def load_detectron2_model(config_path, model_weights_path, device='cuda'):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [0.229, 0.224, 0.225, 1.0, 1.0]

    model = build_model(cfg)
    model.eval()
    model.to(device)

    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Detectron2 model loaded successfully!")
    return model, cfg


def initialize_yolov8(model_path='yolov8n.pt'):
    yolo_model = YOLO(model_path)
    print("YOLOv8 model initialized.")
    return yolo_model


def initialize_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5)
    print("Mediapipe Pose estimator initialized.")
    return pose


def detect_humans_yolov8(yolo_model, image_rgb):
    """
    Detects humans in the image using YOLOv8.

    Args:
        yolo_model (YOLO): Loaded YOLOv8 model.
        image_rgb (numpy.ndarray): RGB image.

    Returns:
        human_bboxes (list of tuples): List of bounding boxes (x1, y1, x2, y2).
    """
    results = yolo_model(image_rgb)
    human_bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Get the class ID
            if cls == 0:  # Assuming class 0 is 'person' in YOLOv8
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                human_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    print(f"YOLOv8 detected {len(human_bboxes)} humans.")
    return human_bboxes


def get_keypoints_mediapipe(pose_estimator, image_rgb, human_bboxes):
    """
    Predicts keypoints for each detected human using Mediapipe.

    Args:
        pose_estimator (mp.solutions.pose.Pose): Mediapipe Pose estimator.
        image_rgb (numpy.ndarray): RGB image.
        human_bboxes (list of tuples): List of bounding boxes (x1, y1, x2, y2).

    Returns:
        all_keypoints (list of list of tuples): List containing keypoints for each human.
    """
    all_keypoints = []
    for idx, bbox in enumerate(human_bboxes):
        x1, y1, x2, y2 = bbox
        person_img = image_rgb[y1:y2, x1:x2]
        if person_img.size == 0:
            print(f"Empty region for bbox {bbox}, skipping.")
            all_keypoints.append([])
            continue
        results_pose = pose_estimator.process(person_img)
        if results_pose.pose_landmarks:
            keypoints = []
            for lm in results_pose.pose_landmarks.landmark:
                if lm.visibility < 0.5:
                    keypoints.append((0, 0, 0))  # (x, y, visibility)
                else:
                    x = x1 + int(lm.x * (x2 - x1))
                    y = y1 + int(lm.y * (y2 - y1))
                    keypoints.append((x, y, 1))
            all_keypoints.append(keypoints)
        else:
            print(f"No pose landmarks detected for bbox {bbox}, skipping.")
            all_keypoints.append([])
    print("Mediapipe pose estimation completed.")
    return all_keypoints


def create_bbox_mask(image_shape, human_bboxes):
    """
    Creates a bounding box mask from human bounding boxes.

    Args:
        image_shape (tuple): Shape of the image (height, width, channels).
        human_bboxes (list of tuples): List of bounding boxes (x1, y1, x2, y2).

    Returns:
        bbox_mask (numpy.ndarray): Mask image with bounding boxes filled as 1.0.
    """
    height, width = image_shape[:2]
    bbox_mask = np.zeros((height, width), dtype=np.float32)
    for bbox in human_bboxes:
        x1, y1, x2, y2 = bbox
        bbox_mask[y1:y2, x1:x2] = 1.0
    return bbox_mask


def create_keypoint_heatmap(image_shape, all_keypoints, radius=5):
    """
    Creates a keypoint heatmap from detected keypoints.

    Args:
        image_shape (tuple): Shape of the image (height, width, channels).
        all_keypoints (list of list of tuples): Keypoints for each human.
        radius (int): Radius of the circle to draw for each keypoint.

    Returns:
        kp_heatmap (numpy.ndarray): Heatmap with keypoints drawn as circles.
    """
    height, width = image_shape[:2]
    kp_heatmap = np.zeros((height, width, 1), dtype=np.float32)
    for keypoints in all_keypoints:
        for (x, y, vis) in keypoints:
            if vis > 0:
                cv2.circle(kp_heatmap, (x, y), radius, 1.0, -1)
    return kp_heatmap


def create_multi_channel_image(image_rgb, bbox_mask, kp_heatmap):
    """
    Combines RGB image, bbox mask, and keypoint heatmap into a multi-channel image.

    Args:
        image_rgb (numpy.ndarray): RGB image normalized to [0, 1].
        bbox_mask (numpy.ndarray): Bounding box mask.
        kp_heatmap (numpy.ndarray): Keypoint heatmap.

    Returns:
        combined (numpy.ndarray): Combined multi-channel image.
    """
    rgb_normalized = image_rgb.astype(np.float32) / 255.0  # Ensure normalization
    combined = np.concatenate([rgb_normalized, bbox_mask[..., None], kp_heatmap], axis=-1)  # HxWx5
    return combined


def normalize_image(combined_image, pixel_mean, pixel_std):
    """
    Normalizes the multi-channel image using pixel mean and std.

    Args:
        combined_image (numpy.ndarray): Combined multi-channel image.
        pixel_mean (list): Mean values for each channel.
        pixel_std (list): Standard deviation for each channel.

    Returns:
        normalized (numpy.ndarray): Normalized image.
    """
    pixel_mean = np.array(pixel_mean).reshape(1, 1, -1)
    pixel_std = np.array(pixel_std).reshape(1, 1, -1)
    normalized = (combined_image - pixel_mean) / pixel_std
    return normalized


def prepare_tensor(normalized_image, device):
    """
    Converts the normalized image to a torch tensor.

    Args:
        normalized_image (numpy.ndarray): Normalized multi-channel image.
        device (str): 'cuda' or 'cpu'.

    Returns:
        input_tensor (torch.Tensor): Torch tensor of shape (5, H, W).
    """
    input_tensor = torch.from_numpy(normalized_image.transpose(2, 0, 1)).float().to(device)  # 5xHxW
    return input_tensor


def detectron2_inference(model, input_tensor, image_shape):
    """
    Performs inference using the Detectron2 model.

    Args:
        model (torch.nn.Module): Loaded Detectron2 model.
        input_tensor (torch.Tensor): Input tensor of shape (5, H, W).
        image_shape (tuple): Shape of the original image (height, width).

    Returns:
        instances (Instances): Detected instances with masks.
    """
    with torch.no_grad():
        outputs = model([{"image": input_tensor, "height": image_shape[0], "width": image_shape[1]}])
    instances = outputs[0]["instances"].to("cpu")
    return instances


def visualize_results(image_bgr, instances, human_bboxes, all_keypoints):
    """
    Draws segmentation masks, bounding boxes, and keypoints on the image and displays it.

    Args:
        image_bgr (numpy.ndarray): Original BGR image.
        instances (Instances): Detected instances with masks.
        human_bboxes (list of tuples): Detected bounding boxes.
        all_keypoints (list of list of tuples): Detected keypoints.

    Returns:
        None
    """
    if not instances.has("pred_masks"):
        print("No segmentation masks to display.")
        return

    masks = instances.pred_masks.numpy()  # (N, H, W)
    num_instances = len(masks)
    print(num_instances)

    # Find the mask with the maximum number of True values
    max_area_mask_idx = -1
    max_area = 0
    for idx, mask in enumerate(masks):
        # Count the number of True values (or 1's) in the mask
        area = np.sum(mask)
        if area > max_area:
            max_area = area
            max_area_mask_idx = idx

    # Now max_area_mask_idx points to the mask with the largest area
    if max_area_mask_idx != -1:
        # Get the mask with the maximum area
        mask = masks[max_area_mask_idx]

        # Assign a random color for the mask
        color = [np.random.randint(0, 255) for _ in range(3)]

        # Create a colored mask with the chosen color
        colored_mask = np.zeros_like(image_bgr, dtype=np.uint8)
        colored_mask[:, :, 0] = mask * color[0]  # Red channel
        colored_mask[:, :, 1] = mask * color[1]  # Green channel
        colored_mask[:, :, 2] = mask * color[2]  # Blue channel

        # Overlay the mask on the original image
        alpha = 0.5  # Transparency factor
        image_bgr = cv2.addWeighted(image_bgr, 1, colored_mask, alpha, 0)

    # Draw bounding boxes and keypoints
    for bbox, keypoints in zip(human_bboxes, all_keypoints):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        for (x, y, vis) in keypoints:
            if vis > 0:
                cv2.circle(image_bgr, (x, y), 3, (255, 0, 0), -1)  # Blue keypoints

    return image_bgr



def process_video(input_video_path, config_path, model_weights_path, yolo_model_path='yolov8n.pt'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detectron_model, cfg = load_detectron2_model(config_path, model_weights_path, device)
    yolo_model = initialize_yolov8(yolo_model_path)
    pose_estimator = initialize_mediapipe()

    # Capture video input
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file or camera stream at {input_video_path}")
        return

    # Create a window to display the processed frame
    cv2.namedWindow('Video Stream - Human Detection and Pose Estimation', cv2.WINDOW_NORMAL)

    while True:
        # Read a frame from the video
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Failed to capture frame, ending.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect humans with YOLOv8
        human_bboxes = detect_humans_yolov8(yolo_model, frame_rgb)
        if len(human_bboxes) == 0:
            print("No humans detected in the frame.")
            continue

        # Predict keypoints with Mediapipe
        all_keypoints = get_keypoints_mediapipe(pose_estimator, frame_rgb, human_bboxes)

        # Create masks and keypoint heatmap
        bbox_mask = create_bbox_mask(frame_rgb.shape, human_bboxes)
        kp_heatmap = create_keypoint_heatmap(frame_rgb.shape, all_keypoints, radius=5)

        # Combine into a multi-channel image
        combined_image = create_multi_channel_image(frame_rgb, bbox_mask, kp_heatmap)

        # Normalize the image
        normalized_image = normalize_image(combined_image, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)

        # Prepare the tensor
        input_tensor = prepare_tensor(normalized_image, device)

        # Perform inference with Detectron2
        instances = detectron2_inference(detectron_model, input_tensor, frame_rgb.shape)

        # Visualize results
        x = visualize_results(frame_bgr, instances, human_bboxes, all_keypoints)

        # Display the processed frame in the same window
        cv2.imshow('Video Stream - Human Detection and Pose Estimation', x)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()




# Call the video processing function
input_video_path = "/home/karuppia/Documents/spotdata/right_2024_11_16_17_51_51_spot.mp4"  # Replace with your video path or 0 for webcam input
config_path = "/home/karuppia/Documents/spotdata/data/output/config.yaml"
model_weights_path = "/home/karuppia/Documents/spotdata/data/output/model_final.pth"
yolo_model_path = "yolov8n.pt"  # Replace with your YOLOv8 model path if different

process_video(input_video_path, config_path, model_weights_path, yolo_model_path)
