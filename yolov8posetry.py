from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp


def find_points_with_y_and_relative_x_numpy(point, points_list):
    ref_x, ref_y = point

    # Convert the points list to a NumPy array
    points_array = np.array(points_list)

    # Filter points with the same y-coordinate
    matching_points = points_array[points_array[:, 1] == ref_y]

    # Calculate absolute differences in x-coordinates
    x_differences = np.abs(matching_points[:, 0] - ref_x)

    # Combine differences and matching points into a list of tuples
    results = list(zip(x_differences, matching_points.tolist()))

    return results

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

p1 = (760, 545)
p2 = (504, 546)

p3 = (727, 430)
p4 = (556, 431)

p5 = (711, 370)
p6 = (584, 371)

p7 = (704, 333)
p8 = (603, 334)

fp_distance = 0
fm_distance = 0
mp_distance = float(calculate_distance(p1, p2))
mm_distance = 0.5 #(~ 0.49 m)
dire = 0
print(mp_distance)

# Load YOLOv8 object detection model (for human detection)
yolo_model = YOLO('yolov8n.pt')

# Load your segmentation model
segmentation_model_path = r'/home/karuppia/Documents/spotdata/runs/segment/train2/weights/best.pt'
segmentation_model = YOLO(segmentation_model_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define 33 different colors for the keypoints
keypoint_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 192, 192), (255, 165, 0), (255, 99, 71),
    (0, 206, 209), (255, 105, 180), (75, 0, 130), (138, 43, 226), (255, 69, 0), (220, 20, 60),
    (255, 99, 71), (255, 140, 0)
]

# Open video file or camera feed
video_path = "/home/karuppia/Documents/spotdata/right_2024_11_16_17_49_37_spot.mp4"
# video_path = "/home/karuppia/Documents/spotdata/val_images/Spot_data_2024_12_08_20_21_15_spot.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Loop through video images
while True:
    d=0
    fp_distance = 0
    # Read a image from the video
    ret, image = cap.read()
    if not ret:
        break  # Break the loop if no image is read
    image = cv2.resize(image, (1280, 720))
    height, width, _ = image.shape


    # Split the image into two halves (left and right)
    # image = image[:, width // 2:]  # Extract right half of the image

    # Process the image with YOLOv8 (Object Detection Model)
    yolo_results = yolo_model(image, conf=0.8)

    # Draw bounding boxes around detected humans on the original image
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Class ID for "person" in COCO dataset is 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                # Crop the image to the bounding box region
                cropped_img = image[y1:y2, x1:x2]

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
                        color = keypoint_colors[idx % len(keypoint_colors)]  # Ensure color is from the list

                        # Draw a circle on the keypoints using the assigned color
                        cv2.circle(cropped_img, (x, y), 3, color, -1)

                # Place the cropped image back in the original image (with pose keypoints)
                image[y1:y2, x1:x2] = cropped_img

    # Perform segmentation on the entire image (with bounding boxes) using the segmentation model
    seg_results = segmentation_model(image)  # Segmentation model inference

    # Check if segmentation masks are found
    # Check if segmentation masks are found
    if seg_results:
        for seg_result in seg_results:
            if seg_result.masks is not None and len(seg_result.masks) > 0:
                # Loop through all masks

                for mask in seg_result.masks.data:
                    # Convert mask to numpy and resize to match the original image dimensions
                    mask_cpu = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_cpu, (image.shape[1], image.shape[0]))

                    # Apply the mask to the image
                    color_broadcasted = np.array((0, 255, 0)) * mask_resized[..., None]  # Green mask
                    image = cv2.addWeighted(image, 1, color_broadcasted.astype('uint8'), 0.5, 0)

                    cv2.circle(image, p1, 1, (0, 0, 255), thickness=1)
                    cv2.circle(image, p2, 1, (0, 0, 255), thickness=1)

                    cv2.circle(image, p3, 1, (0, 0, 255), thickness=1)
                    cv2.circle(image, p4, 1, (0, 0, 255), thickness=1)

                    cv2.circle(image, p5, 1, (0, 0, 255), thickness=1)
                    cv2.circle(image, p6, 1, (0, 0, 255), thickness=1)

                    cv2.circle(image, p7, 1, (0, 0, 255), thickness=1)
                    cv2.circle(image, p8, 1, (0, 0, 255), thickness=1)

                    # Convert the resized mask to a binary format (0 and 255)
                    binary_mask = (mask_resized > 0.5).astype('uint8') * 255

                    # Dilate the mask to enhance edge detection (optional, depends on the mask quality)
                    kernel = np.ones((3, 3), np.uint8)  # Small kernel for dilation
                    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

                    # Find contours (outline points) in the dilated binary mask
                    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # Collect unique vertical and filtered horizontal points
                    filtered_points = []
                    last_x, last_y = None, None  # Track the last added point's coordinates

                    for contour in contours:
                        contour_points = contour.squeeze(axis=1)  # Convert contour to 2D array [(x, y), ...]
                        for i, (x, y) in enumerate(contour_points):
                            if last_x is None or last_y is None:
                                # First point, always add it
                                filtered_points.append((x, y))
                                last_x, last_y = x, y
                                continue

                            # Check for horizontal sequence
                            if y == last_y and (i == 0 or contour_points[i - 1][1] == y):
                                # If the current point continues the horizontal sequence, skip it
                                continue

                            # Add the first point of a new sequence or a vertical/diagonal change
                            filtered_points.append((x, y))
                            last_x, last_y = x, y

                            # Optional: Draw the points on the image for debugging
                            #cv2.circle(image, (x, y), 0, (255, 0, 0), thickness=-1)  # Red points for filtered outline

                    p2_values = sorted(find_points_with_y_and_relative_x_numpy(p2, filtered_points),
                                       key=lambda x: x[1][0], reverse=False)
                    p1_values = sorted(find_points_with_y_and_relative_x_numpy(p1, filtered_points),
                                       key=lambda x: x[1][0], reverse=False)
                    # case 1: 2 points inside the segmented area
                    if (p1_values[0][1][0] <= p1[0] <= p1_values[1][1][0] and p2_values[0][1][0] <= p2[0] <=
                            p2_values[1][1][0]):
                        max_move = max(p1_values[0][0], p1_values[1][0], p2_values[0][0], p2_values[1][0])
                        fp_distance = max_move

                    elif p1_values[0][1][0] <= p1[0] <= p1_values[1][1][0]:
                        fp_distance = p1_values[0][0]
                        d = 1
                    elif p2_values[0][1][0] <= p2[0] <= p2_values[1][1][0]:
                        fp_distance = p2_values[1][0]
                        d = 0
                    fm_distance = (fp_distance * mm_distance) / mp_distance
                    position = (50, 50)  # x, y coordinates for the text
                    # Set font, size, color, and thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (0, 255, 0)  # Green color (B, G, R format)
                    thickness = 2

                    if d == 1:
                        print(fm_distance, "Left")
                        text = "Move " + f"{fm_distance:.4f}" + "m to Left"
                    else:
                        print(fm_distance, "Right")
                        text = "Move " + f"{fm_distance:.4f}" + "m to Right"

                    # Get text size
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                    # Calculate background rectangle coordinates
                    x, y = position
                    background_top_left = (x, y - text_size[1] - 5)  # Add a small margin above the text
                    background_bottom_right = (x + text_size[0] + 5, y + 5)  # Add a small margin below the text

                    # Draw the black rectangle for the background
                    cv2.rectangle(image, background_top_left, background_bottom_right, (0, 0, 0),
                                  -1)  # Black color, filled

                    # Draw the text on top of the black background
                    cv2.putText(image, text, position, font, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

    # Display the image with bounding boxes, pose keypoints, and segmentation output
    cv2.imshow('Segmented and Posed Output', image)

    # Check for key press
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break


cap.release()
cv2.destroyAllWindows()
