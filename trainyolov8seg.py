from ultralytics import YOLO

model = YOLO('/home/karuppia/Documents/spotdata/yolov8posemodel/train2/weights/best.pt')  # load a pretrained model (recommended for training)

model.train(data='/home/karuppia/Documents/spotdata/config.yaml', epochs=10, imgsz=640)
