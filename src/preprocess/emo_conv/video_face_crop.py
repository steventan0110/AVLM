import os
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2



# Load YOLO model from relative path or environment variable
model_path = os.environ.get('YOLO_FACE_MODEL_PATH', './models/yolov8l-face-lindevs.pt')
model = YOLO(model_path, verbose=False)
    
def extract_faces_from_video(video_path, output_path, target_size=(224, 224), model_path="yolov8n-face.pt"):
    """
    Extract faces from a video using YOLOv8, resize to target_size, and save as a new video.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        target_size (tuple): Desired output size of extracted face images.
        model_path (str): Path to the YOLOv8 face detection model.
    """
    
    # Load YOLOv8 face detection model


    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
        return

    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, target_size)

    last_bbox = None  # Store last detected bounding box
    frame_to_write = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        frame_id += 1
        if not ret:
            break

        # Run YOLOv8 face detection
        results = model(frame)
        faces = results[0].boxes if results else []

        if faces:
            # Select the largest face based on bounding box area
            largest_face = max(faces, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_face.xyxy[0])  # Convert bounding box to integers

            last_bbox = (x1, y1, x2, y2)  # Store for future use
        elif last_bbox:
            # If no face is detected, use the last bounding box
            x1, y1, x2, y2 = last_bbox
        else:
            # If no previous bounding box, use the center crop

            x_center, y_center = frame_width // 2, frame_height // 2
            crop_size = min(frame_width, frame_height) // 2
            x1, y1 = x_center - crop_size // 2, y_center - crop_size // 2
            x2, y2 = x1 + crop_size, y1 + crop_size

        # Crop the face
        face_crop = frame[y1:y2, x1:x2]
        # Resize to target size
        resized_face = cv2.resize(face_crop, target_size)
        frame_to_write.append(resized_face)

    assert len(frame_to_write) == total_frames
    # Write to video
    for resized_face in frame_to_write:
        out.write(resized_face)
    
    # Release resources
    cap.release()
    out.release()




if __name__ == "__main__":

    data_file = os.environ.get('DATA_FILE', './data/data.jsonl')
    crop_save_dir = os.environ.get('CROP_SAVE_DIR', './output/cropped_videos')
    os.makedirs(crop_save_dir, exist_ok=True)
    with open(data_file, "r") as f:
        for line in tqdm(f, total=4859):
            data = json.loads(line)
            video_path = data['question_video_path']
            output_path = os.path.join(crop_save_dir, os.path.basename(video_path).replace(".avi", ".mp4"))
            extract_faces_from_video(video_path, output_path)


    