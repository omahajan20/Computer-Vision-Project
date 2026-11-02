
from pathlib import Path

import cv2
from ultralytics import YOLO

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "unit5_object_recognition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = DATASET_DIR / "road_video.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "object_recognition_output.mp4"
SAMPLE_IMAGE = OUTPUT_DIR / "object_detection_sample.jpg"

# Check that the video exists
if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Check the filename inside the dataset folder."
    )

# Load the pretrained YOLO model.
# The first run may download the model weights.
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

# Load OpenCV's built-in frontal-face detector
cascade_path = (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load the Haar Cascade face detector.")

# Open video
capture = cv2.VideoCapture(str(VIDEO_PATH))

if not capture.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}\n"
        "Check that the video file plays correctly."
    )

fps = capture.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO), fourcc, fps, (width, height)
)

if not writer.isOpened():
    capture.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

frame_count = 0
yolo_detection_count = 0
haar_face_count = 0
sample_saved = False

print(f"Input video: {VIDEO_PATH.name}")
print(f"Video dimensions: {width} x {height}")
print(f"Frames to process: {total_frames}")
print("Processing video. Please wait...")

try:
    while True:
        success, frame = capture.read()

        if not success:
            break

        # YOLO object detection
        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        if results[0].boxes is not None:
            yolo_detection_count += len(results[0].boxes)

        # Haar Cascade face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        haar_face_count += len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                "Haar Face",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(annotated_frame)
        frame_count += 1

        # Save one annotated sample frame
        if not sample_saved:
            cv2.imwrite(str(SAMPLE_IMAGE), annotated_frame)
            sample_saved = True

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

finally:
    capture.release()
    writer.release()

print("\nObject recognition completed!")
print(f"Frames processed: {frame_count}")
print(f"Total YOLO detections across frames: {yolo_detection_count}")
print(f"Total Haar face detections across frames: {haar_face_count}")
print(f"Annotated video saved to: {OUTPUT_VIDEO}")
print(f"Sample image saved to: {SAMPLE_IMAGE}")
