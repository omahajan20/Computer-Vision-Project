
import cv2
import os
import numpy as np

# Define paths
VIDEO_PATH = "dataset/road_video.mp4"
OUTPUT_DIR = "output/unit6_motion_estimation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open the input video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

# Prepare output videos
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

frame_diff_writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "frame_differencing.mp4"),
    fourcc, fps, (width, height), isColor=False
)

mog2_writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "mog2_background_subtraction.mp4"),
    fourcc, fps, (width, height), isColor=False
)

optical_flow_writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "lucas_kanade_optical_flow.mp4"),
    fourcc, fps, (width, height), isColor=True
)

# Create MOG2 background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)

# Lucas-Kanade optical flow settings
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

previous_gray = None
previous_points = None

frame_count = 0

print("Starting motion estimation...")

while True:
    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Frame differencing
    if previous_gray is None:
        frame_difference = np.zeros_like(gray)
    else:
        frame_difference = cv2.absdiff(previous_gray, gray)
        _, frame_difference = cv2.threshold(
            frame_difference, 25, 255, cv2.THRESH_BINARY
        )

    frame_diff_writer.write(frame_difference)

    # 2. MOG2 background subtraction
    foreground_mask = background_subtractor.apply(frame)
    mog2_writer.write(foreground_mask)

    # 3. Lucas-Kanade optical flow
    flow_visualization = frame.copy()

    if previous_gray is None:
        previous_points = cv2.goodFeaturesToTrack(
            gray, mask=None, **feature_params
        )
    elif previous_points is not None and len(previous_points) > 0:
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
            **lk_params
        )

        if next_points is not None and status is not None:
            good_new = next_points[status.flatten() == 1]
            good_old = previous_points[status.flatten() == 1]

            for new_point, old_point in zip(good_new, good_old):
                x_new, y_new = new_point.ravel().astype(int)
                x_old, y_old = old_point.ravel().astype(int)

                cv2.line(
                    flow_visualization,
                    (x_old, y_old),
                    (x_new, y_new),
                    (0, 255, 0),
                    2
                )
                cv2.circle(
                    flow_visualization,
                    (x_new, y_new),
                    3,
                    (0, 0, 255),
                    -1
                )

            previous_points = good_new.reshape(-1, 1, 2)

            if len(previous_points) < 10:
                new_features = cv2.goodFeaturesToTrack(
                    gray, mask=None, **feature_params
                )

                if new_features is not None:
                    previous_points = np.concatenate(
                        (previous_points, new_features), axis=0
                    )

    optical_flow_writer.write(flow_visualization)

    previous_gray = gray.copy()
    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames")

# Release resources
cap.release()
frame_diff_writer.release()
mog2_writer.release()
optical_flow_writer.release()

print("\nMotion estimation completed!")
print(f"Frames processed: {frame_count}")
print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")
print("Output videos:")
print("- frame_differencing.mp4")
print("- mog2_background_subtraction.mp4")
print("- lucas_kanade_optical_flow.mp4")
