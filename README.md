# CCA 3 - Computer Vision Project

## Overview
This project implements six computer vision experiments using Python and OpenCV.

## Technologies
- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO11n
- SIFT and HOG
- Haar Cascade

## Experiments

1. Image Restoration - Gaussian noise, salt-and-pepper noise, Gaussian filtering and median filtering.
2. Image Segmentation - Thresholding, erosion, dilation, opening, closing and Canny edge detection.
3. Feature Extraction - HOG and SIFT descriptors.
4. Feature Matching - SIFT matching before and after 30-degree image rotation.
5. Object Recognition - YOLO11n and Haar Cascade detection.
6. Motion Estimation - Frame differencing, MOG2 background subtraction and Lucas-Kanade optical flow.

## Recorded Results
- Video frames processed: 257
- YOLO detections: 16
- Haar detections: 50
- SIFT good matches before rotation: 217
- SIFT good matches after rotation: 270

## Project Structure
- dataset/ - Input images and video
- src/ - Six Python experiment scripts
- output/ - Generated experiment results
- yolo11n.pt - YOLO model weights

## Installation
Create and activate a virtual environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install opencv-contrib-python==4.11.0.86 numpy matplotlib ultralytics

## Run Experiments

    python src/image_restoration.py
    python src/image_segmentation.py
    python src/feature_extraction.py
    python src/feature_matching.py
    python src/object_recognition.py
    python src/motion_estimation.py

## Report
The detailed project report is included in this repository.

## Note
The .venv directory is excluded from Git.
