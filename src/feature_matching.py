
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "unit4_feature_matching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH_1 = DATASET_DIR / "food_photo_1.jpeg"
IMAGE_PATH_2 = DATASET_DIR / "food_photo_2.jpeg"


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def find_good_matches(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test filters ambiguous matches.
    good = []

    for pair in pairs:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good.append(first)

    return good


# Load photos
image1 = load_image(IMAGE_PATH_1)
image2 = load_image(IMAGE_PATH_2)

# Convert to grayscale for SIFT
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match original photos
matches_before = find_good_matches(descriptors1, descriptors2)

# Rotate the first photo by 30 degrees
height, width = image1.shape[:2]
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)

rotated_image = cv2.warpAffine(
    image1,
    rotation_matrix,
    (width, height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT,
)

rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

keypoints_rotated, descriptors_rotated = sift.detectAndCompute(
    rotated_gray, None
)

# Match the rotated first photo against the second photo
matches_after = find_good_matches(descriptors_rotated, descriptors2)

# Draw match visualisations
before_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches_before[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

after_image = cv2.drawMatches(
    rotated_image,
    keypoints_rotated,
    image2,
    keypoints2,
    matches_after[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Save output images
cv2.imwrite(str(OUTPUT_DIR / "rotated_photo.jpg"), rotated_image)
cv2.imwrite(str(OUTPUT_DIR / "matches_before_rotation.jpg"), before_image)
cv2.imwrite(str(OUTPUT_DIR / "matches_after_rotation.jpg"), after_image)

# Create a comparison figure
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

axes[0].imshow(cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Before rotation: {len(matches_before)} good matches")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"After 30-degree rotation: {len(matches_after)} good matches")
axes[1].axis("off")

fig.tight_layout()

figure_path = OUTPUT_DIR / "rotation_matching_comparison.png"
fig.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# Save numerical results
summary_path = OUTPUT_DIR / "matching_summary.txt"

with summary_path.open("w", encoding="utf-8") as file:
    file.write("FEATURE MATCHING UNDER ROTATION\n")
    file.write("===============================\n")
    file.write(f"Photo 1: {IMAGE_PATH_1.name}\n")
    file.write(f"Photo 2: {IMAGE_PATH_2.name}\n")
    file.write("Rotation angle: 30 degrees\n")
    file.write(f"Keypoints in photo 1: {len(keypoints1)}\n")
    file.write(f"Keypoints in photo 2: {len(keypoints2)}\n")
    file.write(f"Keypoints in rotated photo 1: {len(keypoints_rotated)}\n")
    file.write(f"Good matches before rotation: {len(matches_before)}\n")
    file.write(f"Good matches after rotation: {len(matches_after)}\n")

print("Feature matching completed successfully!")
print(f"Keypoints in photo 1: {len(keypoints1)}")
print(f"Keypoints in photo 2: {len(keypoints2)}")
print(f"Keypoints in rotated photo 1: {len(keypoints_rotated)}")
print(f"Good matches before rotation: {len(matches_before)}")
print(f"Good matches after rotation: {len(matches_after)}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Comparison figure: {figure_path}")
