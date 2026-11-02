
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "unit4_feature_extraction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input photos
IMAGE_PATH_1 = DATASET_DIR / "food_photo_1.jpeg"
IMAGE_PATH_2 = DATASET_DIR / "food_photo_2.jpeg"


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}\n"
            "Check that the filename is correct."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image1 = load_image(IMAGE_PATH_1)
image2 = load_image(IMAGE_PATH_2)

# Resize copies to a common size for HOG comparison
hog_size = (256, 256)
gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

gray1_hog = cv2.resize(gray1, hog_size)
gray2_hog = cv2.resize(gray2, hog_size)

# 1. Extract HOG descriptors and visualisations
hog_features1, hog_image1 = hog(
    gray1_hog,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    feature_vector=True,
)

hog_features2, hog_image2 = hog(
    gray2_hog,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    feature_vector=True,
)

# 2. Extract SIFT keypoints and descriptors
sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY), None
)
keypoints2, descriptors2 = sift.detectAndCompute(
    cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY), None
)

# Save SIFT keypoint visualisations
sift_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

sift_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# 3. Match SIFT descriptors
good_matches = []

if descriptors1 is not None and descriptors2 is not None:
    if len(descriptors1) >= 2 and len(descriptors2) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Lowe's ratio test
        good_matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]

match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Save output images
cv2.imwrite(
    str(OUTPUT_DIR / "hog_image_1.jpg"),
    hog_image1 * 255 / max(float(hog_image1.max()), 1.0),
)
cv2.imwrite(
    str(OUTPUT_DIR / "hog_image_2.jpg"),
    hog_image2 * 255 / max(float(hog_image2.max()), 1.0),
)
cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_1.jpg"),
    cv2.cvtColor(sift_image1, cv2.COLOR_RGB2BGR),
)
cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_2.jpg"),
    cv2.cvtColor(sift_image2, cv2.COLOR_RGB2BGR),
)
cv2.imwrite(
    str(OUTPUT_DIR / "sift_matches.jpg"),
    cv2.cvtColor(match_image, cv2.COLOR_RGB2BGR),
)

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

panels = [
    ("Photo 1", image1, None),
    ("HOG — Photo 1", hog_image1, "gray"),
    ("SIFT — Photo 1", sift_image1, None),
    ("Photo 2", image2, None),
    ("HOG — Photo 2", hog_image2, "gray"),
    ("SIFT — Photo 2", sift_image2, None),
]

for ax, (title, image, cmap) in zip(axes.ravel(), panels):
    if cmap:
        ax.imshow(image, cmap=cmap)
    else:
        ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "feature_extraction_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Save numerical summary
summary_path = OUTPUT_DIR / "feature_summary.txt"

with summary_path.open("w", encoding="utf-8") as file:
    file.write("FEATURE EXTRACTION SUMMARY\n")
    file.write("==========================\n")
    file.write(f"Photo 1: {IMAGE_PATH_1.name}\n")
    file.write(f"Photo 2: {IMAGE_PATH_2.name}\n")
    file.write(f"Photo 1 dimensions: {image1.shape}\n")
    file.write(f"Photo 2 dimensions: {image2.shape}\n")
    file.write(f"HOG feature length (photo 1): {len(hog_features1)}\n")
    file.write(f"HOG feature length (photo 2): {len(hog_features2)}\n")
    file.write(f"SIFT keypoints (photo 1): {len(keypoints1)}\n")
    file.write(f"SIFT keypoints (photo 2): {len(keypoints2)}\n")
    file.write(f"Good SIFT matches: {len(good_matches)}\n")

print("Feature extraction completed successfully!")
print(f"HOG feature length, photo 1: {len(hog_features1)}")
print(f"HOG feature length, photo 2: {len(hog_features2)}")
print(f"SIFT keypoints, photo 1: {len(keypoints1)}")
print(f"SIFT keypoints, photo 2: {len(keypoints2)}")
print(f"Good SIFT matches: {len(good_matches)}")
print(f"Results saved to: {OUTPUT_DIR}")
