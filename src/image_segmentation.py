
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "unit3_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the input image
IMAGE_PATH = DATASET_DIR / "food_photo_1.jpeg"
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}\n"
        "Check the filename inside your dataset folder."
    )

# Convert to RGB for plotting and grayscale for processing
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Binary thresholding using Otsu's method
_, threshold = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 2. Morphological operations
kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(threshold, kernel, iterations=1)
dilation = cv2.dilate(threshold, kernel, iterations=1)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

# 3. Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# 4. Watershed segmentation
# Use Otsu thresholding to create an initial foreground estimate.
foreground = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)[1]

# Remove small noise and find likely foreground regions.
foreground = cv2.morphologyEx(
    foreground, cv2.MORPH_OPEN, kernel, iterations=2
)

background = cv2.dilate(foreground, kernel, iterations=3)

distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)

_, sure_foreground = cv2.threshold(
    distance, 0.35 * distance.max(), 255, 0
)
sure_foreground = sure_foreground.astype(np.uint8)

unknown = cv2.subtract(background, sure_foreground)

_, markers = cv2.connectedComponents(sure_foreground)
markers = markers + 1
markers[unknown == 255] = 0

watershed_image = image.copy()
markers = cv2.watershed(watershed_image, markers)

watershed_result = original.copy()
watershed_result[markers == -1] = [255, 0, 0]

# Save all results
results = {
    "original.jpg": original,
    "grayscale.jpg": gray,
    "binary_threshold.jpg": threshold,
    "erosion.jpg": erosion,
    "dilation.jpg": dilation,
    "opening.jpg": opening,
    "closing.jpg": closing,
    "canny_edges.jpg": edges,
    "watershed.jpg": watershed_result,
}

for filename, result in results.items():
    if result.ndim == 2:
        cv2.imwrite(str(OUTPUT_DIR / filename), result)
    else:
        cv2.imwrite(
            str(OUTPUT_DIR / filename),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
        )

# Create a comparison figure
titles = [
    "Original",
    "Grayscale",
    "Binary Threshold",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing",
    "Canny Edges",
    "Watershed",
]

images = [
    original,
    gray,
    threshold,
    erosion,
    dilation,
    opening,
    closing,
    edges,
    watershed_result,
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for ax, title, result in zip(axes.ravel(), titles, images):
    if result.ndim == 2:
        ax.imshow(result, cmap="gray")
    else:
        ax.imshow(result)

    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()

figure_path = OUTPUT_DIR / "segmentation_comparison.png"
fig.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# Report white-pixel coverage for binary results
print("\nSegmentation Results")
print("-" * 40)

for name, result in {
    "Threshold": threshold,
    "Erosion": erosion,
    "Dilation": dilation,
    "Opening": opening,
    "Closing": closing,
    "Canny edges": edges,
}.items():
    coverage = np.count_nonzero(result == 255) / result.size * 100
    print(f"{name}: {coverage:.2f}% white pixels")

print("\nImage segmentation completed successfully!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Comparison figure: {figure_path}")
