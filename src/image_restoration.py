from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "unit3_restoration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input image
IMAGE_PATH = DATASET_DIR / "food_photo_1.jpeg"

# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}\n"
        "Check that the filename is correct and the image is inside dataset."
    )

# OpenCV loads colour images in BGR format.
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Use a fixed random seed so the noise is reproducible.
rng = np.random.default_rng(42)

# 1. Add Gaussian noise
noise = rng.normal(0, 18, original.shape)
gaussian_noisy = np.clip(
    original.astype(np.float32) + noise, 0, 255
).astype(np.uint8)

# 2. Apply Gaussian filter
gaussian_filtered = cv2.GaussianBlur(
    gaussian_noisy, (5, 5), 0
)

# 3. Add salt-and-pepper noise
salt_pepper = original.copy()
random_values = rng.random(original.shape[:2])
salt_pepper[random_values < 0.025] = 0
salt_pepper[random_values > 0.975] = 255

# 4. Apply median filter
median_filtered = cv2.medianBlur(salt_pepper, 5)

# Save individual results
results = {
    "original.jpg": original,
    "gaussian_noisy.jpg": gaussian_noisy,
    "gaussian_filtered.jpg": gaussian_filtered,
    "salt_pepper_noisy.jpg": salt_pepper,
    "median_filtered.jpg": median_filtered,
}

for filename, result in results.items():
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
    )

# Create a comparison figure
titles = [
    "Original",
    "Gaussian Noise",
    "Gaussian Filtered",
    "Salt-and-Pepper Noise",
    "Median Filtered",
]

images = [
    original,
    gaussian_noisy,
    gaussian_filtered,
    salt_pepper,
    median_filtered,
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for ax, title, result in zip(axes, titles, images):
    ax.imshow(result)
    ax.set_title(title)
    ax.axis("off")

axes[-1].axis("off")
fig.tight_layout()

figure_path = OUTPUT_DIR / "restoration_comparison.png"
fig.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Image restoration completed successfully!")
print(f"Input image: {IMAGE_PATH}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Comparison figure: {figure_path}")
