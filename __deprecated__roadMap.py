import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("palisades.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a stronger Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Detect edges using Canny edge detector with optimized thresholds
edges = cv2.Canny(blurred, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on a minimum length threshold
min_contour_length = 100  # Example threshold
filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=True) > min_contour_length]

# Create a black image of the same size
black_img = np.zeros_like(img)
cv2.drawContours(black_img, filtered_contours, -1, (255, 255, 255), 2)  # Thinner contours

# Crop the image
height, width = black_img.shape[:2]
bar_height = 50  # Example value for bar height
new_height = height - bar_height
cropped_black_img = black_img[0:new_height, 0:width]

# Convert to binary array
arr_img = np.array(cropped_black_img)
bin_arr_img = np.where(arr_img == 255, 1, 0)

# Plot the original and processed images side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].axis("off")
axs[0].set_title("Original Image")

# Processed image
axs[1].imshow(cropped_black_img, cmap='gray', vmin=0, vmax=255)
axs[1].axis("off")
axs[1].set_title("Optimized Processed Image with Contours")

# save image
plt.savefig("processed_image.png")

plt.show()
