import cv2
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt

def generate_ground_truth_mask(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Generate a random ground truth mask (black and white)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    x = np.random.randint(0, image.shape[1] - 1, size=10)
    y = np.random.randint(0, image.shape[0] - 1, size=10)
    mask[y, x] = 255

    return mask

def weakly_supervised_segmentation(image_path, mask):
    # Load the input image
    image = cv2.imread(image_path)
    
    # Apply SLIC superpixel segmentation
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    
    # Create a mask by assigning superpixels with any overlap with the mask as foreground, others as background
    mask_slic = np.zeros_like(segments, dtype=np.uint8)
    unique_segments = np.unique(segments)
    
    for segment in unique_segments:
        segment_pixels = np.where(segments == segment)
        if np.any(mask[segment_pixels]):
            mask_slic[segment_pixels] = 1

    # Calculate IOU (Intersection over Union)
    intersection = np.logical_and(mask, mask_slic)
    union = np.logical_or(mask, mask_slic)
    iou = np.sum(intersection) / np.sum(union)

    # Calculate pixel accuracy percentage
    pixel_accuracy = np.sum(mask_slic == mask) / (mask.shape[0] * mask.shape[1]) * 100

    return mask_slic, iou, pixel_accuracy

# Example usage
image_path = 'bird.jpg'
#image = Image.open(image_path).convert("RGB")
# Generate random ground truth mask
ground_truth_mask = generate_ground_truth_mask(image_path)
#threshold = 128
#ground_truth_mask = np.array(image.convert("L"))  # Convert to grayscale
#ground_truth_mask = np.where(gt_mask >= threshold, 255, 0).astype(np.uint8)

# Perform weakly supervised segmentation
segmented_mask, iou, pixel_accuracy = weakly_supervised_segmentation(image_path, ground_truth_mask)

# Display the input image, ground truth mask, and segmented mask using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
fig.suptitle('Weakly Supervised Segmentation')

# Display input image
axes[0].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
axes[0].set_title('Input Image')
axes[0].axis('off')

# Display ground truth mask
axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title('Ground Truth Mask')
axes[1].axis('off')

# Display segmented mask
axes[2].imshow(segmented_mask, cmap='gray')
axes[2].set_title('Segmented Mask')
axes[2].axis('off')

plt.show()

# Print IOU and pixel accuracy
print("IOU:", iou)
print("Pixel Accuracy:", pixel_accuracy)
