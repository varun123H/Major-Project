import cv2
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt

def weakly_supervised_segmentation(image_path, mask):
    # Load the input image
    image = cv2.imread(image_path)
    
    # Resize the ground truth mask to match the image dimensions
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
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
image_path = 'tiger.jpg'

# Load your ground truth mask from a JPEG file
ground_truth_mask = cv2.imread('bird_ground_truth_mask.jpg', cv2.IMREAD_GRAYSCALE)

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
