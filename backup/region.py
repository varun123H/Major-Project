import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_iou(true_mask, pred_mask):
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_pixel_accuracy(true_mask, pred_mask):
    total_pixels = true_mask.shape[0] * true_mask.shape[1]
    correct_pixels = np.sum(true_mask == pred_mask)
    accuracy = correct_pixels / total_pixels
    return accuracy * 100  # Multiply by 100 to get the percentage

# Load the input image
image = cv2.imread('bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a mask with initialized background and foreground regions
mask = np.zeros(image.shape[:2], dtype=np.uint8)
rect = (50, 50, 300, 500)  # Adjust this rectangle to encompass the foreground object
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)

# Apply GrabCut algorithm to segment the foreground object
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Create a mask where 0 and 2 denote the background and 3 and 1 denote the foreground
segmented_mask = np.where((mask == 0) | (mask == 2), 0, 255).astype(np.uint8)

# Convert the segmented mask to binary masks
masks = [np.where(segmented_mask == 0, 0, 255).astype(np.uint8)]

# Generate a random true mask for demonstration purposes
true_mask = np.random.randint(0, 2, size=image.shape[:2], dtype=np.uint8) * 255

# Calculate IOU and pixel accuracy for each mask
iou_scores = []
pixel_accuracies = []
for mask in masks:
    iou = calculate_iou(true_mask, mask)
    pixel_accuracy = calculate_pixel_accuracy(true_mask, mask)
    iou_scores.append(iou)
    pixel_accuracies.append(pixel_accuracy)

# Display the segmented image and metrics
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(image)
plt.title('Input Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(segmented_mask, cmap='gray')
plt.title('Segmented Mask')
plt.axis('off')

plt.subplot(133)
plt.bar(range(len(masks)), iou_scores, tick_label=['Segment 1'])
plt.ylim(0, 1)
plt.title('IOU Scores')

plt.figure(figsize=(8, 4))
plt.bar(range(len(masks)), pixel_accuracies, tick_label=['Segment 1'])
plt.ylim(0, 100)
plt.title('Pixel Accuracy')
plt.ylabel('Accuracy Percentage')

plt.tight_layout()
plt.show()
