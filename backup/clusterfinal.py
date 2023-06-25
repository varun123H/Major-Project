import numpy as np
import cv2
from sklearn.cluster import KMeans
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

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Apply k-means clustering
num_clusters = 3  # Adjust this value as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pixels)
segmented_image = kmeans.cluster_centers_[kmeans.labels_]

# Reshape the segmented image back to the original shape
segmented_image = segmented_image.reshape(image.shape)

# Convert the segmented image to binary masks
masks = []
for i in range(num_clusters):
    mask = np.where(kmeans.labels_ == i, 255, 0).astype(np.uint8)
    masks.append(mask)

# Generate a random true mask for demonstration purposes
true_mask = np.random.randint(0, 2, size=image.shape[:2], dtype=np.uint8)

# Resize the true mask to match the segmented image dimensions
true_mask = cv2.resize(true_mask, (segmented_image.shape[1], segmented_image.shape[0]))

# Calculate IOU and pixel accuracy for each mask
iou_scores = []
pixel_accuracies = []
for mask in masks:
    mask = cv2.resize(mask, (true_mask.shape[1], true_mask.shape[0]))
    iou = calculate_iou(true_mask, mask)
    pixel_accuracy = calculate_pixel_accuracy(true_mask, mask)
    iou_scores.append(iou)
    pixel_accuracies.append(pixel_accuracy)

# Display the segmented image, IOU scores, and pixel accuracy percentages
plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.imshow(image)
plt.title('Input Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(segmented_image.astype(np.uint8))
plt.title('Segmented Image')
plt.axis('off')

plt.subplot(133)
plt.bar(range(num_clusters), iou_scores, tick_label=['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.ylim(0, 1)
plt.title('IOU Scores')

plt.figure(figsize=(8, 4))
plt.bar(range(num_clusters), pixel_accuracies, tick_label=['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.ylim(0, 100)
plt.title('Pixel Accuracy')
plt.ylabel('Accuracy Percentage')

plt.tight_layout()
plt.show()
