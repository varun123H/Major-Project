import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and transform the input image
image_path = 'tiger.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Pass the image through the model for segmentation
with torch.no_grad():
    predictions = model(image_tensor)

# Get the predicted masks, labels, and scores
masks = predictions[0]['masks'].detach().cpu().numpy()
labels = predictions[0]['labels'].detach().cpu().numpy()
scores = predictions[0]['scores'].detach().cpu().numpy()

# Select masks with high scores
high_score_indices = np.where(scores > 0.5)[0]
masks = masks[high_score_indices]
labels = labels[high_score_indices]
scores = scores[high_score_indices]

# Convert PIL image to NumPy array
image_np = np.array(image)

# Generate random ground truth mask
ground_truth_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
for mask in masks:
    mask = mask[0]  # Extract the mask from its enclosing array
    mask = mask > 0.5  # Apply thresholding to get a binary mask
    ground_truth_mask = np.logical_or(ground_truth_mask, mask)

# Apply segmentation masks on the image
segmented_image = image_np.copy()
for i in range(len(masks)):
    mask = masks[i, 0]
    label = labels[i]
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    alpha = 0.5
    color_mask = np.zeros_like(image_np)
    color_mask[mask > 0.5] = color
    segmented_image = cv2.addWeighted(segmented_image, 1 - alpha, color_mask, alpha, 0)
    segmented_image = cv2.putText(segmented_image, f"Class: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Convert segmented image back to PIL format for display
segmented_image_pil = Image.fromarray(segmented_image)

# Compute pixel accuracy
predicted_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
for mask in masks:
    mask = mask[0]  # Extract the mask from its enclosing array
    mask = mask > 0.5  # Apply thresholding to get a binary mask
    predicted_mask = np.logical_or(predicted_mask, mask)

pixel_accuracy = np.mean(predicted_mask == ground_truth_mask)

# Compute IoU (Intersection over Union)
intersection = np.logical_and(predicted_mask, ground_truth_mask)
union = np.logical_or(predicted_mask, ground_truth_mask)
iou = np.sum(intersection) / np.sum(union)

# Display the images and metrics
image.show()
segmented_image_pil.show()
print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
print(f"IoU: {iou:.4f}")
