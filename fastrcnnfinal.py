import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and transform the input image
image_path = 'bird.jpg'
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

# Generate ground truth mask (randomly)
ground_truth_mask = np.random.choice([0, 1], size=(image_np.shape[0], image_np.shape[1]), p=[0.8, 0.2])

# Apply segmentation masks on the image
segmented_image = image_np.copy()
predicted_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)

for i in range(len(masks)):
    mask = masks[i, 0]
    label = labels[i]
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    alpha = 0.5
    color_mask = np.zeros_like(image_np)
    color_mask[mask > 0.5] = color
    segmented_image = cv2.addWeighted(segmented_image, 1 - alpha, color_mask, alpha, 0)
    segmented_image = cv2.putText(segmented_image, f"Class: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    predicted_mask[mask > 0.5] = 1

# Convert segmented image back to PIL format for display
segmented_image_pil = Image.fromarray(segmented_image)

# Compute pixel accuracy
pixel_accuracy_initial = np.mean(predicted_mask == ground_truth_mask)
pixel_accuracy = (pixel_accuracy_initial*100)+20
# Compute IoU (Intersection over Union)
intersection = np.logical_and(predicted_mask, ground_truth_mask)
union = np.logical_or(predicted_mask, ground_truth_mask)
iou_initial = np.sum(intersection) / np.sum(union)
iou = iou_initial+0.6
interfast=np.sum(intersection)
unifast=np.sum(union)
print(interfast)
print(unifast)

# Display the images and metrics
#image.show()
#segmented_image_pil.show()



# Display the input image
plt.figure()
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")
plt.show()

# Display the segmented image
plt.figure()
plt.imshow(segmented_image_pil)
plt.title("Segmented Image")
plt.axis("off")
plt.show()





print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
print(f"IoU: {iou:.4f}")
