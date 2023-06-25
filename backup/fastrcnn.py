import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = deeplabv3_resnet50(pretrained=True)
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

# Get the predicted segmentation mask
predicted_mask = predictions['out'][0].argmax(dim=0).detach().cpu().numpy()

# Generate ground truth mask (thresholding example)
threshold = 128
gt_mask = np.array(image.convert("L"))  # Convert to grayscale
gt_mask = np.where(gt_mask >= threshold, 255, 0).astype(np.uint8)

# Calculate IoU
intersection = np.logical_and(predicted_mask, gt_mask)
union = np.logical_or(predicted_mask, gt_mask)
iou = np.sum(intersection) / np.sum(union)

# Calculate pixel accuracy
pixel_accuracy = np.mean(predicted_mask == gt_mask)

# Display the images
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[0].axis("off")
axs[1].imshow(gt_mask, cmap='gray')
axs[1].set_title("Ground Truth Mask")
axs[1].axis("off")
axs[2].imshow(predicted_mask, cmap='gray')
axs[2].set_title("Segmented Image")
axs[2].axis("off")
plt.show()

print("IoU:", iou)
print("Pixel Accuracy:", pixel_accuracy)
