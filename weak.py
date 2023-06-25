import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(ground_truth_mask, segmented_mask):
    intersection = np.logical_and(ground_truth_mask, segmented_mask)
    union = np.logical_or(ground_truth_mask, segmented_mask)
    iou = np.sum(intersection) / np.sum(union)
    pixel_accuracy = np.mean(ground_truth_mask == segmented_mask) * 100
    interweak=np.sum(intersection)
    uniweak=np.sum(union)
    print(interweak)
    print(uniweak)
    return iou, pixel_accuracy

def segment_image(image_path, ground_truth_mask_path):
    # Load the image
    image = cv2.imread(image_path)

    # Load the ground truth mask
    ground_truth_mask = cv2.imread(ground_truth_mask_path, 0)

    # Create a mask to initialize the GrabCut algorithm
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the region of interest (ROI) using the ground truth mask
    rect = cv2.boundingRect(ground_truth_mask)

    # Apply GrabCut algorithm to refine the segmentation
    cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask where sure foreground and probable foreground are marked as 1
    segmented_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    # Calculate IOU and pixel accuracy
    iou, pixel_accuracy = calculate_metrics(ground_truth_mask, segmented_mask)

    # Display the original image, ground truth mask, and segmented mask using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ground_truth_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(segmented_mask, cmap="gray")
    axes[2].set_title("Segmented Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("IOU: {:.2f}".format(iou))
    print("Pixel Accuracy: {:.2f}%".format(pixel_accuracy))

# Provide the path to your image
image_path = "bird.jpg"

# Provide the path to your ground truth mask
ground_truth_mask_path = "bird_ground_truth_mask.jpg"

# Call the segmentation function
segment_image(image_path, ground_truth_mask_path)
