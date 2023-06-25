import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image_path):
    # Load the image
    imagee = cv2.imread(image_path)
    ideal = (512,512)
    image = cv2.resize(imagee, ideal)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the bounding box for the region of interest (ROI)
    roi = (50,50, 400, 300)  # Example bounding box coordinates (x, y, width, height)

    # Create an initial mask with probable foreground and background regions
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    rect = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Refine the segmentation by iteratively updating the mask
    threshold = 2
    mask = np.where((mask == threshold) | (mask == 0), 0, 1).astype('uint8')  # Convert mask to binary (0 or 1)

    # Calculate the segmentation metrics: IOU and pixel accuracy
    ground_truth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Define the ground truth mask for evaluation (manually or from annotated data)
    # Set pixels inside the region of interest as foreground (1) and outside as background (0)
    ground_truth_mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1

    intersection = np.logical_and(ground_truth_mask, mask)
    union = np.logical_or(ground_truth_mask, mask)
    iou = np.sum(intersection) / np.sum(union)
    pixel_accuracy = np.mean(ground_truth_mask == mask) * 100
    inteq=np.sum(intersection)
    intew=np.sum(union)
    # Save the ground truth mask matrix to a text file
    np.savetxt('ground_truth_mask.txt', ground_truth_mask, fmt='%d')
    np.savetxt('Segmented_mask.txt', mask, fmt='%d')
    # Display the original image, segmented image, and ground truth mask
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')

    axes[2].imshow(ground_truth_mask * 255, cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("IOU: {:.2f}".format(iou))
    print("Pixel Accuracy: {:.2f}%".format(pixel_accuracy))
    print("Intersection sum: {:.2f}".format(inteq))
    print("Union sum: {:.2f}".format(intew))
    

# Provide the path to your image
image_path = "bird.jpg"

# Call the segmentation function
segment_image(image_path)
