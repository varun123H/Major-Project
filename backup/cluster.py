import cv2
import numpy as np

def segment_image(image_path, roi):
    # Load the image
    imagee = cv2.imread(image_path)
    ideal = (512,512)
    image = cv2.resize(imagee, ideal) 

    # Create an empty ground truth mask
    ground_truth_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the region of interest (ROI)
    x, y, width, height = roi
    ground_truth_mask[y:y+height, x:x+width] = 1

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert the pixel values to floating point
    pixels = np.float32(pixels)

    # Define the criteria for K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set the number of clusters (segments)
    num_clusters = 3

    # Perform K-Means clustering
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the labels to 8-bit
    labels = np.uint8(labels)

    # Reshape the labels to the original image shape
    segmented_image = labels.reshape((image.shape[0], image.shape[1]))

    # Calculate the segmentation metrics: IOU and pixel accuracy
    intersection = np.logical_and(ground_truth_mask, segmented_image)
    union = np.logical_or(ground_truth_mask, segmented_image)
    iou = np.sum(intersection) / np.sum(union)
    pixel_accuracy = np.mean(ground_truth_mask == segmented_image) * 100

    # Display the original image, ground truth mask, and segmented image
    cv2.imshow("Original Image", image)
    cv2.imshow("Ground Truth Mask", ground_truth_mask * 255)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("IOU: {:.2f}".format(iou))
    print("Pixel Accuracy: {:.2f}%".format(pixel_accuracy))

# Provide the path to your image
image_path = "bird.jpg"

# Define the region of interest (ROI) as (x, y, width, height)
roi = (50, 50, 400, 300)  # Example ROI coordinates

# Call the segmentation function
segment_image(image_path, roi)
