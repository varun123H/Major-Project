import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def segment_image(image_path, model_path):
    # Load the image
    image = cv2.imread(image_path)
    ideal = (512, 512)
    image = cv2.resize(image, ideal)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the bounding box for the region of interest (ROI)
    roi = (50, 50, 400, 300)  # Example bounding box coordinates (x, y, width, height)

    # Create an initial mask with probable foreground and background regions
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    rect = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Refine the segmentation by iteratively updating the mask
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Convert mask to binary (0 or 1)

    # Calculate the segmentation metrics: IOU and pixel accuracy
    ground_truth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Define the ground truth mask for evaluation (manually or from annotated data)
    # Set pixels inside the region of interest as foreground (1) and outside as background (0)
    ground_truth_mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1

    intersection = np.logical_and(ground_truth_mask, mask)
    union = np.logical_or(ground_truth_mask, mask)
    iou = np.sum(intersection) / np.sum(union)
    pixel_accuracy = np.mean(ground_truth_mask == mask) * 100

    # Load pre-trained object detection model
    model = tf.saved_model.load(model_path)

    # Preprocess the image
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the object detection model
    detections = model(input_tensor)

    # Process the detections
    # Extract the bounding box coordinates and class labels
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Display the original image, segmented image, ground truth mask, and object detection results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')

    axes[2].imshow(ground_truth_mask * 255, cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Object Detection')
    axes[3].axis('off')

    # Draw bounding boxes and class labels for object detections
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])

            class_label = str(classes[i])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, class_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    axes[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

    print("IOU: {:.2f}".format(iou))
    print("Pixel Accuracy: {:.2f}%".format(pixel_accuracy))

# Provide the path to your image
image_path = "bird.jpg"

# Provide the path to the pre-trained model
model_path = "model/saved_model"

# Call the segmentation function
segment_image(image_path, model_path)
