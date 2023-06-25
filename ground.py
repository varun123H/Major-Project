import cv2
import numpy as np
import os

def create_ground_truth_mask(image_path, object_rectangles):
    # Load the image
    image = cv2.imread(image_path)

    # Create an empty mask
    ground_truth_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw filled rectangles on the mask for each object
    for rect in object_rectangles:
        x, y, w, h = rect
        cv2.rectangle(ground_truth_mask, (x, y), (x + w, y + h), 255, -1)

    # Generate a unique filename for the ground truth mask
    #ground_truth_mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_ground_truth_mask.jpg"
    #ground_truth_mask_path = os.path.join("path/to/ground_truth_mask/directory", ground_truth_mask_filename)
    
    # Generate a unique filename for the ground truth mask
    ground_truth_mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_ground_truth_mask.jpg"
    ground_truth_mask_directory = "ground_truth_masks"  # Update with your desired directory name
    ground_truth_mask_path = os.path.join(ground_truth_mask_directory, ground_truth_mask_filename)


    # Save the generated ground truth mask
    cv2.imwrite(ground_truth_mask_path, ground_truth_mask)

    return ground_truth_mask_path

# Provide the path to your image
image_path = "bird.jpg"

# Define the rectangles of objects in the image (x, y, width, height)
object_rectangles = [(50, 50, 100, 100), (50, 50, 100, 100)]

# Generate the ground truth mask
ground_truth_mask_path = create_ground_truth_mask(image_path, object_rectangles)

# Use the generated ground truth mask path in the segmentation code
#segment_image(image_path, ground_truth_mask_path)
