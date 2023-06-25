import cv2
import numpy as np
from influxdb_client import InfluxDBClient

def calculate_iou(true_mask, pred_mask):
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_pixel_accuracy(true_mask, pred_mask):
    total_pixels = true_mask.shape[0] * true_mask.shape[1]
    correct_pixels = np.sum(true_mask == pred_mask)
    accuracy = correct_pixels / total_pixels
    return accuracy * 100

# Load the input image
image = cv2.imread('bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the bounding box for the region of interest (ROI)
roi = (50, 50, 400, 300)  # Example bounding box coordinates (x, y, width, height)

# Create an initial mask with probable foreground and background regions
mask = np.zeros(image.shape[:2], dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)
rect = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Refine the segmentation by iteratively updating the mask
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Generate a random true mask for demonstration purposes
true_mask = np.random.randint(0, 2, size=image.shape[:2], dtype=np.uint8)

# Resize the true mask to match the segmented image dimensions
true_mask = cv2.resize(true_mask, (mask.shape[1], mask.shape[0]))

# Calculate IOU and pixel accuracy
iou = calculate_iou(true_mask, mask)
pixel_accuracy = calculate_pixel_accuracy(true_mask, mask)

# Store metrics in InfluxDB using token-based authentication
token = 'psMVu9OR0PbWeZzYAdKDMCYAKUCSd0ohsz1CLpU0kDuzLkcKL3eWRQBOxEMM6ZH0odO8rvvZbOPPZvQjW_A-Gw=='
org = 'major'
bucket = 'mine'
client = InfluxDBClient(url="http://localhost:8086", token=token)
write_api = client.write_api()

data = [
    {
        "measurement": "image_segmentation",
        "tags": {
            "image_name": "bird.jpg"
        },
        "fields": {
            "iou": iou,
            "pixel_accuracy": pixel_accuracy
        }
    },
    {
        "measurement": "other_measurement",
        "tags": {
            "image_name": "bird.jpg"
        },
        "fields": {
            "field_name": field_value
        }
    }
]


try:
    write_api.write(bucket=bucket, record=data, org=org)
    print("Metrics stored successfully.")
except Exception as e:
    print("Error storing metrics:", e)

# Print the metrics
print("IOU: {:.2f}".format(iou))
print("Pixel Accuracy: {:.2f}%".format(pixel_accuracy))
