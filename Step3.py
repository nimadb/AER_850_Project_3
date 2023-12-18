from ultralytics import YOLO
from PIL import Image, ImageDraw

# Define paths
model_path = '/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/Step3_results/best.pt'

image_path_1 = '/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/data/evaluation/ardmega.jpg'
image_path_2 = '/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/data/evaluation/arduno.jpg'
image_path_3 = '/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/data/evaluation/rasppi.jpg'

output_folder = '/Users/nima.db/Documents/Fall 2023/AER850/Project 3/AER_850_Project_3/Step3_output'

# Load the YOLO model
model = YOLO(model_path)

# Run batched inference on a list of images
results = model([image_path_1,image_path_2,image_path_3])  # return a list of Results objects

# Run inference on 'bus.jpg' with arguments
model.predict(image_path_1, save=True, imgsz=320, conf=0.5)
model.predict(image_path_2, save=True, imgsz=320, conf=0.5)
model.predict(image_path_3, save=True, imgsz=320, conf=0.5)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

