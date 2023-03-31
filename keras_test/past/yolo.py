import cv2
import darknet

# Load the Yolo model
net = darknet.load_net_custom("yolo.cfg", "yolo.weights", 0, 1)

# Load the class names
classes = open("coco.names", "r").read().splitlines()

# Load the input image
image = cv2.imread("input_image.jpg")

# Prepare the input image
sized = cv2.resize(image, (416, 416))
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
sized = sized / 255.0

# Detect objects in the image
results = darknet.detect(net, classes, sized, thresh=0.5)

# Print the detected objects and their coordinates
for result in results:
    print("Object:", result[0], "Confidence:", result[1], "Coordinates:", result[2])