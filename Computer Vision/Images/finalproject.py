import cv2
import numpy as np
from scipy.spatial import distance

# Define HSV ranges for detecting traffic light colors
color_ranges = {
    "Red": [(0, 59, 237), (10, 255, 255)],          # First red range (lower red hues)
    "Red_Alt": [(160, 59, 237), (180, 255, 255)],  # Second red range (upper red hues)
    "Yellow": [(15, 150, 200), (40, 255, 255)],    # Narrowed yellow range for traffic lights
    "Green": [(41, 187, 227), (109, 255, 255)]     # Green range
}

# Function to detect traffic lights in an image
def detect_traffic_lights(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the image to HSV color space
    detected_lights = []  # List to store detected traffic light data

    # Dictionary to store masks for debugging
    masks = {}
    # List to store raw detections
    detections = []

    # Loop through the defined color ranges to detect each color
    for color_name, (lower, upper) in color_ranges.items():
        # Create a binary mask for the current color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Save the mask for debugging purposes
        masks[color_name] = mask

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum area threshold (percentage of image size) to filter small noise
        min_area = 0.00005 * (image.shape[0] * image.shape[1])  # 0.005% of image size

        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # Ignore small contours
                # Get the minimum enclosing circle for the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))  # Center of the circle
                radius = int(radius)  # Radius of the circle

                # Filter based on reasonable radius range for traffic lights
                if 4 < radius < 25:
                    # Save the detection as a dictionary with color, center, and radius
                    detections.append({"color": color_name, "center": center, "radius": radius})

    # Resolve conflicts where multiple colors are detected in the same region
    resolved_detections = resolve_color_conflicts(detections)

    # Draw the results on the image
    for det in resolved_detections:
        color = det["color"]
        center = det["center"]
        radius = det["radius"]

        # Draw a circle around the detected light
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        # Label the detected light with its color
        cv2.putText(image, f"{color} Light", (center[0] - 30, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected_lights.append(det)

    return image, masks, detected_lights

# Function to resolve color conflicts (e.g., overlapping detections)
def resolve_color_conflicts(detections):
    min_distance = 30  # Minimum distance to consider lights as overlapping
    resolved = []  # List to store resolved detections

    while detections:
        # Take the first detection
        current = detections.pop(0)
        center1 = current["center"]

        # Check for overlapping detections
        overlapping = []
        for det in detections:
            center2 = det["center"]
            # If two detections are close, consider them overlapping
            if distance.euclidean(center1, center2) < min_distance:
                overlapping.append(det)

        # If overlaps exist, keep the detection with the largest radius
        if overlapping:
            overlapping.append(current)  # Include the current detection
            dominant = max(overlapping, key=lambda x: x["radius"])  # Choose the largest
            resolved.append(dominant)

            # Remove overlapping detections from the list
            detections = [d for d in detections if d not in overlapping]
        else:
            resolved.append(current)  # No overlap, keep the current detection

    return resolved

# Load the test image
image_path = "/Users/gui/Desktop/Python/lights/18.jpeg"  # Path to the image
image = cv2.imread(image_path)  # Read the image

if image is None:
    print("Error: Unable to load image.")
    exit()

# Resize the image for better processing if it's too large
image = cv2.resize(image, (800, 800)) if max(image.shape[:2]) > 800 else image

# Detect traffic lights in the image
processed_image, masks, lights = detect_traffic_lights(image)

# Display masks for debugging
for color_name, mask in masks.items():
    cv2.imshow(f"{color_name} Mask", mask)  # Show the mask for each color

# Display the final result with detections
# Display the final result with detections in full screen
cv2.namedWindow("Traffic Light Detection", cv2.WND_PROP_FULLSCREEN)  # Create a window
cv2.setWindowProperty("Traffic Light Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set to full screen
cv2.imshow("Traffic Light Detection", processed_image)  # Show the image
cv2.waitKey(0)  # Wait for a key press to close the windows
cv2.destroyAllWindows()  # Close all OpenCV windows


# Print details of detected traffic lights
print("Detected Traffic Lights:")
for light in lights:
    print(f"Color: {light['color']}, Center: {light['center']}, Radius: {light['radius']}")
